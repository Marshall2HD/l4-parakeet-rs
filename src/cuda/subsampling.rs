use super::{SM89_CUBIN, UploadedAot, linear_launch_config, upload_aot};
use crate::artifact::{AotStorage, AotTensor};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

pub(super) const SUBSAMPLING_CUBIN: &[u8] = include_bytes!(env!("PARAKEET_SM89_SUBSAMPLING_CUBIN"));
pub(super) const CHANNELS: usize = 256;
pub(super) const INPUT_FEATURES: usize = 128;
pub(super) const OUTPUT_FEATURES: usize = 16;
pub(super) const MODEL_WIDTH: usize = 1_024;
pub(super) const FLATTENED_WIDTH: usize = CHANNELS * OUTPUT_FEATURES;
pub(super) const TILE_OUTPUT_FRAMES: usize = 256;
const FEATURE_GROUP: usize = 8;
pub(super) const FP8_MIN_FRAMES: usize = 1024;
const FP8_WEIGHT_HALVES: usize = MODEL_WIDTH * FLATTENED_WIDTH / size_of::<f16>();
const FP8_SCALE_HALVES: usize = MODEL_WIDTH * size_of::<f32>() / size_of::<f16>();
const FP8_ACTIVATION_HALVES: usize = TILE_OUTPUT_FRAMES * FLATTENED_WIDTH / size_of::<f16>();
const FP8_PROJECTION_HALVES: usize = FP8_WEIGHT_HALVES
    + FP8_SCALE_HALVES
    + FP8_ACTIVATION_HALVES
    + TILE_OUTPUT_FRAMES * size_of::<f32>() / size_of::<f16>();
const FP8_POINTWISE_WEIGHT_HALVES: usize = CHANNELS * CHANNELS / size_of::<f16>();
const FP8_WORKSPACE_HALVES: usize = FP8_PROJECTION_HALVES
    + FP8_POINTWISE_WEIGHT_HALVES
    + CHANNELS * size_of::<f32>() / size_of::<f16>();

pub(super) struct Fp8Projection<'a> {
    pub quantize: &'a CudaFunction,
    pub linear: &'a CudaFunction,
    pub pointwise_quantize: &'a CudaFunction,
    pub pointwise_linear: &'a CudaFunction,
    pub depthwise_quantize: &'a CudaFunction,
    pub workspace: &'a mut CudaSlice<f16>,
}

#[derive(Debug, Serialize)]
pub struct SubsamplingBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub artifact: String,
    pub artifact_payload_bytes: u64,
    pub artifact_load_seconds: f64,
    pub input_frames: usize,
    pub valid_input_frames: usize,
    pub output_frames: usize,
    pub valid_output_frames: usize,
    pub tile_output_frames: usize,
    pub explicit_workspace_bytes: usize,
    pub warmup_iterations: usize,
    pub warm_iterations: usize,
    pub warm_latency_ms: f64,
    pub correctness_values: usize,
    pub max_abs_error: Option<f32>,
    pub max_scaled_error: Option<f32>,
    pub mean_abs_error: Option<f64>,
    pub rmse: Option<f64>,
    pub reference_rms: Option<f64>,
    pub normalized_rmse: Option<f64>,
}

pub fn benchmark_subsampling(
    device: usize,
    artifact_path: &Path,
    mel_feature_major: &[f32],
    input_frames: usize,
    valid_input_frames: usize,
    reference: Option<&[f32]>,
    warmup_iterations: usize,
    warm_iterations: usize,
) -> Result<SubsamplingBenchmarkReport, Box<dyn Error>> {
    if input_frames == 0
        || valid_input_frames > input_frames
        || mel_feature_major.len() != INPUT_FEATURES * input_frames
        || warmup_iterations == 0
        || warm_iterations == 0
    {
        return Err("invalid subsampling input shape or benchmark iteration count".into());
    }

    let output_frames = subsample_len(input_frames);
    let valid_output_frames = subsample_len(valid_input_frames);
    if let Some(expected) = reference
        && expected.len() != output_frames * MODEL_WIDTH
    {
        return Err(format!(
            "subsampling reference has {} values, expected {}",
            expected.len(),
            output_frames * MODEL_WIDTH
        )
        .into());
    }

    let mut mel_time_major = vec![0.0_f32; mel_feature_major.len()];
    for frame in 0..input_frames {
        for feature in 0..INPUT_FEATURES {
            mel_time_major[frame * INPUT_FEATURES + feature] =
                mel_feature_major[feature * input_frames + frame];
        }
    }

    let uploaded = upload_aot(device, artifact_path)?;
    let (major, minor) = uploaded.context.compute_capability()?;
    let subsampling_module = uploaded
        .context
        .load_module(Ptx::from_binary(SUBSAMPLING_CUBIN.to_vec()))?;
    let first = subsampling_module.load_function("pk_sm89_subsample_first")?;
    let depthwise = subsampling_module.load_function("pk_sm89_subsample_depthwise")?;
    let depthwise_quantize = subsampling_module.load_function("pk_sm89_subsample_depthwise_fp8")?;
    let flatten = subsampling_module.load_function("pk_sm89_subsample_flatten")?;
    let linear_module = uploaded
        .context
        .load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
    let linear = linear_module.load_function("pk_sm89_fp16_linear_epilogue")?;

    let first_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.0.weight",
        AotStorage::Fp16,
    )?;
    let first_bias = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.0.bias",
        AotStorage::Fp16,
    )?;
    let second_depthwise_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.2.weight",
        AotStorage::Fp16,
    )?;
    let second_depthwise_bias = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.2.bias",
        AotStorage::Fp16,
    )?;
    let second_pointwise_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.3.weight",
        AotStorage::Sm89Fp16Linear,
    )?;
    let second_pointwise_bias = fp32_tensor(
        &uploaded,
        "encoder.subsampling.layers.3.bias",
        AotStorage::Sm89Fp32Bias,
    )?;
    let third_depthwise_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.5.weight",
        AotStorage::Fp16,
    )?;
    let third_depthwise_bias = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.5.bias",
        AotStorage::Fp16,
    )?;
    let third_pointwise_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.layers.6.weight",
        AotStorage::Sm89Fp16Linear,
    )?;
    let third_pointwise_bias = fp32_tensor(
        &uploaded,
        "encoder.subsampling.layers.6.bias",
        AotStorage::Sm89Fp32Bias,
    )?;
    let projection_weight = fp16_tensor(
        &uploaded,
        "encoder.subsampling.linear.weight",
        AotStorage::Sm89Fp16Linear,
    )?;
    let projection_bias = fp32_tensor(
        &uploaded,
        "encoder.subsampling.linear.bias",
        AotStorage::Sm89Fp32Bias,
    )?;

    let first_features = INPUT_FEATURES.div_ceil(2);
    let second_features = first_features.div_ceil(2);
    let third_features = second_features.div_ceil(2);
    if third_features != OUTPUT_FEATURES {
        return Err("unexpected subsampling feature width".into());
    }
    let max_second_frames = 2 * TILE_OUTPUT_FRAMES + 1;
    let max_first_frames = 2 * max_second_frames + 1;
    let padded_tile_frames = TILE_OUTPUT_FRAMES.next_multiple_of(16);

    let input = uploaded.stream.clone_htod(&mel_time_major)?;
    let mut first_output = uploaded
        .stream
        .alloc_zeros::<f16>(max_first_frames * first_features * CHANNELS)?;
    let mut second_depthwise = uploaded
        .stream
        .alloc_zeros::<f16>(max_second_frames * second_features * CHANNELS)?;
    let mut second_pointwise = uploaded
        .stream
        .alloc_zeros::<f16>(max_second_frames * second_features * CHANNELS)?;
    let mut third_depthwise = uploaded
        .stream
        .alloc_zeros::<f16>(TILE_OUTPUT_FRAMES * third_features * CHANNELS)?;
    let mut third_pointwise = uploaded
        .stream
        .alloc_zeros::<f16>(TILE_OUTPUT_FRAMES * third_features * CHANNELS)?;
    let mut flattened = uploaded
        .stream
        .alloc_zeros::<f16>(padded_tile_frames * FLATTENED_WIDTH)?;
    let mut tile_projection = uploaded
        .stream
        .alloc_zeros::<f16>(padded_tile_frames * MODEL_WIDTH)?;
    let mut output = uploaded
        .stream
        .alloc_zeros::<f16>(output_frames * MODEL_WIDTH)?;

    let mut fp8 = if output_frames >= FP8_MIN_FRAMES {
        let module = uploaded
            .context
            .load_module(Ptx::from_binary(super::encoder::ENCODER_CUBIN.to_vec()))?;
        Some((
            module.load_function("pk_sm89_quantize_ffn_contract")?,
            module.load_function("pk_sm89_subsample_projection_fp8")?,
            module.load_function("pk_sm89_quantize_rows256")?,
            module.load_function("pk_sm89_subsample_pointwise_fp8")?,
            uploaded.stream.alloc_zeros::<f16>(FP8_WORKSPACE_HALVES)?,
        ))
    } else {
        None
    };
    for _ in 0..warmup_iterations {
        run_subsampling(
            &uploaded.stream,
            &first,
            &depthwise,
            &flatten,
            &linear,
            &input,
            &first_weight,
            &first_bias,
            &second_depthwise_weight,
            &second_depthwise_bias,
            &second_pointwise_weight,
            &second_pointwise_bias,
            &third_depthwise_weight,
            &third_depthwise_bias,
            &third_pointwise_weight,
            &third_pointwise_bias,
            &projection_weight,
            &projection_bias,
            &mut first_output,
            &mut second_depthwise,
            &mut second_pointwise,
            &mut third_depthwise,
            &mut third_pointwise,
            &mut flattened,
            &mut tile_projection,
            &mut output,
            input_frames,
            valid_output_frames,
            fp8.as_mut().map(
                |(quantize, linear, pointwise_quantize, pointwise_linear, workspace)| {
                    Fp8Projection {
                        quantize,
                        linear,
                        pointwise_quantize,
                        pointwise_linear,
                        depthwise_quantize: &depthwise_quantize,
                        workspace,
                    }
                },
            ),
        )?;
    }
    uploaded.stream.synchronize()?;

    let started = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..warm_iterations {
        run_subsampling(
            &uploaded.stream,
            &first,
            &depthwise,
            &flatten,
            &linear,
            &input,
            &first_weight,
            &first_bias,
            &second_depthwise_weight,
            &second_depthwise_bias,
            &second_pointwise_weight,
            &second_pointwise_bias,
            &third_depthwise_weight,
            &third_depthwise_bias,
            &third_pointwise_weight,
            &third_pointwise_bias,
            &projection_weight,
            &projection_bias,
            &mut first_output,
            &mut second_depthwise,
            &mut second_pointwise,
            &mut third_depthwise,
            &mut third_pointwise,
            &mut flattened,
            &mut tile_projection,
            &mut output,
            input_frames,
            valid_output_frames,
            fp8.as_mut().map(
                |(quantize, linear, pointwise_quantize, pointwise_linear, workspace)| {
                    Fp8Projection {
                        quantize,
                        linear,
                        pointwise_quantize,
                        pointwise_linear,
                        depthwise_quantize: &depthwise_quantize,
                        workspace,
                    }
                },
            ),
        )?;
    }
    let ended = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let warm_latency_ms = f64::from(started.elapsed_ms(&ended)?) / warm_iterations as f64;

    let (
        correctness_values,
        max_abs_error,
        max_scaled_error,
        mean_abs_error,
        rmse,
        reference_rms,
        normalized_rmse,
    ) = match reference {
        Some(expected) => {
            let actual = uploaded.stream.clone_dtoh(&output)?;
            let mut maximum = 0.0_f32;
            let mut maximum_scaled = 0.0_f32;
            let mut maximum_scaled_index = 0_usize;
            let mut maximum_scaled_values = (0.0_f32, 0.0_f32);
            let mut sum = 0.0_f64;
            let mut square_sum = 0.0_f64;
            let mut reference_square_sum = 0.0_f64;
            for (index, (&observed, &expected)) in actual.iter().zip(expected).enumerate() {
                let observed = observed.to_f32();
                if !observed.is_finite() {
                    return Err(format!(
                        "subsampling produced a non-finite value at element {index}"
                    )
                    .into());
                }
                let error = (observed - expected).abs();
                maximum = maximum.max(error);
                let scaled = error / (1.0 + expected.abs());
                if scaled > maximum_scaled {
                    maximum_scaled = scaled;
                    maximum_scaled_index = index;
                    maximum_scaled_values = (observed, expected);
                }
                sum += f64::from(error);
                square_sum += f64::from(error).powi(2);
                reference_square_sum += f64::from(expected).powi(2);
            }
            let rmse = (square_sum / actual.len() as f64).sqrt();
            let reference_rms = (reference_square_sum / actual.len() as f64).sqrt();
            let normalized_rmse = rmse / reference_rms;
            if maximum > 2.0 || normalized_rmse > 0.002 {
                return Err(format!(
                    "subsampling parity failed: max absolute {maximum}, normalized RMSE {normalized_rmse}; worst scaled error {maximum_scaled} at element {maximum_scaled_index}: observed {}, expected {}",
                    maximum_scaled_values.0, maximum_scaled_values.1,
                )
                .into());
            }
            (
                actual.len(),
                Some(maximum),
                Some(maximum_scaled),
                Some(sum / actual.len() as f64),
                Some(rmse),
                Some(reference_rms),
                Some(normalized_rmse),
            )
        }
        None => (0, None, None, None, None, None, None),
    };

    let explicit_workspace_bytes = mel_time_major.len() * size_of::<f32>()
        + first_output.len() * size_of::<f16>()
        + second_depthwise.len() * size_of::<f16>()
        + second_pointwise.len() * size_of::<f16>()
        + third_depthwise.len() * size_of::<f16>()
        + third_pointwise.len() * size_of::<f16>()
        + flattened.len() * size_of::<f16>()
        + tile_projection.len() * size_of::<f16>()
        + output.len() * size_of::<f16>()
        + fp8.as_ref().map_or(0, |(_, _, _, _, workspace)| {
            workspace.len() * size_of::<f16>()
        });

    Ok(SubsamplingBenchmarkReport {
        schema_version: 1,
        device,
        name: uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        artifact: artifact_path.display().to_string(),
        artifact_payload_bytes: uploaded.artifact.header.payload_bytes,
        artifact_load_seconds: uploaded.wall_seconds,
        input_frames,
        valid_input_frames,
        output_frames,
        valid_output_frames,
        tile_output_frames: TILE_OUTPUT_FRAMES,
        explicit_workspace_bytes,
        warmup_iterations,
        warm_iterations,
        warm_latency_ms,
        correctness_values,
        max_abs_error,
        max_scaled_error,
        mean_abs_error,
        rmse,
        reference_rms,
        normalized_rmse,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_subsampling(
    stream: &Arc<CudaStream>,
    first: &CudaFunction,
    depthwise: &CudaFunction,
    flatten: &CudaFunction,
    linear: &CudaFunction,
    input: &CudaSlice<f32>,
    first_weight: &CudaView<'_, f16>,
    first_bias: &CudaView<'_, f16>,
    second_depthwise_weight: &CudaView<'_, f16>,
    second_depthwise_bias: &CudaView<'_, f16>,
    second_pointwise_weight: &CudaView<'_, f16>,
    second_pointwise_bias: &CudaView<'_, f32>,
    third_depthwise_weight: &CudaView<'_, f16>,
    third_depthwise_bias: &CudaView<'_, f16>,
    third_pointwise_weight: &CudaView<'_, f16>,
    third_pointwise_bias: &CudaView<'_, f32>,
    projection_weight: &CudaView<'_, f16>,
    projection_bias: &CudaView<'_, f32>,
    first_output: &mut CudaSlice<f16>,
    second_depthwise: &mut CudaSlice<f16>,
    second_pointwise: &mut CudaSlice<f16>,
    third_depthwise: &mut CudaSlice<f16>,
    third_pointwise: &mut CudaSlice<f16>,
    flattened: &mut CudaSlice<f16>,
    tile_projection: &mut CudaSlice<f16>,
    output: &mut CudaSlice<f16>,
    input_frames: usize,
    valid_output_frames: usize,
    mut fp8: Option<Fp8Projection<'_>>,
) -> Result<(), Box<dyn Error>> {
    let first_frames = input_frames.div_ceil(2);
    let second_frames = first_frames.div_ceil(2);
    let output_frames = second_frames.div_ceil(2);
    let first_features = INPUT_FEATURES.div_ceil(2);
    let second_features = first_features.div_ceil(2);
    let threads = 256_u32;

    // Pack within every request. In the pipeline this borrows the idle encoder
    // FFN arena; the weights remain live until all subsampling tiles finish.
    if let Some(fp8) = &mut fp8 {
        let (mut weight_arena, mut remaining) = fp8.workspace.split_at_mut(FP8_WEIGHT_HALVES);
        let mut scale_arena = remaining.slice_mut(..FP8_SCALE_HALVES);
        let mut weights = unsafe {
            weight_arena
                .transmute_mut::<u8>(MODEL_WIDTH * FLATTENED_WIDTH)
                .ok_or("FP8 subsampling weights exceed arena")?
        };
        let mut scales = unsafe {
            scale_arena
                .transmute_mut::<f32>(MODEL_WIDTH)
                .ok_or("FP8 subsampling scales exceed arena")?
        };
        let mut builder = stream.launch_builder(fp8.quantize);
        builder
            .arg(projection_weight)
            .arg(&mut weights)
            .arg(&mut scales);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (MODEL_WIDTH as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }
    if let Some(fp8) = &mut fp8 {
        let mut arena = fp8.workspace.slice_mut(FP8_PROJECTION_HALVES..);
        let (mut weight_arena, mut scale_arena) = arena.split_at_mut(FP8_POINTWISE_WEIGHT_HALVES);
        let mut weights = unsafe {
            weight_arena
                .transmute_mut::<u8>(CHANNELS * CHANNELS)
                .ok_or("FP8 pointwise weights exceed arena")?
        };
        let mut scales = unsafe {
            scale_arena
                .transmute_mut::<f32>(CHANNELS)
                .ok_or("FP8 pointwise scales exceed arena")?
        };
        let mut builder = stream.launch_builder(fp8.pointwise_quantize);
        builder
            .arg(second_pointwise_weight)
            .arg(&mut weights)
            .arg(&mut scales);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (CHANNELS as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }
    for output_start in (0..output_frames).step_by(TILE_OUTPUT_FRAMES) {
        let output_end = (output_start + TILE_OUTPUT_FRAMES).min(output_frames);
        let active_output = output_end - output_start;
        let second_start = (2 * output_start).saturating_sub(1);
        let second_end = (2 * (output_end - 1) + 2).min(second_frames);
        let active_second = second_end - second_start;
        let first_start = (2 * second_start).saturating_sub(1);
        let first_end = (2 * (second_end - 1) + 2).min(first_frames);
        let active_first = first_end - first_start;

        launch_first(
            stream,
            first,
            input,
            first_weight,
            first_bias,
            first_output,
            input_frames,
            first_start,
            active_first,
            first_features,
        )?;
        let second_rows = (active_second * second_features).next_multiple_of(16);
        if let Some(fp8) = &mut fp8 {
            let mut arena = fp8.workspace.slice_mut(FP8_PROJECTION_HALVES..);
            let (mut weight_arena, mut scale_arena) =
                arena.split_at_mut(FP8_POINTWISE_WEIGHT_HALVES);
            let weights = unsafe {
                weight_arena
                    .transmute_mut::<u8>(CHANNELS * CHANNELS)
                    .ok_or("FP8 pointwise weights exceed arena")?
            };
            let scales = unsafe {
                scale_arena
                    .transmute_mut::<f32>(CHANNELS)
                    .ok_or("FP8 pointwise scales exceed arena")?
            };
            // Fused depthwise reads first_output, so its packed output must use
            // the separate second-depthwise arena, not alias the producer input.
            let elements = second_rows * CHANNELS;
            let (mut activation_arena, mut row_scale_arena) =
                second_depthwise.split_at_mut(elements / size_of::<f16>());
            let mut activations = unsafe {
                activation_arena
                    .transmute_mut::<u8>(elements)
                    .ok_or("FP8 pointwise activations exceed depthwise arena")?
            };
            let mut row_scales = unsafe {
                row_scale_arena
                    .transmute_mut::<f32>(second_rows)
                    .ok_or("FP8 pointwise row scales exceed depthwise arena")?
            };
            let first_start_i32 = i32::try_from(first_start)?;
            let first_frames_i32 = i32::try_from(first_frames)?;
            let first_features_i32 = i32::try_from(first_features)?;
            let second_start_i32 = i32::try_from(second_start)?;
            let active_second_i32 = i32::try_from(active_second)?;
            let second_features_i32 = i32::try_from(second_features)?;
            let mut builder = stream.launch_builder(fp8.depthwise_quantize);
            builder
                .arg(&*first_output)
                .arg(second_depthwise_weight)
                .arg(second_depthwise_bias)
                .arg(&mut activations)
                .arg(&first_start_i32)
                .arg(&first_frames_i32)
                .arg(&first_features_i32)
                .arg(&second_start_i32)
                .arg(&active_second_i32)
                .arg(&second_features_i32)
                .arg(&mut row_scales);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (
                        second_features.div_ceil(FEATURE_GROUP) as u32,
                        active_second as u32,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let activations = activations.slice(..);
            let weights = weights.slice(..);
            let scales = scales.slice(..);
            let row_scales = row_scales.slice(..);
            let rows_i32 = i32::try_from(second_rows)?;
            let mut builder = stream.launch_builder(fp8.pointwise_linear);
            builder
                .arg(&activations)
                .arg(&weights)
                .arg(&scales)
                .arg(&row_scales)
                .arg(second_pointwise_bias)
                .arg(&mut *second_pointwise)
                .arg(&rows_i32);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: ((CHANNELS / 128) as u32, second_rows.div_ceil(128) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 32768,
                })
            }?;
        } else {
            launch_depthwise(
                stream,
                depthwise,
                first_output,
                second_depthwise_weight,
                second_depthwise_bias,
                second_depthwise,
                first_start,
                first_frames,
                first_features,
                second_start,
                active_second,
                second_features,
            )?;
            super::launch_linear_view(
                stream,
                linear,
                linear_launch_config(second_rows, CHANNELS),
                second_depthwise,
                second_pointwise_weight,
                Some(second_pointwise_bias),
                &mut second_pointwise.as_view_mut(),
                second_rows,
                CHANNELS,
                CHANNELS,
                None,
                1.0,
                0.0,
                2,
            )?;
        }
        launch_depthwise(
            stream,
            depthwise,
            second_pointwise,
            third_depthwise_weight,
            third_depthwise_bias,
            third_depthwise,
            second_start,
            second_frames,
            second_features,
            output_start,
            active_output,
            OUTPUT_FEATURES,
        )?;
        let third_rows = (active_output * OUTPUT_FEATURES).next_multiple_of(16);
        super::launch_linear_view(
            stream,
            linear,
            linear_launch_config(third_rows, CHANNELS),
            third_depthwise,
            third_pointwise_weight,
            Some(third_pointwise_bias),
            &mut third_pointwise.as_view_mut(),
            third_rows,
            CHANNELS,
            CHANNELS,
            None,
            1.0,
            0.0,
            2,
        )?;

        let padded_output = active_output.next_multiple_of(16);
        let total_flattened = padded_output * FLATTENED_WIDTH;
        let mut builder = stream.launch_builder(flatten);
        let active_output_i32 = i32::try_from(active_output)?;
        let padded_output_i32 = i32::try_from(padded_output)?;
        let output_features_i32 = i32::try_from(OUTPUT_FEATURES)?;
        let output_start_i32 = i32::try_from(output_start)?;
        let valid_output_i32 = i32::try_from(valid_output_frames)?;
        builder
            .arg(&*third_pointwise)
            .arg(&mut *flattened)
            .arg(&active_output_i32)
            .arg(&padded_output_i32)
            .arg(&output_features_i32)
            .arg(&output_start_i32)
            .arg(&valid_output_i32);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (u32::try_from(total_flattened)?.div_ceil(threads), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;

        if let Some(fp8) = &mut fp8 {
            let (mut weight_arena, mut remaining) = fp8.workspace.split_at_mut(FP8_WEIGHT_HALVES);
            let (mut scale_arena, mut remaining) = remaining.split_at_mut(FP8_SCALE_HALVES);
            let (mut activation_arena, mut row_scale_arena) =
                remaining.split_at_mut(FP8_ACTIVATION_HALVES);
            let weights = unsafe {
                weight_arena
                    .transmute_mut::<u8>(MODEL_WIDTH * FLATTENED_WIDTH)
                    .ok_or("FP8 subsampling weights exceed arena")?
            };
            let scales = unsafe {
                scale_arena
                    .transmute_mut::<f32>(MODEL_WIDTH)
                    .ok_or("FP8 subsampling scales exceed arena")?
            };
            let mut activations = unsafe {
                activation_arena
                    .transmute_mut::<u8>(total_flattened)
                    .ok_or("FP8 subsampling activations exceed arena")?
            };
            let mut row_scales = unsafe {
                row_scale_arena
                    .transmute_mut::<f32>(padded_output)
                    .ok_or("FP8 subsampling row scales exceed arena")?
            };
            let mut builder = stream.launch_builder(fp8.quantize);
            builder
                .arg(&*flattened)
                .arg(&mut activations)
                .arg(&mut row_scales);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (padded_output as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let activations = activations.slice(..);
            let weights = weights.slice(..);
            let scales = scales.slice(..);
            let row_scales = row_scales.slice(..);
            let mut builder = stream.launch_builder(fp8.linear);
            builder
                .arg(&activations)
                .arg(&weights)
                .arg(&scales)
                .arg(&row_scales)
                .arg(projection_bias)
                .arg(&mut *tile_projection)
                .arg(&padded_output_i32);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (
                        (MODEL_WIDTH / 128) as u32,
                        padded_output.div_ceil(128) as u32,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 32768,
                })
            }?;
        } else {
            super::launch_linear_view(
                stream,
                linear,
                linear_launch_config(padded_output, MODEL_WIDTH),
                flattened,
                projection_weight,
                Some(projection_bias),
                &mut tile_projection.as_view_mut(),
                padded_output,
                MODEL_WIDTH,
                FLATTENED_WIDTH,
                None,
                1.0,
                0.0,
                0,
            )?;
        }
        let source = tile_projection.slice(..active_output * MODEL_WIDTH);
        let mut destination =
            output.slice_mut(output_start * MODEL_WIDTH..output_end * MODEL_WIDTH);
        stream.memcpy_dtod(&source, &mut destination)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_first(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaSlice<f32>,
    weight: &CudaView<'_, f16>,
    bias: &CudaView<'_, f16>,
    output: &mut CudaSlice<f16>,
    input_frames: usize,
    output_start: usize,
    active_output: usize,
    output_features: usize,
) -> Result<(), Box<dyn Error>> {
    let input_frames = i32::try_from(input_frames)?;
    let output_start = i32::try_from(output_start)?;
    let active_output = i32::try_from(active_output)?;
    let output_features_i32 = i32::try_from(output_features)?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weight)
        .arg(bias)
        .arg(output)
        .arg(&input_frames)
        .arg(&output_start)
        .arg(&active_output)
        .arg(&output_features_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(output_features.div_ceil(FEATURE_GROUP))?,
                u32::try_from(active_output)?,
                1,
            ),
            block_dim: (CHANNELS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_depthwise(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaSlice<f16>,
    weight: &CudaView<'_, f16>,
    bias: &CudaView<'_, f16>,
    output: &mut CudaSlice<f16>,
    input_start: usize,
    input_frames: usize,
    input_features: usize,
    output_start: usize,
    active_output: usize,
    output_features: usize,
) -> Result<(), Box<dyn Error>> {
    let input_start = i32::try_from(input_start)?;
    let input_frames = i32::try_from(input_frames)?;
    let input_features = i32::try_from(input_features)?;
    let output_start = i32::try_from(output_start)?;
    let active_output = i32::try_from(active_output)?;
    let output_features_i32 = i32::try_from(output_features)?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weight)
        .arg(bias)
        .arg(output)
        .arg(&input_start)
        .arg(&input_frames)
        .arg(&input_features)
        .arg(&output_start)
        .arg(&active_output)
        .arg(&output_features_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(output_features.div_ceil(FEATURE_GROUP))?,
                u32::try_from(active_output)?,
                1,
            ),
            block_dim: (CHANNELS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

pub(super) fn fp16_tensor<'a>(
    uploaded: &'a UploadedAot,
    name: &str,
    storage: AotStorage,
) -> Result<CudaView<'a, f16>, Box<dyn Error>> {
    let tensor = tensor(uploaded, name, storage)?;
    let bytes = tensor_bytes(uploaded, tensor)?;
    // SAFETY: AOT validation proves the storage is a 16-bit class, and every
    // tensor begins at a 256-byte-aligned offset.
    unsafe { bytes.transmute::<f16>(usize::try_from(tensor.bytes / 2)?) }
        .ok_or_else(|| format!("{name}: FP16 tensor view exceeds artifact").into())
}

pub(super) fn fp32_tensor<'a>(
    uploaded: &'a UploadedAot,
    name: &str,
    storage: AotStorage,
) -> Result<CudaView<'a, f32>, Box<dyn Error>> {
    let tensor = tensor(uploaded, name, storage)?;
    let bytes = tensor_bytes(uploaded, tensor)?;
    // SAFETY: AOT validation proves the storage is a 32-bit class, and every
    // tensor begins at a 256-byte-aligned offset.
    unsafe { bytes.transmute::<f32>(usize::try_from(tensor.bytes / 4)?) }
        .ok_or_else(|| format!("{name}: FP32 tensor view exceeds artifact").into())
}

fn tensor<'a>(
    uploaded: &'a UploadedAot,
    name: &str,
    storage: AotStorage,
) -> Result<&'a AotTensor, Box<dyn Error>> {
    let tensor = uploaded
        .artifact
        .header
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| format!("artifact does not contain {name}"))?;
    if tensor.storage != storage {
        return Err(format!(
            "{name}: expected {storage:?} storage, got {:?}",
            tensor.storage
        )
        .into());
    }
    Ok(tensor)
}

fn tensor_bytes<'a>(
    uploaded: &'a UploadedAot,
    tensor: &AotTensor,
) -> Result<CudaView<'a, u8>, Box<dyn Error>> {
    let start = usize::try_from(tensor.offset)?;
    let end = start
        .checked_add(usize::try_from(tensor.bytes)?)
        .ok_or("tensor range overflow")?;
    Ok(uploaded.weights.slice(start..end))
}

pub(super) fn subsample_len(input: usize) -> usize {
    input.div_ceil(2).div_ceil(2).div_ceil(2)
}
