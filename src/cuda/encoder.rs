use super::cublas::Cublas;
use super::{
    SM89_CUBIN, UploadedAot, launch_linear_view, linear_large_launch_config, linear_launch_config,
    upload_aot,
};
use crate::artifact::{AotStorage, AotTensor};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use float8::F8E4M3;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::mem::size_of;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

pub(super) const ENCODER_CUBIN: &[u8] = include_bytes!(env!("PARAKEET_SM89_ENCODER_CUBIN"));
pub(super) const MODEL_WIDTH: usize = 1_024;
pub(super) const FF_WIDTH: usize = 4_096;
pub(super) const CONV_EXPANDED: usize = 2_048;
pub(super) const ATTENTION_LEFT: usize = 128;
pub(super) const ATTENTION_RIGHT: usize = 128;
pub(super) const POSITION_ROWS: usize = ATTENTION_LEFT + ATTENTION_RIGHT + 1;
const LARGE_LINEAR_ROWS: usize = 1_024;
const FP8_MAX: f32 = 448.0;
const FP8_INPUT_SCALE: f32 = 8.0 / FP8_MAX;

#[derive(Debug, Serialize)]
pub struct EncoderLayerBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub artifact: String,
    pub first_layer: usize,
    pub layers: usize,
    pub rows: usize,
    pub valid_rows: usize,
    pub padded_rows: usize,
    pub attention_left: usize,
    pub attention_right: usize,
    pub explicit_workspace_bytes: usize,
    pub warmup_iterations: usize,
    pub warm_iterations: usize,
    pub warm_latency_ms: f64,
    pub correctness_values: usize,
    pub max_abs_error: Option<f32>,
    pub mean_abs_error: Option<f64>,
    pub rmse: Option<f64>,
    pub reference_rms: Option<f64>,
    pub normalized_rmse: Option<f64>,
}

pub fn benchmark_encoder_layer(
    device: usize,
    artifact_path: &Path,
    first_layer: usize,
    layers: usize,
    input_f32: &[f32],
    rows: usize,
    valid_rows: usize,
    reference: Option<&[f32]>,
    warmup_iterations: usize,
    warm_iterations: usize,
) -> Result<EncoderLayerBenchmarkReport, Box<dyn Error>> {
    if layers == 0
        || first_layer
            .checked_add(layers)
            .is_none_or(|layer_end| layer_end > 24)
        || rows == 0
        || valid_rows > rows
        || input_f32.len() != rows * MODEL_WIDTH
        || warmup_iterations == 0
        || warm_iterations == 0
    {
        return Err("invalid encoder layer, input shape, or benchmark iterations".into());
    }
    if let Some(expected) = reference
        && expected.len() != input_f32.len()
    {
        return Err("encoder layer reference shape does not match input".into());
    }

    let padded_rows = rows.next_multiple_of(16);
    let mut input_host = vec![f16::ZERO; padded_rows * MODEL_WIDTH];
    for (target, source) in input_host.iter_mut().zip(input_f32) {
        *target = f16::from_f32(*source);
    }
    let position_host = relative_position_encoding();

    let uploaded = upload_aot(device, artifact_path)?;
    let (major, minor) = uploaded.context.compute_capability()?;
    let encoder_module = uploaded
        .context
        .load_module(Ptx::from_binary(ENCODER_CUBIN.to_vec()))?;
    let linear_module = uploaded
        .context
        .load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
    let kernels = EncoderKernels {
        cublas: Cublas::new(uploaded.stream.clone())?,
        layer_norm: encoder_module.load_function("pk_sm89_layer_norm")?,
        layer_norm_pair: encoder_module.load_function("pk_sm89_layer_norm_pair")?,
        layer_norm_quantize_fp8: encoder_module
            .load_function("pk_sm89_layer_norm_quantize_fp8_dynamic")?,
        layer_norm_quantize_int4: encoder_module
            .load_function("pk_sm89_layer_norm_quantize_int4_dynamic")?,
        silu: encoder_module.load_function("pk_sm89_silu_in_place")?,
        ffn_expand: encoder_module.load_function("pk_sm89_ffn_expand_async")?,
        quantize_fp8: encoder_module.load_function("pk_sm89_quantize_fp8_fixed")?,
        quantize_value_int4: encoder_module.load_function("pk_sm89_quantize_value_int4")?,
        quantize_value_int4_register: encoder_module
            .load_function("pk_sm89_quantize_value_int4_register")?,
        quantize_fp8_transpose: encoder_module
            .load_function("pk_sm89_quantize_fp8_fixed_transpose")?,
        ffn_expand_fp8: encoder_module.load_function("pk_sm89_ffn_expand_fp8_async")?,
        ffn_expand_fp8_packed: encoder_module.load_function("pk_sm89_ffn_expand_fp8_packed")?,
        ffn_contract_fp8: encoder_module.load_function("pk_sm89_ffn_contract_fp8")?,
        ffn_expand_int4_sparse: encoder_module.load_function("pk_sm89_ffn_expand_int4_sparse")?,
        ffn_contract_fp8_sparse: encoder_module.load_function("pk_sm89_ffn_contract_fp8_sparse")?,
        quantize_ffn_contract: encoder_module.load_function("pk_sm89_quantize_ffn_contract")?,
        glu: encoder_module.load_function("pk_sm89_glu_masked")?,
        conv_glu_fp8: encoder_module.load_function("pk_sm89_conv_glu_fp8")?,
        quantize_rows1024: encoder_module.load_function("pk_sm89_quantize_rows1024")?,
        depthwise: encoder_module.load_function("pk_sm89_depthwise_batchnorm_silu")?,
        depthwise_pack: encoder_module.load_function("pk_sm89_depthwise_pack")?,
        conv_residual_fp8: encoder_module.load_function("pk_sm89_conv_residual_fp8")?,
        pack_position: encoder_module.load_function("pk_sm89_pack_position_heads")?,
        attention: encoder_module.load_function("pk_sm89_local_relpos_attention")?,
        attention_long: encoder_module.load_function("pk_sm89_local_relpos_attention_tc_scores")?,
        attention_packed: encoder_module.load_function("pk_sm89_local_relpos_attention_packed")?,
        pack_scores: encoder_module.load_function("pk_sm89_pack_score_vectors")?,
        pack_key: encoder_module.load_function("pk_sm89_pack_key_int4")?,
        attention_output_fp8: encoder_module.load_function("pk_sm89_attention_output_fp8")?,
        qkv_fp8: encoder_module.load_function("pk_sm89_qkv_fp8")?,
        linear: linear_module.load_function("pk_sm89_fp16_linear_epilogue")?,
        linear_large: linear_module.load_function("pk_sm89_fp16_linear_epilogue_m64")?,
        qkv: linear_module.load_function("pk_sm89_fp16_qkv")?,
    };
    let weights = (first_layer..first_layer + layers)
        .map(|layer| LayerWeights::load(&uploaded, layer))
        .collect::<Result<Vec<_>, _>>()?;
    let ff1_fp8 =
        QuantizedFfnWeights::load(&uploaded, artifact_path, first_layer..first_layer + layers)?;

    let position_input = uploaded.stream.clone_htod(&position_host)?;
    let mut buffers = LayerBuffers {
        source: Some(uploaded.stream.clone_htod(&input_host)?),
        bounds: None,
        state_a: uploaded.stream.alloc_zeros(padded_rows * MODEL_WIDTH)?,
        state_b: uploaded.stream.alloc_zeros(padded_rows * MODEL_WIDTH)?,
        normalized: uploaded.stream.alloc_zeros(padded_rows * MODEL_WIDTH)?,
        workspace: uploaded.stream.alloc_zeros(padded_rows * FF_WIDTH)?,
        position_cache: uploaded
            .stream
            .alloc_zeros(weights.len() * POSITION_ROWS.next_multiple_of(16) * MODEL_WIDTH)?,
    };
    cache_position_projections(
        &uploaded.stream,
        &kernels.linear,
        &kernels.pack_position,
        &weights,
        &position_input,
        &mut buffers.position_cache,
    )?;
    uploaded.stream.synchronize()?;
    drop(position_input);

    for _ in 0..warmup_iterations {
        run_layers(
            &uploaded.stream,
            &kernels,
            &weights,
            &ff1_fp8.layers,
            None,
            &mut buffers,
            rows,
            valid_rows,
            padded_rows,
        )?;
    }
    uploaded.stream.synchronize()?;
    let started = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..warm_iterations {
        run_layers(
            &uploaded.stream,
            &kernels,
            &weights,
            &ff1_fp8.layers,
            None,
            &mut buffers,
            rows,
            valid_rows,
            padded_rows,
        )?;
    }
    let ended = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let warm_latency_ms = f64::from(started.elapsed_ms(&ended)?) / warm_iterations as f64;

    let (correctness_values, max_abs_error, mean_abs_error, rmse, reference_rms, normalized_rmse) =
        match reference {
            Some(expected) => {
                let actual = uploaded.stream.clone_dtoh(&buffers.state_b)?;
                let mut maximum = 0.0_f32;
                let mut absolute_sum = 0.0_f64;
                let mut square_sum = 0.0_f64;
                let mut reference_square_sum = 0.0_f64;
                for (index, (&observed, &expected)) in actual[..rows * MODEL_WIDTH]
                    .iter()
                    .zip(expected)
                    .enumerate()
                {
                    let observed = observed.to_f32();
                    if !observed.is_finite() {
                        return Err(format!(
                            "encoder layer produced a non-finite value at element {index}"
                        )
                        .into());
                    }
                    let error = f64::from((observed - expected).abs());
                    maximum = maximum.max(error as f32);
                    absolute_sum += error;
                    square_sum += error * error;
                    reference_square_sum += f64::from(expected).powi(2);
                }
                let count = expected.len() as f64;
                let rmse = (square_sum / count).sqrt();
                let reference_rms = (reference_square_sum / count).sqrt();
                let normalized_rmse = rmse / reference_rms;
                if normalized_rmse > 0.03 {
                    return Err(format!(
                        "encoder layer normalized RMSE {normalized_rmse} exceeds 0.03; max error {maximum}"
                    )
                    .into());
                }
                (
                    expected.len(),
                    Some(maximum),
                    Some(absolute_sum / count),
                    Some(rmse),
                    Some(reference_rms),
                    Some(normalized_rmse),
                )
            }
            None => (0, None, None, None, None, None),
        };

    let explicit_workspace_bytes = buffers
        .source
        .as_ref()
        .map_or(0, |source| source.len() * size_of::<f16>())
        + buffers.state_a.len() * size_of::<f16>()
        + buffers.state_b.len() * size_of::<f16>()
        + buffers.normalized.len() * size_of::<f16>()
        + buffers.workspace.len() * size_of::<f16>()
        + buffers.position_cache.len() * size_of::<f16>()
        + ff1_fp8.device_bytes();

    Ok(EncoderLayerBenchmarkReport {
        schema_version: 1,
        device,
        name: uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        artifact: artifact_path.display().to_string(),
        first_layer,
        layers,
        rows,
        valid_rows,
        padded_rows,
        attention_left: ATTENTION_LEFT,
        attention_right: ATTENTION_RIGHT,
        explicit_workspace_bytes,
        warmup_iterations,
        warm_iterations,
        warm_latency_ms,
        correctness_values,
        max_abs_error,
        mean_abs_error,
        rmse,
        reference_rms,
        normalized_rmse,
    })
}

pub(super) struct EncoderKernels {
    pub(super) cublas: Cublas,
    pub(super) layer_norm: CudaFunction,
    pub(super) layer_norm_pair: CudaFunction,
    pub(super) layer_norm_quantize_fp8: CudaFunction,
    pub(super) layer_norm_quantize_int4: CudaFunction,
    pub(super) silu: CudaFunction,
    pub(super) ffn_expand: CudaFunction,
    pub(super) quantize_fp8: CudaFunction,
    pub(super) quantize_value_int4: CudaFunction,
    pub(super) quantize_value_int4_register: CudaFunction,
    pub(super) quantize_fp8_transpose: CudaFunction,
    pub(super) ffn_expand_fp8: CudaFunction,
    pub(super) ffn_expand_fp8_packed: CudaFunction,
    pub(super) ffn_contract_fp8: CudaFunction,
    pub(super) ffn_expand_int4_sparse: CudaFunction,
    pub(super) ffn_contract_fp8_sparse: CudaFunction,
    pub(super) quantize_ffn_contract: CudaFunction,
    pub(super) glu: CudaFunction,
    pub(super) conv_glu_fp8: CudaFunction,
    pub(super) quantize_rows1024: CudaFunction,
    pub(super) depthwise: CudaFunction,
    pub(super) depthwise_pack: CudaFunction,
    pub(super) conv_residual_fp8: CudaFunction,
    pub(super) pack_position: CudaFunction,
    pub(super) attention: CudaFunction,
    pub(super) attention_long: CudaFunction,
    pub(super) attention_packed: CudaFunction,
    pub(super) pack_scores: CudaFunction,
    pub(super) pack_key: CudaFunction,
    pub(super) attention_output_fp8: CudaFunction,
    pub(super) qkv_fp8: CudaFunction,
    pub(super) linear: CudaFunction,
    pub(super) linear_large: CudaFunction,
    pub(super) qkv: CudaFunction,
}

pub(super) struct QuantizedFfnLayer {
    weights: CudaSlice<u8>,
    scales: CudaSlice<f32>,
}

pub(super) struct QuantizedFfnWeights {
    pub(super) layers: Vec<QuantizedFfnLayer>,
}

impl QuantizedFfnWeights {
    pub(super) fn load(
        uploaded: &UploadedAot,
        artifact_path: &Path,
        layers: Range<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::load_for(
            uploaded,
            artifact_path,
            layers,
            "feed_forward1.linear1.weight",
        )
    }

    pub(super) fn load_ff2(
        uploaded: &UploadedAot,
        artifact_path: &Path,
        layers: Range<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        Self::load_for(
            uploaded,
            artifact_path,
            layers,
            "feed_forward2.linear1.weight",
        )
    }

    fn load_for(
        uploaded: &UploadedAot,
        artifact_path: &Path,
        layers: Range<usize>,
        suffix: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let int4 = suffix == "feed_forward2.linear1.weight";
        let packed_width = MODEL_WIDTH / if int4 { 2 } else { 1 };
        let mut packed_layers = Vec::with_capacity(layers.len());
        for layer in layers {
            let name = format!("encoder.layers.{layer}.{suffix}");
            let record = tensor(uploaded, &name, AotStorage::Sm89Fp16Linear)?;
            if record.physical_shape != [FF_WIDTH, MODEL_WIDTH] {
                return Err(format!("{name}: incompatible FFN expansion shape").into());
            }
            let source =
                super::read_artifact_fp16_tensor(artifact_path, &uploaded.artifact, record)?;
            let mut packed = vec![0_u8; FF_WIDTH * packed_width];
            let mut scales = vec![1.0_f32; FF_WIDTH];
            for (row, (values, output)) in source
                .chunks_exact(MODEL_WIDTH)
                .zip(packed.chunks_exact_mut(packed_width))
                .enumerate()
            {
                let scale = values
                    .iter()
                    .map(|value| {
                        let value = value.to_f32();
                        if int4 {
                            (value / 7.0).max(-value / 8.0)
                        } else {
                            value.abs()
                        }
                    })
                    .fold(0.0_f32, f32::max)
                    / if int4 { 1.0 } else { FP8_MAX };
                scales[row] = scale;
                if int4 {
                    for (target, pair) in output.iter_mut().zip(values.chunks_exact(2)) {
                        let a = (pair[0].to_f32() / scale)
                            .round_ties_even()
                            .clamp(-8.0, 7.0) as i8 as u8;
                        let b = (pair[1].to_f32() / scale)
                            .round_ties_even()
                            .clamp(-8.0, 7.0) as i8 as u8;
                        *target = (a & 15) | ((b & 15) << 4);
                    }
                } else {
                    for (target, value) in output.iter_mut().zip(values) {
                        *target = F8E4M3::from_f32(value.to_f32() / scale).to_bits();
                    }
                }
            }
            packed_layers.push(QuantizedFfnLayer {
                weights: uploaded.stream.clone_htod(&packed)?,
                scales: uploaded.stream.clone_htod(&scales)?,
            });
        }
        Ok(Self {
            layers: packed_layers,
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.weights.len() + layer.scales.len() * size_of::<f32>())
            .sum()
    }
}

pub(super) struct LayerBuffers {
    pub(super) source: Option<CudaSlice<f16>>,
    pub(super) bounds: Option<CudaSlice<i32>>,
    pub(super) state_a: CudaSlice<f16>,
    pub(super) state_b: CudaSlice<f16>,
    pub(super) normalized: CudaSlice<f16>,
    pub(super) workspace: CudaSlice<f16>,
    pub(super) position_cache: CudaSlice<f16>,
}

pub(super) fn cache_position_projections(
    stream: &Arc<CudaStream>,
    linear: &CudaFunction,
    pack_position: &CudaFunction,
    weights: &[LayerWeights<'_>],
    position_input: &CudaSlice<f16>,
    position_cache: &mut CudaSlice<f16>,
) -> Result<(), Box<dyn Error>> {
    let position_elements = POSITION_ROWS.next_multiple_of(16) * MODEL_WIDTH;
    let mut row_major = stream.alloc_zeros::<f16>(position_elements)?;
    for (index, weights) in weights.iter().enumerate() {
        launch_linear_view(
            stream,
            linear,
            linear_launch_config(POSITION_ROWS.next_multiple_of(16), MODEL_WIDTH),
            position_input,
            &weights.position,
            None,
            &mut row_major.as_view_mut(),
            POSITION_ROWS.next_multiple_of(16),
            MODEL_WIDTH,
            MODEL_WIDTH,
            None,
            1.0,
            0.0,
            0,
        )?;
        let mut position =
            position_cache.slice_mut(index * position_elements..(index + 1) * position_elements);
        let threads = 256_u32;
        let mut builder = stream.launch_builder(pack_position);
        builder.arg(&row_major).arg(&mut position);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (u32::try_from(position_elements)?.div_ceil(threads), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }
    Ok(())
}

pub(super) fn run_layers(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    weights: &[LayerWeights<'_>],
    ff1_fp8: &[QuantizedFfnLayer],
    ff2_int4: Option<(usize, &[QuantizedFfnLayer])>,
    buffers: &mut LayerBuffers,
    rows: usize,
    valid_rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    if buffers.bounds.is_some() && padded_rows >= LARGE_LINEAR_ROWS {
        return Err("packed batch must stay below the long-form GEMM policy".into());
    }
    let active_elements = padded_rows
        .checked_mul(MODEL_WIDTH)
        .ok_or("encoder active element count overflow")?;
    if let Some(source) = &buffers.source {
        let source = source.slice(..active_elements);
        let mut state_a = buffers.state_a.slice_mut(..active_elements);
        stream.memcpy_dtod(&source, &mut state_a)?;
    }
    for (index, layer_weights) in weights.iter().enumerate() {
        if index > 0 {
            std::mem::swap(&mut buffers.state_a, &mut buffers.state_b);
        }
        let layer_ff2_int4 = ff2_int4.and_then(|(first_layer, layers)| {
            index
                .checked_sub(first_layer)
                .and_then(|layer| layers.get(layer))
        });
        run_layer(
            stream,
            kernels,
            layer_weights,
            weights.get(index + 1),
            ff1_fp8.get(index + 1).is_some(),
            ff1_fp8.get(index),
            layer_ff2_int4,
            buffers,
            index,
            rows,
            valid_rows,
            padded_rows,
        )?;
    }
    Ok(())
}

pub(super) struct LayerWeights<'a> {
    norm_ff1_weight: CudaView<'a, f16>,
    norm_ff1_bias: CudaView<'a, f16>,
    ff1_expand: CudaView<'a, f16>,
    ff1_contract: CudaView<'a, f16>,
    norm_attention_weight: CudaView<'a, f16>,
    norm_attention_bias: CudaView<'a, f16>,
    query: CudaView<'a, f16>,
    key: CudaView<'a, f16>,
    value: CudaView<'a, f16>,
    position: CudaView<'a, f16>,
    attention_output: CudaView<'a, f16>,
    bias_u: CudaView<'a, f16>,
    bias_v: CudaView<'a, f16>,
    norm_conv_weight: CudaView<'a, f16>,
    norm_conv_bias: CudaView<'a, f16>,
    pointwise1: CudaView<'a, f16>,
    depthwise: CudaView<'a, f16>,
    conv_norm_weight: CudaView<'a, f32>,
    conv_norm_bias: CudaView<'a, f32>,
    conv_running_mean: CudaView<'a, f32>,
    conv_running_variance: CudaView<'a, f32>,
    pointwise2: CudaView<'a, f16>,
    norm_ff2_weight: CudaView<'a, f16>,
    norm_ff2_bias: CudaView<'a, f16>,
    ff2_expand: CudaView<'a, f16>,
    ff2_contract: CudaView<'a, f16>,
    norm_out_weight: CudaView<'a, f16>,
    norm_out_bias: CudaView<'a, f16>,
}

impl<'a> LayerWeights<'a> {
    pub(super) fn load(uploaded: &'a UploadedAot, layer: usize) -> Result<Self, Box<dyn Error>> {
        let prefix = format!("encoder.layers.{layer}.");
        let fp16 =
            |suffix: &str, storage| fp16_tensor(uploaded, &(prefix.clone() + suffix), storage);
        let fp32 =
            |suffix: &str| fp32_tensor(uploaded, &(prefix.clone() + suffix), AotStorage::Fp32);
        let linear = AotStorage::Sm89Fp16Linear;
        let plain = AotStorage::Fp16;
        Ok(Self {
            norm_ff1_weight: fp16("norm_feed_forward1.weight", plain)?,
            norm_ff1_bias: fp16("norm_feed_forward1.bias", plain)?,
            ff1_expand: fp16("feed_forward1.linear1.weight", linear)?,
            ff1_contract: fp16("feed_forward1.linear2.weight", linear)?,
            norm_attention_weight: fp16("norm_self_att.weight", plain)?,
            norm_attention_bias: fp16("norm_self_att.bias", plain)?,
            query: fp16("self_attn.q_proj.weight", linear)?,
            key: fp16("self_attn.k_proj.weight", linear)?,
            value: fp16("self_attn.v_proj.weight", linear)?,
            position: fp16("self_attn.relative_k_proj.weight", linear)?,
            attention_output: fp16("self_attn.o_proj.weight", linear)?,
            bias_u: fp16("self_attn.bias_u", plain)?,
            bias_v: fp16("self_attn.bias_v", plain)?,
            norm_conv_weight: fp16("norm_conv.weight", plain)?,
            norm_conv_bias: fp16("norm_conv.bias", plain)?,
            pointwise1: fp16("conv.pointwise_conv1.weight", linear)?,
            depthwise: fp16("conv.depthwise_conv.weight", plain)?,
            conv_norm_weight: fp32("conv.norm.weight")?,
            conv_norm_bias: fp32("conv.norm.bias")?,
            conv_running_mean: fp32("conv.norm.running_mean")?,
            conv_running_variance: fp32("conv.norm.running_var")?,
            pointwise2: fp16("conv.pointwise_conv2.weight", linear)?,
            norm_ff2_weight: fp16("norm_feed_forward2.weight", plain)?,
            norm_ff2_bias: fp16("norm_feed_forward2.bias", plain)?,
            ff2_expand: fp16("feed_forward2.linear1.weight", linear)?,
            ff2_contract: fp16("feed_forward2.linear2.weight", linear)?,
            norm_out_weight: fp16("norm_out.weight", plain)?,
            norm_out_bias: fp16("norm_out.bias", plain)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    weights: &LayerWeights<'_>,
    next_weights: Option<&LayerWeights<'_>>,
    next_ff1_fp8: bool,
    ff1_fp8: Option<&QuantizedFfnLayer>,
    ff2_int4: Option<&QuantizedFfnLayer>,
    buffers: &mut LayerBuffers,
    position_index: usize,
    rows: usize,
    valid_rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let use_large_linear = padded_rows >= LARGE_LINEAR_ROWS;
    let use_long_attention = padded_rows >= 512 && buffers.bounds.is_none();
    let sequence_linear = if use_large_linear {
        &kernels.linear_large
    } else {
        &kernels.linear
    };
    let sequence_launch = |n| {
        if use_large_linear {
            linear_large_launch_config(padded_rows, n)
        } else {
            linear_launch_config(padded_rows, n)
        }
    };
    let packed_ff1 =
        ff1_fp8.is_some() && padded_rows * FF_WIDTH >= MODEL_WIDTH * (FF_WIDTH + size_of::<f32>());
    // Long inter-layer boundaries already produced normalized alongside the
    // residual state. The first layer and short paths still normalize here.
    if padded_rows < 1040 || position_index == 0 {
        launch_norm(
            stream,
            &kernels.layer_norm,
            &buffers.state_a,
            &weights.norm_ff1_weight,
            &weights.norm_ff1_bias,
            &mut buffers.normalized,
            rows,
            padded_rows,
        )?;
    }
    {
        let input = buffers.normalized.slice(..);
        let mut output = buffers.workspace.slice_mut(..);
        if let Some(ff1_fp8) = ff1_fp8 {
            let quantized_elements = padded_rows * MODEL_WIDTH;
            let mut quantized_arena = unsafe {
                buffers
                    .state_b
                    .transmute_mut::<u8>(quantized_elements)
                    .ok_or("FP8 activation view exceeds encoder state buffer")?
            };
            let quantized = if padded_rows >= 1040 && position_index > 0 {
                // The previous layer emitted these bytes after the same FP16
                // rounding and fixed-scale conversion as the standalone pack.
                unsafe {
                    input
                        .transmute::<u8>(quantized_elements)
                        .ok_or("packed normalized input exceeds arena")?
                }
            } else {
                launch_quantize_fp8(
                    stream,
                    &kernels.quantize_fp8,
                    &input,
                    &mut quantized_arena,
                    quantized_elements,
                )?;
                quantized_arena.slice(..)
            };
            launch_ffn_expand_fp8(
                stream,
                if packed_ff1 {
                    &kernels.ffn_expand_fp8_packed
                } else {
                    &kernels.ffn_expand_fp8
                },
                &quantized,
                &ff1_fp8.weights,
                &ff1_fp8.scales,
                None,
                &mut output,
                padded_rows,
            )?;
        } else {
            launch_ffn_expand(
                stream,
                &kernels.ffn_expand,
                &input,
                &weights.ff1_expand,
                &mut output,
                padded_rows,
            )?;
        }
    }
    if packed_ff1 {
        launch_ffn_contract_fp8(
            stream,
            kernels,
            &weights.ff1_contract,
            buffers,
            padded_rows,
            false,
        )?;
    } else {
        let input = buffers.workspace.slice(..);
        let mut output = buffers.state_a.slice_mut(..);
        launch_linear_arena(
            stream,
            &kernels.cublas,
            &kernels.silu,
            sequence_linear,
            sequence_launch(MODEL_WIDTH),
            &input,
            &weights.ff1_contract,
            None,
            &mut output,
            padded_rows,
            MODEL_WIDTH,
            FF_WIDTH,
            None,
            0.5,
            1.0,
            0,
        )?;
    }

    let row_values = padded_rows * MODEL_WIDTH;
    // The fused FP8 projection needs the per-request weight arena to fit
    // in the workspace alongside query, key and row scales.
    let use_fused_qkv = padded_rows >= 1040
        && 3 * padded_rows * MODEL_WIDTH
            + padded_rows * size_of::<f32>() / size_of::<f16>()
            + 3 * MODEL_WIDTH * MODEL_WIDTH / size_of::<f16>()
            + 3 * MODEL_WIDTH * size_of::<f32>() / size_of::<f16>()
            <= padded_rows * FF_WIDTH;
    if use_fused_qkv {
        launch_qkv_fp8(stream, kernels, weights, buffers, rows, padded_rows)?;
    } else {
        launch_norm(
            stream,
            &kernels.layer_norm,
            &buffers.state_a,
            &weights.norm_attention_weight,
            &weights.norm_attention_bias,
            &mut buffers.normalized,
            rows,
            padded_rows,
        )?;
    }
    if use_fused_qkv {
        // Query, packed key and transposed value are already in place.
    } else if use_large_linear {
        for (index, weight) in [&weights.query, &weights.key].into_iter().enumerate() {
            let input = buffers.normalized.slice(..);
            let mut output = buffers
                .workspace
                .slice_mut(index * row_values..(index + 1) * row_values);
            launch_linear_arena(
                stream,
                &kernels.cublas,
                &kernels.silu,
                sequence_linear,
                sequence_launch(MODEL_WIDTH),
                &input,
                weight,
                None,
                &mut output,
                padded_rows,
                MODEL_WIDTH,
                MODEL_WIDTH,
                None,
                1.0,
                0.0,
                0,
            )?;
        }
        let input = buffers.normalized.slice(..);
        let mut output = buffers.state_b.slice_mut(..row_values);
        kernels.cublas.linear_transposed_output(
            &input,
            &weights.value,
            &mut output,
            padded_rows,
            MODEL_WIDTH,
            MODEL_WIDTH,
        )?;
    } else {
        let input = buffers.normalized.slice(..);
        let mut output = buffers.workspace.slice_mut(..3 * row_values);
        launch_qkv_arena(
            stream,
            &kernels.qkv,
            &input,
            weights,
            &mut output,
            padded_rows,
            MODEL_WIDTH,
            MODEL_WIDTH,
        )?;
    }
    let value_stride = padded_rows;
    let mut value_fp8 = if use_long_attention {
        // Original FP8 boundary values, packed S4 values, and per-feature
        // scales fit in the now-dead FP16 normalized arena without allocation.
        let value_bytes = if padded_rows >= 1040 {
            row_values * 3 / 2 + MODEL_WIDTH * size_of::<f32>()
        } else {
            row_values
        };
        let mut output = unsafe {
            buffers
                .normalized
                .transmute_mut::<u8>(value_bytes)
                .ok_or("FP8 value view exceeds normalized encoder buffer")?
        };
        if use_large_linear {
            let value = buffers.state_b.slice(..row_values);
            if padded_rows >= 1040 {
                launch_quantize_value_int4(
                    stream,
                    kernels,
                    &value,
                    &mut output,
                    padded_rows,
                    valid_rows,
                )?;
            } else {
                launch_quantize_fp8(
                    stream,
                    &kernels.quantize_fp8,
                    &value,
                    &mut output,
                    row_values,
                )?;
            }
        } else {
            let value = buffers.workspace.slice(2 * row_values..3 * row_values);
            launch_quantize_fp8_transpose(
                stream,
                &kernels.quantize_fp8_transpose,
                &value,
                &mut output,
                padded_rows,
            )?;
        }
        Some(output)
    } else {
        None
    };
    {
        let (mut query, mut remaining) = buffers.workspace.split_at_mut(row_values);
        let (key, mut remaining) = remaining.split_at_mut(row_values);
        let (value, mut attention) = remaining.split_at_mut(row_values);
        let position_elements = POSITION_ROWS.next_multiple_of(16) * MODEL_WIDTH;
        let position = buffers
            .position_cache
            .slice(position_index * position_elements..(position_index + 1) * position_elements);
        let use_packed_attention = padded_rows >= 1040;
        let attention_kernel = if use_packed_attention {
            &kernels.attention_packed
        } else if use_long_attention {
            &kernels.attention_long
        } else {
            &kernels.attention
        };
        // The fused projection left the packed key vectors and scales in the
        // key arena; the dead FP16 key is never materialized on that path.
        let prepacked_key = if use_fused_qkv {
            Some(unsafe {
                key.transmute::<u8>(qkv_key_arena_bytes(padded_rows))
                    .ok_or("packed key arena exceeds key buffer")?
            })
        } else {
            None
        };
        launch_attention_arena(
            stream,
            attention_kernel,
            &kernels.pack_scores,
            &kernels.pack_key,
            &mut buffers.state_b.slice_mut(..),
            &query.slice(..),
            &key.slice(..),
            &value.slice(..),
            value_fp8.as_mut().map(|values| values.slice(..)).as_ref(),
            prepacked_key.as_ref(),
            &position,
            &weights.bias_u,
            &weights.bias_v,
            &mut attention,
            rows,
            padded_rows,
            valid_rows,
            value_stride,
            buffers.bounds.as_ref(),
        )?;
        let mut output = buffers.state_a.slice_mut(..);
        if use_packed_attention {
            launch_attention_output_fp8(
                stream,
                kernels,
                &attention.slice(..),
                &weights.attention_output,
                &mut output,
                &mut query,
                padded_rows,
            )?;
        } else {
            launch_linear_arena(
                stream,
                &kernels.cublas,
                &kernels.silu,
                sequence_linear,
                sequence_launch(MODEL_WIDTH),
                &attention.slice(..),
                &weights.attention_output,
                None,
                &mut output,
                padded_rows,
                MODEL_WIDTH,
                MODEL_WIDTH,
                None,
                1.0,
                1.0,
                0,
            )?;
        }
    }

    {
        let (mut conv_expanded, mut remaining) = buffers.workspace.split_at_mut(2 * row_values);
        let (mut conv_glu, mut conv_depthwise) = remaining.split_at_mut(row_values);
        // Long shapes have room for both packed operands/scales in the now-dead
        // expanded arena. Short requests retain the original FP16 projection.
        if padded_rows >= 1040 {
            launch_conv_glu_fp8(
                stream,
                kernels,
                &buffers.state_a,
                weights,
                &mut conv_glu,
                &mut conv_expanded,
                rows,
                padded_rows,
                rows.min(valid_rows),
            )?;
        } else {
            launch_norm(
                stream,
                &kernels.layer_norm,
                &buffers.state_a,
                &weights.norm_conv_weight,
                &weights.norm_conv_bias,
                &mut buffers.normalized,
                rows,
                padded_rows,
            )?;
            launch_linear_arena(
                stream,
                &kernels.cublas,
                &kernels.silu,
                sequence_linear,
                sequence_launch(CONV_EXPANDED),
                &buffers.normalized.slice(..),
                &weights.pointwise1,
                None,
                &mut conv_expanded,
                padded_rows,
                CONV_EXPANDED,
                MODEL_WIDTH,
                None,
                1.0,
                0.0,
                0,
            )?;
            launch_glu_arena(
                stream,
                &kernels.glu,
                &conv_expanded.slice(..),
                &mut conv_glu,
                rows,
                padded_rows,
                valid_rows,
                buffers.bounds.as_ref(),
            )?;
        }
        let mut output = buffers.state_a.slice_mut(..);
        if padded_rows >= 1040 {
            launch_conv_residual_fp8(
                stream,
                kernels,
                &conv_glu.slice(..),
                weights,
                &mut output,
                &mut conv_expanded,
                rows,
                padded_rows,
            )?;
        } else {
            launch_depthwise_arena(
                stream,
                &kernels.depthwise,
                &conv_glu.slice(..),
                weights,
                &mut conv_depthwise,
                rows,
                padded_rows,
            )?;
            launch_linear_arena(
                stream,
                &kernels.cublas,
                &kernels.silu,
                sequence_linear,
                sequence_launch(MODEL_WIDTH),
                &conv_depthwise.slice(..),
                &weights.pointwise2,
                None,
                &mut output,
                padded_rows,
                MODEL_WIDTH,
                MODEL_WIDTH,
                None,
                1.0,
                1.0,
                0,
            )?;
        }
    }

    // Packed activations leave half of the active arena free for one quantized
    // contraction matrix and its scales. Smaller shapes retain the FP16 path.
    let ff2_int4 =
        ff2_int4.filter(|_| padded_rows * FF_WIDTH >= MODEL_WIDTH * (FF_WIDTH + size_of::<f32>()));
    let packed_ff2 = ff2_int4.is_some();
    if let Some(ff2_int4) = ff2_int4 {
        let quantized_elements = padded_rows * MODEL_WIDTH / 2;
        let mut quantized = unsafe {
            buffers
                .normalized
                .transmute_mut::<u8>(quantized_elements)
                .ok_or("INT4 activation view exceeds normalized encoder buffer")?
        };
        let mut scales = unsafe {
            buffers
                .state_b
                .transmute_mut::<f32>(padded_rows)
                .ok_or("INT4 row scales exceed inactive encoder state buffer")?
        };
        launch_norm_quantize_fp8(
            stream,
            &kernels.layer_norm_quantize_int4,
            &buffers.state_a,
            &weights.norm_ff2_weight,
            &weights.norm_ff2_bias,
            &mut quantized,
            &mut scales,
            rows,
            padded_rows,
        )?;
        let mut output = buffers.workspace.slice_mut(..);
        launch_ffn_expand_fp8(
            stream,
            &kernels.ffn_expand_int4_sparse,
            &quantized.slice(..),
            &ff2_int4.weights,
            &ff2_int4.scales,
            Some(&scales.slice(..)),
            &mut output,
            padded_rows,
        )?;
    } else {
        launch_norm(
            stream,
            &kernels.layer_norm,
            &buffers.state_a,
            &weights.norm_ff2_weight,
            &weights.norm_ff2_bias,
            &mut buffers.normalized,
            rows,
            padded_rows,
        )?;
        let input = buffers.normalized.slice(..);
        let mut output = buffers.workspace.slice_mut(..);
        launch_ffn_expand(
            stream,
            &kernels.ffn_expand,
            &input,
            &weights.ff2_expand,
            &mut output,
            padded_rows,
        )?;
    }
    if packed_ff2 {
        launch_ffn_contract_fp8(
            stream,
            kernels,
            &weights.ff2_contract,
            buffers,
            padded_rows,
            true,
        )?;
    } else {
        let input = buffers.workspace.slice(..);
        let mut output = buffers.state_a.slice_mut(..);
        launch_linear_arena(
            stream,
            &kernels.cublas,
            &kernels.silu,
            sequence_linear,
            sequence_launch(MODEL_WIDTH),
            &input,
            &weights.ff2_contract,
            None,
            &mut output,
            padded_rows,
            MODEL_WIDTH,
            FF_WIDTH,
            None,
            0.5,
            1.0,
            0,
        )?;
    }
    if let Some(next_weights) = next_weights.filter(|_| padded_rows >= 1040) {
        return launch_norm_pair(
            stream,
            &kernels.layer_norm_pair,
            weights,
            next_weights,
            next_ff1_fp8,
            buffers,
            rows,
            padded_rows,
        );
    }
    launch_norm(
        stream,
        &kernels.layer_norm,
        &buffers.state_a,
        &weights.norm_out_weight,
        &weights.norm_out_bias,
        &mut buffers.state_b,
        rows,
        padded_rows,
    )
}

fn launch_ffn_contract_fp8(
    stream: &CudaStream,
    kernels: &EncoderKernels,
    weights: &CudaView<'_, f16>,
    buffers: &mut LayerBuffers,
    padded_rows: usize,
    sparse: bool,
) -> Result<(), Box<dyn Error>> {
    let activation_bytes = padded_rows * FF_WIDTH;
    let weight_bytes = MODEL_WIDTH * FF_WIDTH;
    let (mut activations, mut remaining) = buffers
        .workspace
        .split_at_mut(activation_bytes / size_of::<f16>());
    let (mut weight_arena, mut scale_arena) =
        remaining.split_at_mut(weight_bytes / size_of::<f16>());
    let activations = unsafe {
        activations
            .transmute_mut::<u8>(activation_bytes)
            .ok_or("packed FFN activations exceed arena")?
    };
    let mut weights_fp8 = unsafe {
        weight_arena
            .transmute_mut::<u8>(weight_bytes)
            .ok_or("packed FFN weights exceed arena")?
    };
    let mut scales = unsafe {
        scale_arena
            .transmute_mut::<f32>(MODEL_WIDTH)
            .ok_or("packed FFN scales exceed arena")?
    };
    let mut builder = stream.launch_builder(&kernels.quantize_ffn_contract);
    builder.arg(weights).arg(&mut weights_fp8).arg(&mut scales);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (MODEL_WIDTH as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    let rows_i32 = i32::try_from(padded_rows)?;
    let activations = activations.slice(..);
    let weights_fp8 = weights_fp8.slice(..);
    let scales = scales.slice(..);
    let inner_width = i32::try_from(FF_WIDTH)?;
    let mut builder = stream.launch_builder(if sparse {
        &kernels.ffn_contract_fp8_sparse
    } else {
        &kernels.ffn_contract_fp8
    });
    builder
        .arg(&activations)
        .arg(&weights_fp8)
        .arg(&scales)
        .arg(&mut buffers.state_a)
        .arg(&rows_i32);
    if sparse {
        builder.arg(&inner_width);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (MODEL_WIDTH / 128) as u32,
                u32::try_from(padded_rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: if sparse {
                2 * (128 * 32 + 128 * 64 + 128 * 8)
            } else {
                2 * (128 * 64 + 128 * 64)
            },
        })
    }?;
    Ok(())
}

fn launch_ffn_expand(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    weights: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, f16>,
    rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows_i32 = i32::try_from(rows)?;
    let mut builder = stream.launch_builder(function);
    builder.arg(input).arg(weights).arg(output).arg(&rows_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (FF_WIDTH / 128) as u32,
                u32::try_from(rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 2 * (128 * 32 + 128 * 32) * size_of::<f16>() as u32,
        })
    }?;
    Ok(())
}

fn launch_quantize_value_int4(
    stream: &CudaStream,
    kernels: &EncoderKernels,
    input: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, u8>,
    padded_rows: usize,
    valid_rows: usize,
) -> Result<(), Box<dyn Error>> {
    // 1024 threads × 24 register-held half2 values. Longer inputs use the
    // general two-pass producer instead of truncating or adding an arena.
    let register_strip = padded_rows <= 49_152;
    let elements_i32 = i32::try_from(padded_rows * MODEL_WIDTH)?;
    let valid_rows_i32 = i32::try_from(valid_rows)?;
    let inverse_scale = FP8_INPUT_SCALE.recip();
    let function = if register_strip {
        &kernels.quantize_value_int4_register
    } else {
        &kernels.quantize_value_int4
    };
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&elements_i32)
        .arg(&inverse_scale);
    if register_strip {
        builder.arg(&valid_rows_i32);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (MODEL_WIDTH as u32, 1, 1),
            block_dim: (if register_strip { 1024 } else { 256 }, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn launch_quantize_fp8(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, u8>,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let elements_i32 = i32::try_from(elements)?;
    let inverse_scale = FP8_INPUT_SCALE.recip();
    let threads = 256_u32;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&elements_i32)
        .arg(&inverse_scale);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (u32::try_from(elements)?.div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn launch_quantize_fp8_transpose(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, u8>,
    rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows_i32 = i32::try_from(rows)?;
    let inverse_scale = FP8_INPUT_SCALE.recip();
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&rows_i32)
        .arg(&inverse_scale);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(MODEL_WIDTH / 32)?,
                u32::try_from(rows.div_ceil(32))?,
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn launch_ffn_expand_fp8(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, u8>,
    weights: &CudaSlice<u8>,
    weight_scales: &CudaSlice<f32>,
    input_scales: Option<&CudaView<'_, f32>>,
    output: &mut CudaViewMut<'_, f16>,
    rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows_i32 = i32::try_from(rows)?;
    let dynamic_scale = i32::from(input_scales.is_some());
    let mut builder = stream.launch_builder(function);
    builder.arg(input).arg(weights).arg(weight_scales);
    if let Some(input_scales) = input_scales {
        builder.arg(input_scales);
    } else {
        builder.arg(weight_scales);
    }
    builder
        .arg(output)
        .arg(&rows_i32)
        .arg(&FP8_INPUT_SCALE)
        .arg(&dynamic_scale);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (FF_WIDTH / 128) as u32,
                u32::try_from(rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 2 * (128 * 64 + 128 * 64),
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_norm(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaSlice<f16>,
    weight: &CudaView<'_, f16>,
    bias: &CudaView<'_, f16>,
    output: &mut CudaSlice<f16>,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let epsilon = 1e-5_f32;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weight)
        .arg(bias)
        .arg(output)
        .arg(&rows)
        .arg(&padded_rows_i32)
        .arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (u32::try_from(padded_rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_norm_pair(
    stream: &CudaStream,
    function: &CudaFunction,
    weights: &LayerWeights<'_>,
    next_weights: &LayerWeights<'_>,
    next_ff1_fp8: bool,
    buffers: &mut LayerBuffers,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let epsilon = 1e-5_f32;
    let inverse_scale = if next_ff1_fp8 {
        FP8_INPUT_SCALE.recip()
    } else {
        0.0
    };
    let mut builder = stream.launch_builder(function);
    builder
        .arg(&buffers.state_a)
        .arg(&weights.norm_out_weight)
        .arg(&weights.norm_out_bias)
        .arg(&mut buffers.state_b)
        .arg(&next_weights.norm_ff1_weight)
        .arg(&next_weights.norm_ff1_bias)
        .arg(&mut buffers.normalized)
        .arg(&rows)
        .arg(&padded_rows_i32)
        .arg(&epsilon)
        .arg(&inverse_scale);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (u32::try_from(padded_rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_norm_quantize_fp8(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaSlice<f16>,
    weight: &CudaView<'_, f16>,
    bias: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, u8>,
    scales: &mut CudaViewMut<'_, f32>,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let rows_i32 = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let epsilon = 1e-5_f32;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weight)
        .arg(bias)
        .arg(output)
        .arg(scales)
        .arg(&rows_i32)
        .arg(&padded_rows_i32)
        .arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (u32::try_from(padded_rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_conv_glu_fp8(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    input: &CudaSlice<f16>,
    layer: &LayerWeights<'_>,
    output: &mut CudaViewMut<'_, f16>,
    scratch: &mut CudaViewMut<'_, f16>,
    rows: usize,
    padded_rows: usize,
    valid_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let elements = padded_rows * MODEL_WIDTH;
    let (mut activation_arena, mut remaining) = scratch.split_at_mut(elements / size_of::<f16>());
    let (mut weight_arena, mut remaining) =
        remaining.split_at_mut(CONV_EXPANDED * MODEL_WIDTH / size_of::<f16>());
    let (mut row_scale_arena, mut scale_arena) =
        remaining.split_at_mut(padded_rows * size_of::<f32>() / size_of::<f16>());
    let mut activations = unsafe {
        activation_arena
            .transmute_mut::<u8>(elements)
            .ok_or("FP8 GLU activations exceed arena")?
    };
    let mut weights = unsafe {
        weight_arena
            .transmute_mut::<u8>(CONV_EXPANDED * MODEL_WIDTH)
            .ok_or("FP8 GLU weights exceed arena")?
    };
    let mut row_scales = unsafe {
        row_scale_arena
            .transmute_mut::<f32>(padded_rows)
            .ok_or("FP8 GLU row scales exceed arena")?
    };
    let mut scales = unsafe {
        scale_arena
            .transmute_mut::<f32>(CONV_EXPANDED)
            .ok_or("FP8 GLU weight scales exceed arena")?
    };
    // Reuse the existing exact FP16-rounded normalization/quantization kernel.
    launch_norm_quantize_fp8(
        stream,
        &kernels.layer_norm_quantize_fp8,
        input,
        &layer.norm_conv_weight,
        &layer.norm_conv_bias,
        &mut activations,
        &mut row_scales,
        rows,
        padded_rows,
    )?;
    // Runtime packing remains inside each layer/request, not engine load.
    let mut builder = stream.launch_builder(&kernels.quantize_rows1024);
    builder
        .arg(&layer.pointwise1)
        .arg(&mut weights)
        .arg(&mut scales);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (CONV_EXPANDED as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    let activations = activations.slice(..);
    let weights = weights.slice(..);
    let row_scales = row_scales.slice(..);
    let scales = scales.slice(..);
    let rows_i32 = i32::try_from(padded_rows)?;
    let valid_rows_i32 = i32::try_from(valid_rows)?;
    let mut builder = stream.launch_builder(&kernels.conv_glu_fp8);
    builder
        .arg(&activations)
        .arg(&weights)
        .arg(&scales)
        .arg(&row_scales)
        .arg(output)
        .arg(&rows_i32)
        .arg(&valid_rows_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (MODEL_WIDTH / 64) as u32,
                u32::try_from(padded_rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 32768,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_conv_residual_fp8(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    input: &CudaView<'_, f16>,
    layer: &LayerWeights<'_>,
    output: &mut CudaViewMut<'_, f16>,
    scratch: &mut CudaViewMut<'_, f16>,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    // The pointwise1 expanded arena is dead after GLU. Keep runtime packing
    // inside the request and away from both GLU input and residual output.
    let elements = padded_rows * MODEL_WIDTH;
    let (mut activation_arena, mut remaining) = scratch.split_at_mut(elements / size_of::<f16>());
    let (mut weight_arena, mut remaining) =
        remaining.split_at_mut(MODEL_WIDTH * MODEL_WIDTH / size_of::<f16>());
    let (mut row_scale_arena, mut scale_arena) =
        remaining.split_at_mut(padded_rows * size_of::<f32>() / size_of::<f16>());
    let mut activations = unsafe {
        activation_arena
            .transmute_mut::<u8>(elements)
            .ok_or("FP8 depthwise activations exceed arena")?
    };
    let mut weights = unsafe {
        weight_arena
            .transmute_mut::<u8>(MODEL_WIDTH * MODEL_WIDTH)
            .ok_or("FP8 pointwise2 weights exceed arena")?
    };
    let mut row_scales = unsafe {
        row_scale_arena
            .transmute_mut::<f32>(padded_rows)
            .ok_or("FP8 depthwise row scales exceed arena")?
    };
    let mut scales = unsafe {
        scale_arena
            .transmute_mut::<f32>(MODEL_WIDTH)
            .ok_or("FP8 pointwise2 weight scales exceed arena")?
    };
    let mut builder = stream.launch_builder(&kernels.quantize_rows1024);
    builder
        .arg(&layer.pointwise2)
        .arg(&mut weights)
        .arg(&mut scales);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (MODEL_WIDTH as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    let rows_i32 = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let epsilon = 1e-5_f32;
    let mut builder = stream.launch_builder(&kernels.depthwise_pack);
    builder
        .arg(input)
        .arg(&layer.depthwise)
        .arg(&layer.conv_norm_weight)
        .arg(&layer.conv_norm_bias)
        .arg(&layer.conv_running_mean)
        .arg(&layer.conv_running_variance)
        .arg(&mut activations)
        .arg(&mut row_scales)
        .arg(&rows_i32)
        .arg(&padded_rows_i32)
        .arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (u32::try_from(padded_rows.div_ceil(8))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    let activations = activations.slice(..);
    let weights = weights.slice(..);
    let row_scales = row_scales.slice(..);
    let scales = scales.slice(..);
    let mut builder = stream.launch_builder(&kernels.conv_residual_fp8);
    builder
        .arg(&activations)
        .arg(&weights)
        .arg(&scales)
        .arg(&row_scales)
        .arg(output)
        .arg(&padded_rows_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (MODEL_WIDTH / 128) as u32,
                u32::try_from(padded_rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 32768,
        })
    }?;
    Ok(())
}

fn launch_attention_output_fp8(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    input: &CudaView<'_, f16>,
    weight: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, f16>,
    scratch: &mut CudaViewMut<'_, f16>,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    // Query is dead after attention; reuse its arena for request-time weight
    // packing. Packed context and the residual output remain disjoint.
    let (mut weight_arena, mut scale_arena) =
        scratch.split_at_mut(MODEL_WIDTH * MODEL_WIDTH / size_of::<f16>());
    let input = unsafe {
        input
            .transmute::<u8>(padded_rows * MODEL_WIDTH)
            .ok_or("FP8 attention context exceeds arena")?
    };
    let mut weights = unsafe {
        weight_arena
            .transmute_mut::<u8>(MODEL_WIDTH * MODEL_WIDTH)
            .ok_or("FP8 attention weights exceed arena")?
    };
    let mut scales = unsafe {
        scale_arena
            .transmute_mut::<f32>(MODEL_WIDTH)
            .ok_or("FP8 attention weight scales exceed arena")?
    };
    let mut builder = stream.launch_builder(&kernels.quantize_rows1024);
    builder.arg(weight).arg(&mut weights).arg(&mut scales);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (MODEL_WIDTH as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    let weights = weights.slice(..);
    let scales = scales.slice(..);
    let rows = i32::try_from(padded_rows)?;
    let mut builder = stream.launch_builder(&kernels.attention_output_fp8);
    builder
        .arg(&input)
        .arg(&weights)
        .arg(&scales)
        .arg(output)
        .arg(&rows);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (MODEL_WIDTH / 128) as u32,
                u32::try_from(padded_rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 32768,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_linear_arena(
    stream: &Arc<CudaStream>,
    cublas: &Cublas,
    silu: &CudaFunction,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaView<'_, f16>,
    weights: &CudaView<'_, f16>,
    bias: Option<&CudaView<'_, f32>>,
    output: &mut CudaViewMut<'_, f16>,
    m: usize,
    n: usize,
    k: usize,
    residual: Option<&CudaView<'_, f16>>,
    output_scale: f32,
    residual_scale: f32,
    activation: i32,
) -> Result<(), Box<dyn Error>> {
    let is_feed_forward =
        (n == FF_WIDTH && k == MODEL_WIDTH) || (n == MODEL_WIDTH && k == FF_WIDTH);
    let is_residual_projection =
        n == MODEL_WIDTH && k == MODEL_WIDTH && residual_scale != 0.0 && activation == 0;
    if (m >= LARGE_LINEAR_ROWS || is_feed_forward || is_residual_projection) && bias.is_none() {
        if activation == 0 {
            if let Some(residual) = residual {
                stream.memcpy_dtod(residual, output)?;
            }
            return cublas.linear(
                input,
                weights,
                output,
                m,
                n,
                k,
                output_scale,
                residual_scale,
            );
        }
        if activation == 1 && residual.is_none() {
            cublas.linear(input, weights, output, m, n, k, output_scale, 0.0)?;
            let elements = i32::try_from(m * n)?;
            let threads = 256_u32;
            let mut builder = stream.launch_builder(silu);
            builder.arg(output).arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (u32::try_from(m * n)?.div_ceil(threads), 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            return Ok(());
        }
    }
    let null_pointer = 0_u64;
    let m = i32::try_from(m)?;
    let n = i32::try_from(n)?;
    let k = i32::try_from(k)?;
    let mut builder = stream.launch_builder(function);
    builder.arg(input).arg(weights);
    if let Some(bias) = bias {
        builder.arg(bias);
    } else {
        builder.arg(&null_pointer);
    }
    if let Some(residual) = residual {
        builder.arg(residual);
    } else {
        builder.arg(&null_pointer);
    }
    builder
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&output_scale)
        .arg(&residual_scale)
        .arg(&activation);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn launch_qkv_arena(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    weights: &LayerWeights<'_>,
    output: &mut CudaViewMut<'_, f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let config = linear_launch_config(m, n);
    let config = LaunchConfig {
        grid_dim: (config.grid_dim.0, config.grid_dim.1, 3),
        ..config
    };
    let m = i32::try_from(m)?;
    let n = i32::try_from(n)?;
    let k = i32::try_from(k)?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(&weights.query)
        .arg(&weights.key)
        .arg(&weights.value)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k);
    unsafe { builder.launch(config) }?;
    Ok(())
}

/// Packed key vectors (64 bytes per row and head) followed by their scales.
fn qkv_key_arena_bytes(padded_rows: usize) -> usize {
    padded_rows * MODEL_WIDTH + padded_rows * 8 * size_of::<f32>()
}

/// One FP8 launch for the query, key and value projections of packed shapes.
///
/// The normalization kernel is the retained exact FP16-rounded LayerNorm with
/// dynamic E4M3 row quantization. Weight packing stays inside the request in
/// the dead FP16 value slot of the workspace, so no persistent bytes change.
fn launch_qkv_fp8(
    stream: &Arc<CudaStream>,
    kernels: &EncoderKernels,
    layer: &LayerWeights<'_>,
    buffers: &mut LayerBuffers,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let row_values = padded_rows * MODEL_WIDTH;
    let mut activations = unsafe {
        buffers
            .normalized
            .transmute_mut::<u8>(row_values)
            .ok_or("FP8 attention input exceeds normalized buffer")?
    };
    let (mut query, mut remaining) = buffers.workspace.split_at_mut(row_values);
    let (mut key_arena, mut remaining) = remaining.split_at_mut(row_values);
    let (mut row_scale_arena, mut remaining) =
        remaining.split_at_mut(padded_rows * size_of::<f32>() / size_of::<f16>());
    let (mut weight_arena, mut scale_arena) =
        remaining.split_at_mut(3 * MODEL_WIDTH * MODEL_WIDTH / size_of::<f16>());
    let mut row_scales = unsafe {
        row_scale_arena
            .transmute_mut::<f32>(padded_rows)
            .ok_or("FP8 attention row scales exceed arena")?
    };
    let mut weights = unsafe {
        weight_arena
            .transmute_mut::<u8>(3 * MODEL_WIDTH * MODEL_WIDTH)
            .ok_or("FP8 attention projection weights exceed arena")?
    };
    let mut scales = unsafe {
        scale_arena
            .transmute_mut::<f32>(3 * MODEL_WIDTH)
            .ok_or("FP8 attention projection scales exceed arena")?
    };
    let mut key_arena = unsafe {
        key_arena
            .transmute_mut::<u8>(qkv_key_arena_bytes(padded_rows))
            .ok_or("packed key arena exceeds key buffer")?
    };
    launch_norm_quantize_fp8(
        stream,
        &kernels.layer_norm_quantize_fp8,
        &buffers.state_a,
        &layer.norm_attention_weight,
        &layer.norm_attention_bias,
        &mut activations,
        &mut row_scales,
        rows,
        padded_rows,
    )?;
    for (index, weight) in [&layer.query, &layer.key, &layer.value]
        .into_iter()
        .enumerate()
    {
        let mut packed = weights.slice_mut(index * MODEL_WIDTH * MODEL_WIDTH..);
        let mut packed_scales = scales.slice_mut(index * MODEL_WIDTH..);
        let mut builder = stream.launch_builder(&kernels.quantize_rows1024);
        builder.arg(weight).arg(&mut packed).arg(&mut packed_scales);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (MODEL_WIDTH as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }
    let activations = activations.slice(..);
    let weights = weights.slice(..);
    let scales = scales.slice(..);
    let row_scales = row_scales.slice(..);
    let (mut packed_key, mut key_scale_bytes) = key_arena.split_at_mut(row_values);
    let mut key_scales = unsafe {
        key_scale_bytes
            .transmute_mut::<f32>(padded_rows * 8)
            .ok_or("packed key scales exceed key buffer")?
    };
    let mut value_t = buffers.state_b.slice_mut(..row_values);
    let rows_i32 = i32::try_from(padded_rows)?;
    let mut builder = stream.launch_builder(&kernels.qkv_fp8);
    builder
        .arg(&activations)
        .arg(&weights)
        .arg(&scales)
        .arg(&row_scales)
        .arg(&mut query)
        .arg(&mut packed_key)
        .arg(&mut key_scales)
        .arg(&mut value_t)
        .arg(&rows_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                (3 * MODEL_WIDTH / 128) as u32,
                u32::try_from(padded_rows.div_ceil(128))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 128 * 136 * size_of::<f16>() as u32,
        })
    }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_arena(
    stream: &CudaStream,
    function: &CudaFunction,
    pack_scores: &CudaFunction,
    pack_key: &CudaFunction,
    scratch: &mut CudaViewMut<'_, f16>,
    query: &CudaView<'_, f16>,
    key: &CudaView<'_, f16>,
    value: &CudaView<'_, f16>,
    value_fp8: Option<&CudaView<'_, u8>>,
    prepacked_key: Option<&CudaView<'_, u8>>,
    position: &CudaView<'_, f16>,
    bias_u: &CudaView<'_, f16>,
    bias_v: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, f16>,
    rows: usize,
    padded_rows: usize,
    valid_rows: usize,
    value_stride: usize,
    bounds: Option<&CudaSlice<i32>>,
) -> Result<(), Box<dyn Error>> {
    let rows_i32 = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let valid_rows_i32 = i32::try_from(valid_rows)?;
    let attention_left = i32::try_from(ATTENTION_LEFT)?;
    let attention_right = i32::try_from(ATTENTION_RIGHT)?;
    let value_stride = i32::try_from(value_stride)?;
    let key_bytes = padded_rows * MODEL_WIDTH;
    let key_scale_bytes = padded_rows * 8 * size_of::<f32>();
    let position_bytes = POSITION_ROWS.next_multiple_of(16) * MODEL_WIDTH;
    let position_scale_bytes = position_bytes / 128 * size_of::<f32>();
    let position_offset = key_bytes + key_scale_bytes;
    // FP16 V has been quantized before this call. Reuse its now-dead arena;
    // query, key, residual and attention output remain in separate buffers.
    let mut packed = if padded_rows >= 1040 {
        Some(unsafe {
            scratch
                .transmute_mut::<u8>(position_offset + position_bytes + position_scale_bytes)
                .ok_or("attention score packing exceeds dead V arena")?
        })
    } else {
        None
    };
    if let Some(arena) = packed.as_mut() {
        let (mut keys, mut positions) = arena.split_at_mut(position_offset);
        if prepacked_key.is_none() {
            let (mut values, mut scales) = keys.split_at_mut(key_bytes);
            let vectors = i32::try_from(key_bytes / 128)?;
            let config = LaunchConfig {
                grid_dim: (u32::try_from(padded_rows)?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(pack_key)
                    .arg(key)
                    .arg(&mut values)
                    .arg(&mut scales)
                    .arg(&vectors)
                    .launch(config)
            }?;
        }
        let (mut values, mut scales) = positions.split_at_mut(position_bytes);
        let vectors = i32::try_from(position_bytes / 128)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(position_bytes / MODEL_WIDTH)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(pack_scores)
                .arg(position)
                .arg(&mut values)
                .arg(&mut scales)
                .arg(&vectors)
                .launch(config)
        }?;
    }
    let packed_key = prepacked_key
        .map(|arena| arena.slice(..key_bytes))
        .or_else(|| packed.as_ref().map(|arena| arena.slice(..key_bytes)));
    let key_scales = prepacked_key
        .map(|arena| arena.slice(key_bytes..position_offset))
        .or_else(|| {
            packed
                .as_ref()
                .map(|arena| arena.slice(key_bytes..position_offset))
        });
    let packed_position = packed
        .as_ref()
        .map(|arena| arena.slice(position_offset..position_offset + position_bytes));
    let position_scales = packed
        .as_ref()
        .map(|arena| arena.slice(position_offset + position_bytes..));
    let mut builder = stream.launch_builder(function);
    builder.arg(query).arg(key).arg(value);
    if let Some(value_fp8) = value_fp8 {
        builder.arg(value_fp8);
    }
    builder
        .arg(position)
        .arg(bias_u)
        .arg(bias_v)
        .arg(output)
        .arg(&rows_i32)
        .arg(&padded_rows_i32)
        .arg(&valid_rows_i32)
        .arg(&attention_left)
        .arg(&attention_right);
    let null_bounds = 0_u64;
    if padded_rows < 512 || bounds.is_some() {
        if let Some(bounds) = bounds {
            builder.arg(bounds);
        } else {
            builder.arg(&null_bounds);
        }
    }
    if value_fp8.is_some() {
        builder.arg(&value_stride);
    }
    if let (Some(key), Some(ks), Some(position), Some(ps)) =
        (&packed_key, &key_scales, &packed_position, &position_scales)
    {
        builder.arg(key).arg(ks).arg(position).arg(ps);
    }
    let config = if padded_rows >= 512 && bounds.is_none() {
        LaunchConfig {
            grid_dim: (u32::try_from(padded_rows.div_ceil(16))?, 8, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (u32::try_from(padded_rows)?, 8, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    unsafe { builder.launch(config) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_glu_arena(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    output: &mut CudaViewMut<'_, f16>,
    rows: usize,
    padded_rows: usize,
    valid_rows: usize,
    bounds: Option<&CudaSlice<i32>>,
) -> Result<(), Box<dyn Error>> {
    let threads = 256_u32;
    let rows_i32 = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let valid_rows_i32 = i32::try_from(valid_rows)?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&rows_i32)
        .arg(&padded_rows_i32)
        .arg(&valid_rows_i32);
    let null_bounds = 0_u64;
    if let Some(bounds) = bounds {
        builder.arg(bounds);
    } else {
        builder.arg(&null_bounds);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(padded_rows * MODEL_WIDTH)?.div_ceil(threads),
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn launch_depthwise_arena(
    stream: &CudaStream,
    function: &CudaFunction,
    input: &CudaView<'_, f16>,
    weights: &LayerWeights<'_>,
    output: &mut CudaViewMut<'_, f16>,
    rows: usize,
    padded_rows: usize,
) -> Result<(), Box<dyn Error>> {
    let threads = 256_u32;
    let rows_i32 = i32::try_from(rows)?;
    let padded_rows_i32 = i32::try_from(padded_rows)?;
    let epsilon = 1e-5_f32;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(&weights.depthwise)
        .arg(&weights.conv_norm_weight)
        .arg(&weights.conv_norm_bias)
        .arg(&weights.conv_running_mean)
        .arg(&weights.conv_running_variance)
        .arg(output)
        .arg(&rows_i32)
        .arg(&padded_rows_i32)
        .arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                u32::try_from(padded_rows.div_ceil(8) * MODEL_WIDTH / threads as usize)?,
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

pub(super) fn relative_position_encoding() -> Vec<f16> {
    let factor = -(10_000.0_f64.ln() / MODEL_WIDTH as f64);
    let mut output = Vec::with_capacity(POSITION_ROWS.next_multiple_of(16) * MODEL_WIDTH);
    for position_index in 0..POSITION_ROWS.next_multiple_of(16) {
        let position = if position_index < POSITION_ROWS {
            ATTENTION_LEFT as f64 - position_index as f64
        } else {
            0.0
        };
        for pair in 0..MODEL_WIDTH / 2 {
            let argument = position * ((2 * pair) as f64 * factor).exp();
            output.push(f16::from_f32(argument.sin() as f32));
            output.push(f16::from_f32(argument.cos() as f32));
        }
    }
    output
}

fn fp16_tensor<'a>(
    uploaded: &'a UploadedAot,
    name: &str,
    storage: AotStorage,
) -> Result<CudaView<'a, f16>, Box<dyn Error>> {
    let tensor = tensor(uploaded, name, storage)?;
    let bytes = tensor_bytes(uploaded, tensor)?;
    // SAFETY: validated AOT storage and alignment establish a complete FP16 view.
    unsafe { bytes.transmute::<f16>(usize::try_from(tensor.bytes / 2)?) }
        .ok_or_else(|| format!("{name}: FP16 tensor view exceeds artifact").into())
}

fn fp32_tensor<'a>(
    uploaded: &'a UploadedAot,
    name: &str,
    storage: AotStorage,
) -> Result<CudaView<'a, f32>, Box<dyn Error>> {
    let tensor = tensor(uploaded, name, storage)?;
    let bytes = tensor_bytes(uploaded, tensor)?;
    // SAFETY: validated AOT storage and alignment establish a complete FP32 view.
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
