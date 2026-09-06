use super::{SM89_CUBIN, UploadedAot, launch_linear_view, linear_launch_config, upload_aot};
use crate::artifact::{AotStorage, AotTensor};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::Ptx;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;

pub(super) const DECODER_CUBIN: &[u8] = include_bytes!(env!("PARAKEET_SM89_DECODER_CUBIN"));
pub(super) const ENCODER_WIDTH: usize = 1_024;
pub(super) const JOINT_WIDTH: usize = 640;
pub(super) const MAX_SYMBOLS: usize = 10;
pub(super) const FP8_MIN_FRAMES: usize = 1_024;
pub(super) const DECODER_CONTROL_VALUES: usize = 9;
pub(super) const L4_SMS: u32 = 58;
// Token ids that can become the prediction input; blank never updates the LSTM.
pub(super) const INPUT_TABLE_TOKENS: usize = 1_024;
pub(super) const GATE_WIDTH: usize = 4 * JOINT_WIDTH;

pub(super) fn input_table_halves() -> usize {
    INPUT_TABLE_TOKENS * GATE_WIDTH
}

pub(super) fn workspace_floats(frames: usize) -> usize {
    if frames >= FP8_MIN_FRAMES {
        // Matches kCacheScales, kCacheMaxima, and kCacheWarps in decoder.cu:
        // the state prefix, three per-tensor scales, and per-warp maxima.
        9_360 + 16 + 3 * (3 * L4_SMS as usize) * 4
    } else {
        9_350
    }
}

#[derive(Debug, Serialize)]
pub struct DecoderBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub artifact: String,
    pub frames: usize,
    pub padded_frames: usize,
    pub explicit_workspace_bytes: usize,
    pub warmup_iterations: usize,
    pub warm_iterations: usize,
    pub warm_latency_ms: f64,
    pub token_count: usize,
    pub token_ids: Vec<i32>,
    pub transcript: Option<String>,
    pub reference_token_count: usize,
    pub exact_token_match: Option<bool>,
}

pub fn benchmark_decoder(
    device: usize,
    artifact_path: &Path,
    encoder_f32: &[f32],
    frames: usize,
    reference: Option<&[i32]>,
    vocabulary: Option<&[String]>,
    warmup_iterations: usize,
    warm_iterations: usize,
) -> Result<DecoderBenchmarkReport, Box<dyn Error>> {
    if frames == 0
        || encoder_f32.len() != frames * ENCODER_WIDTH
        || warmup_iterations == 0
        || warm_iterations == 0
    {
        return Err("invalid decoder input shape or benchmark iterations".into());
    }
    let padded_frames = frames.next_multiple_of(16);
    let mut encoder_host = vec![f16::ZERO; padded_frames * ENCODER_WIDTH];
    for (target, source) in encoder_host.iter_mut().zip(encoder_f32) {
        *target = f16::from_f32(*source);
    }

    let uploaded = upload_aot(device, artifact_path)?;
    let (major, minor) = uploaded.context.compute_capability()?;
    let linear = uploaded
        .context
        .load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?
        .load_function("pk_sm89_fp16_linear_epilogue")?;
    let decoder = uploaded
        .context
        .load_module(Ptx::from_binary(DECODER_CUBIN.to_vec()))?
        .load_function(if frames >= FP8_MIN_FRAMES {
            "pk_sm89_tdt_persistent_fp8_ih"
        } else {
            "pk_sm89_tdt_persistent_v2"
        })?;
    let weights = DecoderWeights::load(&uploaded)?;
    let encoder = uploaded.stream.clone_htod(&encoder_host)?;
    let mut encoder_projection = uploaded
        .stream
        .alloc_zeros::<f16>(padded_frames * JOINT_WIDTH)?;
    let mut output_tokens = uploaded.stream.alloc_zeros::<i32>(frames * MAX_SYMBOLS)?;
    let mut output_count = uploaded.stream.alloc_zeros::<i32>(1)?;
    let mut decoder_workspace = uploaded
        .stream
        .alloc_zeros::<f32>(workspace_floats(frames))?;
    let mut decoder_control = uploaded.stream.alloc_zeros::<i32>(DECODER_CONTROL_VALUES)?;
    let mut input_table = uploaded.stream.alloc_zeros::<f16>(input_table_halves())?;

    for _ in 0..warmup_iterations {
        launch_decode(
            &uploaded.stream,
            &linear,
            &decoder,
            &weights,
            &encoder,
            &mut encoder_projection,
            &mut input_table,
            &mut decoder_workspace,
            &mut decoder_control,
            &mut output_tokens,
            &mut output_count,
            frames,
            padded_frames,
        )?;
    }
    uploaded.stream.synchronize()?;
    let started = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..warm_iterations {
        launch_decode(
            &uploaded.stream,
            &linear,
            &decoder,
            &weights,
            &encoder,
            &mut encoder_projection,
            &mut input_table,
            &mut decoder_workspace,
            &mut decoder_control,
            &mut output_tokens,
            &mut output_count,
            frames,
            padded_frames,
        )?;
    }
    let ended = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let warm_latency_ms = f64::from(started.elapsed_ms(&ended)?) / warm_iterations as f64;

    let count_host = uploaded.stream.clone_dtoh(&output_count)?;
    let token_count = usize::try_from(count_host[0])?;
    if token_count > output_tokens.len() {
        return Err(format!(
            "decoder produced {token_count} tokens into capacity {}",
            output_tokens.len()
        )
        .into());
    }
    let mut token_ids = uploaded.stream.clone_dtoh(&output_tokens)?;
    token_ids.truncate(token_count);
    let exact_token_match = reference.map(|expected| expected == token_ids);
    if exact_token_match == Some(false) {
        let first = reference
            .expect("reference exists")
            .iter()
            .zip(&token_ids)
            .position(|(expected, observed)| expected != observed)
            .unwrap_or(
                reference
                    .expect("reference exists")
                    .len()
                    .min(token_ids.len()),
            );
        return Err(format!(
            "decoder token parity failed at token {first}: expected {:?}, observed {:?}",
            reference.expect("reference exists").get(first),
            token_ids.get(first)
        )
        .into());
    }
    let transcript = vocabulary
        .map(|pieces| decode_tokens(pieces, &token_ids))
        .transpose()?;

    Ok(DecoderBenchmarkReport {
        schema_version: 1,
        device,
        name: uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        artifact: artifact_path.display().to_string(),
        frames,
        padded_frames,
        explicit_workspace_bytes: encoder.len() * size_of::<f16>()
            + encoder_projection.len() * size_of::<f16>()
            + input_table.len() * size_of::<f16>()
            + decoder_workspace.len() * size_of::<f32>()
            + decoder_control.len() * size_of::<i32>()
            + output_tokens.len() * size_of::<i32>()
            + output_count.len() * size_of::<i32>(),
        warmup_iterations,
        warm_iterations,
        warm_latency_ms,
        token_count,
        token_ids,
        transcript,
        reference_token_count: reference.map_or(0, <[i32]>::len),
        exact_token_match,
    })
}

pub(super) fn decode_tokens(
    pieces: &[String],
    token_ids: &[i32],
) -> Result<String, Box<dyn Error>> {
    let mut text = String::new();
    for &id in token_ids {
        let piece = pieces.get(usize::try_from(id)?).ok_or_else(|| {
            format!(
                "token ID {id} is outside vocabulary of size {}",
                pieces.len()
            )
        })?;
        text.push_str(&piece.replace('▁', " "));
    }
    Ok(text.trim_start().to_owned())
}

pub(super) struct DecoderWeights<'a> {
    encoder_projection_weight: CudaView<'a, f16>,
    encoder_projection_bias: CudaView<'a, f32>,
    embedding: CudaView<'a, f16>,
    weight_ih_l0: CudaView<'a, f16>,
    weight_hh_l0: CudaView<'a, f16>,
    bias_ih_l0: CudaView<'a, f32>,
    bias_hh_l0: CudaView<'a, f32>,
    weight_ih_l1: CudaView<'a, f16>,
    weight_hh_l1: CudaView<'a, f16>,
    bias_ih_l1: CudaView<'a, f32>,
    bias_hh_l1: CudaView<'a, f32>,
    decoder_projection_weight: CudaView<'a, f16>,
    decoder_projection_bias: CudaView<'a, f32>,
    joint_weight: CudaView<'a, f16>,
    joint_bias: CudaView<'a, f32>,
}

impl<'a> DecoderWeights<'a> {
    pub(super) fn load(uploaded: &'a UploadedAot) -> Result<Self, Box<dyn Error>> {
        let linear = AotStorage::Sm89Fp16Linear;
        let bias = AotStorage::Sm89Fp32Bias;
        Ok(Self {
            encoder_projection_weight: fp16_tensor(uploaded, "encoder_projector.weight", linear)?,
            encoder_projection_bias: fp32_tensor(uploaded, "encoder_projector.bias", bias)?,
            embedding: fp16_tensor(uploaded, "decoder.embedding.weight", AotStorage::Fp16)?,
            weight_ih_l0: fp16_tensor(uploaded, "decoder.lstm.weight_ih_l0", linear)?,
            weight_hh_l0: fp16_tensor(uploaded, "decoder.lstm.weight_hh_l0", linear)?,
            bias_ih_l0: fp32_tensor(uploaded, "decoder.lstm.bias_ih_l0", AotStorage::Fp32)?,
            bias_hh_l0: fp32_tensor(uploaded, "decoder.lstm.bias_hh_l0", AotStorage::Fp32)?,
            weight_ih_l1: fp16_tensor(uploaded, "decoder.lstm.weight_ih_l1", linear)?,
            weight_hh_l1: fp16_tensor(uploaded, "decoder.lstm.weight_hh_l1", linear)?,
            bias_ih_l1: fp32_tensor(uploaded, "decoder.lstm.bias_ih_l1", AotStorage::Fp32)?,
            bias_hh_l1: fp32_tensor(uploaded, "decoder.lstm.bias_hh_l1", AotStorage::Fp32)?,
            decoder_projection_weight: fp16_tensor(
                uploaded,
                "decoder.decoder_projector.weight",
                linear,
            )?,
            decoder_projection_bias: fp32_tensor(uploaded, "decoder.decoder_projector.bias", bias)?,
            joint_weight: fp16_tensor(uploaded, "joint.head.weight", linear)?,
            joint_bias: fp32_tensor(uploaded, "joint.head.bias", bias)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_decode(
    stream: &CudaStream,
    linear: &cudarc::driver::CudaFunction,
    decoder: &cudarc::driver::CudaFunction,
    weights: &DecoderWeights<'_>,
    encoder: &CudaSlice<f16>,
    encoder_projection: &mut CudaSlice<f16>,
    input_table: &mut CudaSlice<f16>,
    decoder_workspace: &mut CudaSlice<f32>,
    decoder_control: &mut CudaSlice<i32>,
    output_tokens: &mut CudaSlice<i32>,
    output_count: &mut CudaSlice<i32>,
    frames: usize,
    padded_frames: usize,
) -> Result<(), Box<dyn Error>> {
    if input_table.len() != input_table_halves() {
        return Err("decoder input table capacity does not match the vocabulary".into());
    }
    if frames >= FP8_MIN_FRAMES {
        // Long decodes look up every previous token's layer-0 input projection
        // instead of streaming that matrix per emitted token. The table is
        // derived per request, like the packed layer-1 weights.
        launch_input_table(stream, linear, weights, input_table)?;
    }
    launch_linear_view(
        stream,
        linear,
        linear_launch_config(padded_frames, JOINT_WIDTH),
        encoder,
        &weights.encoder_projection_weight,
        Some(&weights.encoder_projection_bias),
        &mut encoder_projection.as_view_mut(),
        padded_frames,
        JOINT_WIDTH,
        ENCODER_WIDTH,
        None,
        1.0,
        0.0,
        0,
    )?;

    // Long decoding needs one warp per hidden unit; additional warps only
    // mirror the final unit's work. Keep the short kernel's launch unchanged.
    let blocks = if frames >= FP8_MIN_FRAMES {
        JOINT_WIDTH as u32 / 4
    } else {
        3 * L4_SMS
    };
    let frames = i32::try_from(frames)?;
    let mut builder = stream.launch_builder(decoder);
    builder
        .arg(encoder_projection)
        .arg(&weights.embedding)
        .arg(&weights.weight_ih_l0)
        .arg(&weights.weight_hh_l0)
        .arg(&weights.bias_ih_l0)
        .arg(&weights.bias_hh_l0)
        .arg(&weights.weight_ih_l1)
        .arg(&weights.weight_hh_l1)
        .arg(&weights.bias_ih_l1)
        .arg(&weights.bias_hh_l1)
        .arg(&weights.decoder_projection_weight)
        .arg(&weights.decoder_projection_bias)
        .arg(&weights.joint_weight)
        .arg(&weights.joint_bias)
        .arg(input_table)
        .arg(decoder_workspace)
        .arg(decoder_control)
        .arg(output_tokens)
        .arg(output_count)
        .arg(&frames);
    unsafe {
        builder.launch_cooperative(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn launch_input_table(
    stream: &CudaStream,
    linear: &cudarc::driver::CudaFunction,
    weights: &DecoderWeights<'_>,
    input_table: &mut CudaSlice<f16>,
) -> Result<(), Box<dyn Error>> {
    let null_pointer = 0_u64;
    let m = i32::try_from(INPUT_TABLE_TOKENS)?;
    let n = i32::try_from(GATE_WIDTH)?;
    let k = i32::try_from(JOINT_WIDTH)?;
    let output_scale = 1.0_f32;
    let residual_scale = 0.0_f32;
    let activation = 0_i32;
    let mut builder = stream.launch_builder(linear);
    builder
        .arg(&weights.embedding)
        .arg(&weights.weight_ih_l0)
        .arg(&null_pointer)
        .arg(&null_pointer)
        .arg(input_table)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&output_scale)
        .arg(&residual_scale)
        .arg(&activation);
    unsafe { builder.launch(linear_launch_config(INPUT_TABLE_TOKENS, GATE_WIDTH)) }?;
    Ok(())
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
