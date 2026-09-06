use super::{SM89_CUBIN, UploadedAot, decoder, encoder, frontend, subsampling, upload_aot};
use crate::artifact::{AotArtifactIndex, AotStorage};
use crate::config::{FeatureExtractorConfig, ModelProfile};
use cudarc::cufft::{CudaFft, result as cufft_result, sys as cufft_sys};
use cudarc::driver::{CudaFunction, CudaSlice, sys};
use cudarc::nvrtc::Ptx;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::mem::size_of;
use std::path::Path;

const ENCODER_LAYERS: usize = 24;
// Quantizing the final FF1 expansion exceeded the dev-other quality gate.
const FP8_FF1_LAYERS: usize = ENCODER_LAYERS - 1;
const FP8_FF2_FIRST_LAYER: usize = 0;
const FP8_FF2_LAST_LAYER: usize = ENCODER_LAYERS;

#[derive(Debug, Serialize)]
pub struct PipelineBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub cufft_version: i32,
    pub artifact: String,
    pub artifact_payload_bytes: u64,
    pub artifact_load_seconds: f64,
    pub samples: usize,
    pub audio_seconds: f64,
    pub mel_frames: usize,
    pub valid_mel_frames: usize,
    pub encoder_frames: usize,
    pub valid_encoder_frames: usize,
    pub padded_encoder_frames: usize,
    pub workspace_device_bytes: usize,
    pub model_and_workspace_bytes: u64,
    pub warmup_iterations: usize,
    pub measured_trials: usize,
    pub trial_latency_ms: Vec<f64>,
    pub median_latency_ms: f64,
    pub realtime_factor: f64,
    pub token_count: usize,
    pub token_ids: Vec<i32>,
    pub transcript: String,
    pub reference_token_count: usize,
    pub exact_token_match: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PipelineTranscription {
    pub text: String,
    pub token_ids: Vec<i32>,
    pub audio_seconds: f64,
    pub inference_latency_ms: f64,
}

struct PipelineKernels {
    frame_window: CudaFunction,
    mel_log: CudaFunction,
    normalize: CudaFunction,
    subsample_first: CudaFunction,
    subsample_depthwise: CudaFunction,
    subsample_depthwise_fp8: CudaFunction,
    subsample_flatten: CudaFunction,
    subsample_projection_fp8: CudaFunction,
    subsample_pointwise_fp8: CudaFunction,
    subsample_quantize_pointwise: CudaFunction,
    linear: CudaFunction,
    encoder: encoder::EncoderKernels,
    decoder: CudaFunction,
    decoder_fp8: CudaFunction,
}

struct PipelineWorkspace {
    samples: CudaSlice<f32>,
    framed: CudaSlice<f32>,
    spectrum: CudaSlice<cufft_sys::float2>,
    mel_output: CudaSlice<f32>,
    first_output: CudaSlice<f16>,
    second_depthwise: CudaSlice<f16>,
    second_pointwise: CudaSlice<f16>,
    third_depthwise: CudaSlice<f16>,
    third_pointwise: CudaSlice<f16>,
    flattened: CudaSlice<f16>,
    tile_projection: CudaSlice<f16>,
    encoder: encoder::LayerBuffers,
    encoder_projection: CudaSlice<f16>,
    decoder_input_table: CudaSlice<f16>,
    decoder_workspace: CudaSlice<f32>,
    decoder_control: CudaSlice<i32>,
    output_tokens: CudaSlice<i32>,
    output_count: CudaSlice<i32>,
}

/// A single-model, single-stream L4 inference engine.
///
/// Model weights, CUDA modules, cuBLAS/cuFFT state, and maximum-sized device
/// workspaces are created once. `transcribe` reuses all device allocations, so
/// callers only upload samples and download the emitted token prefix per call.
/// The mutable API deliberately serializes access to the CUDA stream.
pub struct PipelineEngine {
    device: usize,
    uploaded: UploadedAot,
    config: FeatureExtractorConfig,
    vocabulary: Vec<String>,
    kernels: PipelineKernels,
    fft: CudaFft,
    fft_batch_frames: usize,
    window: CudaSlice<f32>,
    filter_offsets: CudaSlice<i32>,
    filter_bins: CudaSlice<i32>,
    filter_values: CudaSlice<f32>,
    ff1_fp8: encoder::QuantizedFfnWeights,
    ff2_int4: encoder::QuantizedFfnWeights,
    workspace: PipelineWorkspace,
    max_samples: usize,
}

impl PipelineEngine {
    pub fn load(
        device: usize,
        artifact_path: &Path,
        max_samples: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let artifact = AotArtifactIndex::open(artifact_path)?;
        if artifact.header.profile != ModelProfile::V2English {
            return Err(format!(
                "pipeline requires the v2-english profile, found {}",
                artifact.header.profile
            )
            .into());
        }
        let config = FeatureExtractorConfig::v2_english();
        let window = artifact.read_f32("frontend.window")?;
        let mel_filters = artifact.read_f32("frontend.mel_filters")?;
        let vocabulary = artifact.header.vocabulary.clone();
        Self::new(
            device,
            artifact_path,
            config,
            &window,
            &mel_filters,
            vocabulary,
            max_samples,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        device: usize,
        artifact_path: &Path,
        config: FeatureExtractorConfig,
        window: &[f32],
        mel_filters: &[f32],
        vocabulary: Vec<String>,
        max_samples: usize,
    ) -> Result<Self, Box<dyn Error>> {
        validate_frontend(&config, window, mel_filters, &vocabulary)?;
        if max_samples == 0 {
            return Err("pipeline workspace capacity must be nonzero".into());
        }

        let max_mel_frames = max_samples / config.hop_length + 1;
        let max_encoder_frames = subsampling::subsample_len(max_mel_frames);
        let max_valid_encoder_frames = subsampling::subsample_len(max_samples / config.hop_length);
        let max_padded_encoder_frames = max_encoder_frames.next_multiple_of(16);
        let fft_batch_frames = max_mel_frames.min(frontend::MAX_CHUNK_FRAMES);
        let uploaded = upload_aot(device, artifact_path)?;
        let (major, minor) = uploaded.context.compute_capability()?;
        if (major, minor) != (8, 9) {
            return Err(format!(
                "the bespoke CUDA pipeline requires compute capability 8.9, found {major}.{minor}"
            )
            .into());
        }

        let frontend_module = uploaded
            .context
            .load_module(Ptx::from_binary(frontend::FRONTEND_CUBIN.to_vec()))?;
        let subsampling_module = uploaded
            .context
            .load_module(Ptx::from_binary(subsampling::SUBSAMPLING_CUBIN.to_vec()))?;
        let encoder_module = uploaded
            .context
            .load_module(Ptx::from_binary(encoder::ENCODER_CUBIN.to_vec()))?;
        let decoder_module = uploaded
            .context
            .load_module(Ptx::from_binary(decoder::DECODER_CUBIN.to_vec()))?;
        let linear_module = uploaded
            .context
            .load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
        let linear = linear_module.load_function("pk_sm89_fp16_linear_epilogue")?;
        let kernels = PipelineKernels {
            frame_window: frontend_module.load_function("pk_sm89_frame_window")?,
            mel_log: frontend_module.load_function("pk_sm89_mel_log_sparse")?,
            normalize: frontend_module.load_function("pk_sm89_normalize_mel")?,
            subsample_first: subsampling_module.load_function("pk_sm89_subsample_first")?,
            subsample_depthwise: subsampling_module.load_function("pk_sm89_subsample_depthwise")?,
            subsample_depthwise_fp8: subsampling_module
                .load_function("pk_sm89_subsample_depthwise_fp8")?,
            subsample_flatten: subsampling_module.load_function("pk_sm89_subsample_flatten")?,
            subsample_projection_fp8: encoder_module
                .load_function("pk_sm89_subsample_projection_fp8")?,
            subsample_pointwise_fp8: encoder_module
                .load_function("pk_sm89_subsample_pointwise_fp8")?,
            subsample_quantize_pointwise: encoder_module
                .load_function("pk_sm89_quantize_rows256")?,
            encoder: encoder::EncoderKernels {
                cublas: super::cublas::Cublas::new(uploaded.stream.clone())?,
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
                ffn_expand_fp8_packed: encoder_module
                    .load_function("pk_sm89_ffn_expand_fp8_packed")?,
                ffn_contract_fp8: encoder_module.load_function("pk_sm89_ffn_contract_fp8")?,
                ffn_expand_int4_sparse: encoder_module
                    .load_function("pk_sm89_ffn_expand_int4_sparse")?,
                ffn_contract_fp8_sparse: encoder_module
                    .load_function("pk_sm89_ffn_contract_fp8_sparse")?,
                quantize_ffn_contract: encoder_module
                    .load_function("pk_sm89_quantize_ffn_contract")?,
                glu: encoder_module.load_function("pk_sm89_glu_masked")?,
                conv_glu_fp8: encoder_module.load_function("pk_sm89_conv_glu_fp8")?,
                quantize_rows1024: encoder_module.load_function("pk_sm89_quantize_rows1024")?,
                depthwise: encoder_module.load_function("pk_sm89_depthwise_batchnorm_silu")?,
                depthwise_pack: encoder_module.load_function("pk_sm89_depthwise_pack")?,
                conv_residual_fp8: encoder_module.load_function("pk_sm89_conv_residual_fp8")?,
                pack_position: encoder_module.load_function("pk_sm89_pack_position_heads")?,
                attention: encoder_module.load_function("pk_sm89_local_relpos_attention")?,
                attention_long: encoder_module
                    .load_function("pk_sm89_local_relpos_attention_tc_scores")?,
                attention_packed: encoder_module
                    .load_function("pk_sm89_local_relpos_attention_packed")?,
                pack_scores: encoder_module.load_function("pk_sm89_pack_score_vectors")?,
                pack_key: encoder_module.load_function("pk_sm89_pack_key_int4")?,
                attention_output_fp8: encoder_module
                    .load_function("pk_sm89_attention_output_fp8")?,
                qkv_fp8: encoder_module.load_function("pk_sm89_qkv_fp8")?,
                linear: linear.clone(),
                linear_large: linear_module.load_function("pk_sm89_fp16_linear_epilogue_m64")?,
                qkv: linear_module.load_function("pk_sm89_fp16_qkv")?,
            },
            decoder: decoder_module.load_function("pk_sm89_tdt_persistent_v2")?,
            decoder_fp8: decoder_module.load_function("pk_sm89_tdt_persistent_fp8_ih")?,
            linear,
        };

        let mut centered_window = vec![0.0_f32; frontend::FFT_SIZE];
        let window_left = (frontend::FFT_SIZE - window.len()) / 2;
        centered_window[window_left..window_left + window.len()].copy_from_slice(window);
        let window = uploaded.stream.clone_htod(&centered_window)?;
        let (filter_offsets, filter_bins, filter_values) = frontend::sparse_filters(mel_filters)?;
        let filter_offsets = uploaded.stream.clone_htod(&filter_offsets)?;
        let filter_bins = uploaded.stream.clone_htod(&filter_bins)?;
        let filter_values = uploaded.stream.clone_htod(&filter_values)?;
        let fft = CudaFft::plan_1d(
            i32::try_from(frontend::FFT_SIZE)?,
            cufft_sys::cufftType::CUFFT_R2C,
            i32::try_from(fft_batch_frames)?,
            uploaded.stream.clone(),
        )?;

        let first_features = subsampling::INPUT_FEATURES.div_ceil(2);
        let second_features = first_features.div_ceil(2);
        let max_second_frames = 2 * subsampling::TILE_OUTPUT_FRAMES + 1;
        let max_first_frames = 2 * max_second_frames + 1;
        let padded_tile_frames = subsampling::TILE_OUTPUT_FRAMES.next_multiple_of(16);
        let mut workspace = PipelineWorkspace {
            samples: uploaded.stream.alloc_zeros(max_samples)?,
            framed: uploaded
                .stream
                .alloc_zeros(fft_batch_frames * frontend::FFT_SIZE)?,
            spectrum: uploaded
                .stream
                .alloc_zeros(fft_batch_frames * frontend::FFT_BINS)?,
            mel_output: uploaded
                .stream
                .alloc_zeros(max_mel_frames * frontend::MEL_BINS)?,
            first_output: uploaded
                .stream
                .alloc_zeros(max_first_frames * first_features * subsampling::CHANNELS)?,
            second_depthwise: uploaded
                .stream
                .alloc_zeros(max_second_frames * second_features * subsampling::CHANNELS)?,
            second_pointwise: uploaded
                .stream
                .alloc_zeros(max_second_frames * second_features * subsampling::CHANNELS)?,
            third_depthwise: uploaded.stream.alloc_zeros(
                subsampling::TILE_OUTPUT_FRAMES
                    * subsampling::OUTPUT_FEATURES
                    * subsampling::CHANNELS,
            )?,
            third_pointwise: uploaded.stream.alloc_zeros(
                subsampling::TILE_OUTPUT_FRAMES
                    * subsampling::OUTPUT_FEATURES
                    * subsampling::CHANNELS,
            )?,
            flattened: uploaded
                .stream
                .alloc_zeros(padded_tile_frames * subsampling::FLATTENED_WIDTH)?,
            tile_projection: uploaded
                .stream
                .alloc_zeros(padded_tile_frames * encoder::MODEL_WIDTH)?,
            encoder: encoder::LayerBuffers {
                source: None,
                state_a: uploaded
                    .stream
                    .alloc_zeros(max_padded_encoder_frames * encoder::MODEL_WIDTH)?,
                state_b: uploaded
                    .stream
                    .alloc_zeros(max_padded_encoder_frames * encoder::MODEL_WIDTH)?,
                normalized: uploaded
                    .stream
                    .alloc_zeros(max_padded_encoder_frames * encoder::MODEL_WIDTH)?,
                workspace: uploaded
                    .stream
                    .alloc_zeros(max_padded_encoder_frames * encoder::FF_WIDTH)?,
                position_cache: uploaded.stream.alloc_zeros(
                    ENCODER_LAYERS
                        * encoder::POSITION_ROWS.next_multiple_of(16)
                        * encoder::MODEL_WIDTH,
                )?,
            },
            encoder_projection: uploaded
                .stream
                .alloc_zeros(max_padded_encoder_frames * decoder::JOINT_WIDTH)?,
            decoder_input_table: uploaded.stream.alloc_zeros(decoder::input_table_halves())?,
            decoder_workspace: uploaded
                .stream
                .alloc_zeros(decoder::workspace_floats(max_valid_encoder_frames))?,
            decoder_control: uploaded
                .stream
                .alloc_zeros(decoder::DECODER_CONTROL_VALUES)?,
            output_tokens: uploaded
                .stream
                .alloc_zeros((max_valid_encoder_frames * decoder::MAX_SYMBOLS).max(1))?,
            output_count: uploaded.stream.alloc_zeros(1)?,
        };
        let encoder_weights = (0..ENCODER_LAYERS)
            .map(|layer| encoder::LayerWeights::load(&uploaded, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let ff1_fp8 =
            encoder::QuantizedFfnWeights::load(&uploaded, artifact_path, 0..FP8_FF1_LAYERS)?;
        let ff2_int4 = encoder::QuantizedFfnWeights::load_ff2(
            &uploaded,
            artifact_path,
            FP8_FF2_FIRST_LAYER..FP8_FF2_LAST_LAYER,
        )?;
        let position_input = uploaded
            .stream
            .clone_htod(&encoder::relative_position_encoding())?;
        encoder::cache_position_projections(
            &uploaded.stream,
            &kernels.linear,
            &kernels.encoder.pack_position,
            &encoder_weights,
            &position_input,
            &mut workspace.encoder.position_cache,
        )?;
        uploaded.stream.synchronize()?;

        Ok(Self {
            device,
            uploaded,
            config,
            vocabulary,
            kernels,
            fft,
            fft_batch_frames,
            window,
            filter_offsets,
            filter_bins,
            filter_values,
            ff1_fp8,
            ff2_int4,
            workspace,
            max_samples,
        })
    }

    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<PipelineTranscription, Box<dyn Error>> {
        self.prepare_samples(samples)?;
        let started = self
            .uploaded
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        self.run_prepared(samples.len())?;
        let ended = self
            .uploaded
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let inference_latency_ms = f64::from(started.elapsed_ms(&ended)?);
        let token_ids = self.read_tokens()?;
        let text = decoder::decode_tokens(&self.vocabulary, &token_ids)?;
        Ok(PipelineTranscription {
            text,
            token_ids,
            audio_seconds: samples.len() as f64 / self.config.sampling_rate as f64,
            inference_latency_ms,
        })
    }

    fn prepare_samples(&mut self, samples: &[f32]) -> Result<(), Box<dyn Error>> {
        if samples.is_empty() {
            return Err("audio must contain at least one sample".into());
        }
        if samples.len() > self.max_samples {
            return Err(format!(
                "audio has {} samples, exceeding the configured capacity of {}",
                samples.len(),
                self.max_samples
            )
            .into());
        }
        self.uploaded
            .stream
            .memcpy_htod(samples, &mut self.workspace.samples)?;
        Ok(())
    }

    fn run_prepared(&mut self, sample_count: usize) -> Result<(), Box<dyn Error>> {
        let mel_frames = sample_count / self.config.hop_length + 1;
        let valid_mel_frames = sample_count / self.config.hop_length;
        let encoder_frames = subsampling::subsample_len(mel_frames);
        let valid_encoder_frames = subsampling::subsample_len(valid_mel_frames);
        let padded_encoder_frames = encoder_frames.next_multiple_of(16);

        let uploaded = &self.uploaded;
        let kernels = &self.kernels;
        let workspace = &mut self.workspace;
        let first_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.0.weight",
            AotStorage::Fp16,
        )?;
        let first_bias = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.0.bias",
            AotStorage::Fp16,
        )?;
        let second_depthwise_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.2.weight",
            AotStorage::Fp16,
        )?;
        let second_depthwise_bias = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.2.bias",
            AotStorage::Fp16,
        )?;
        let second_pointwise_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.3.weight",
            AotStorage::Sm89Fp16Linear,
        )?;
        let second_pointwise_bias = subsampling::fp32_tensor(
            uploaded,
            "encoder.subsampling.layers.3.bias",
            AotStorage::Sm89Fp32Bias,
        )?;
        let third_depthwise_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.5.weight",
            AotStorage::Fp16,
        )?;
        let third_depthwise_bias = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.5.bias",
            AotStorage::Fp16,
        )?;
        let third_pointwise_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.layers.6.weight",
            AotStorage::Sm89Fp16Linear,
        )?;
        let third_pointwise_bias = subsampling::fp32_tensor(
            uploaded,
            "encoder.subsampling.layers.6.bias",
            AotStorage::Sm89Fp32Bias,
        )?;
        let projection_weight = subsampling::fp16_tensor(
            uploaded,
            "encoder.subsampling.linear.weight",
            AotStorage::Sm89Fp16Linear,
        )?;
        let projection_bias = subsampling::fp32_tensor(
            uploaded,
            "encoder.subsampling.linear.bias",
            AotStorage::Sm89Fp32Bias,
        )?;
        let encoder_weights = (0..ENCODER_LAYERS)
            .map(|layer| encoder::LayerWeights::load(uploaded, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let decoder_weights = decoder::DecoderWeights::load(uploaded)?;

        frontend::run_frontend(
            &uploaded.stream,
            &self.fft,
            &kernels.frame_window,
            &kernels.mel_log,
            &kernels.normalize,
            &workspace.samples,
            &self.window,
            &self.filter_offsets,
            &self.filter_bins,
            &self.filter_values,
            &mut workspace.framed,
            &mut workspace.spectrum,
            &mut workspace.mel_output,
            sample_count,
            mel_frames,
            self.fft_batch_frames,
            valid_mel_frames,
            self.config.preemphasis,
            self.config.log_zero_guard,
            self.config.normalize_epsilon,
        )?;
        subsampling::run_subsampling(
            &uploaded.stream,
            &kernels.subsample_first,
            &kernels.subsample_depthwise,
            &kernels.subsample_flatten,
            &kernels.linear,
            &workspace.mel_output,
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
            &mut workspace.first_output,
            &mut workspace.second_depthwise,
            &mut workspace.second_pointwise,
            &mut workspace.third_depthwise,
            &mut workspace.third_pointwise,
            &mut workspace.flattened,
            &mut workspace.tile_projection,
            &mut workspace.encoder.state_a,
            mel_frames,
            valid_encoder_frames,
            if encoder_frames >= subsampling::FP8_MIN_FRAMES {
                Some(subsampling::Fp8Projection {
                    quantize: &kernels.encoder.quantize_ffn_contract,
                    linear: &kernels.subsample_projection_fp8,
                    pointwise_quantize: &kernels.subsample_quantize_pointwise,
                    pointwise_linear: &kernels.subsample_pointwise_fp8,
                    depthwise_quantize: &kernels.subsample_depthwise_fp8,
                    workspace: &mut workspace.encoder.workspace,
                })
            } else {
                None
            },
        )?;
        encoder::run_layers(
            &uploaded.stream,
            &kernels.encoder,
            &encoder_weights,
            &self.ff1_fp8.layers,
            Some((FP8_FF2_FIRST_LAYER, &self.ff2_int4.layers)),
            &mut workspace.encoder,
            encoder_frames,
            valid_encoder_frames,
            padded_encoder_frames,
        )?;
        decoder::launch_decode(
            &uploaded.stream,
            &kernels.linear,
            // Packing is paid per request; short decodes cannot amortize it.
            if valid_encoder_frames >= decoder::FP8_MIN_FRAMES {
                &kernels.decoder_fp8
            } else {
                &kernels.decoder
            },
            &decoder_weights,
            &workspace.encoder.state_b,
            &mut workspace.encoder_projection,
            &mut workspace.decoder_input_table,
            &mut workspace.decoder_workspace,
            &mut workspace.decoder_control,
            &mut workspace.output_tokens,
            &mut workspace.output_count,
            valid_encoder_frames,
            padded_encoder_frames,
        )?;
        Ok(())
    }

    fn read_tokens(&self) -> Result<Vec<i32>, Box<dyn Error>> {
        let count = self
            .uploaded
            .stream
            .clone_dtoh(&self.workspace.output_count)?;
        let token_count = usize::try_from(count[0])?;
        if token_count > self.workspace.output_tokens.len() {
            return Err(format!(
                "decoder produced {token_count} tokens into capacity {}",
                self.workspace.output_tokens.len()
            )
            .into());
        }
        Ok(self
            .uploaded
            .stream
            .clone_dtoh(&self.workspace.output_tokens.slice(..token_count))?)
    }

    fn workspace_device_bytes(&self) -> usize {
        let workspace = &self.workspace;
        workspace.samples.len() * size_of::<f32>()
            + self.window.len() * size_of::<f32>()
            + self.filter_offsets.len() * size_of::<i32>()
            + self.filter_bins.len() * size_of::<i32>()
            + self.filter_values.len() * size_of::<f32>()
            + workspace.framed.len() * size_of::<f32>()
            + workspace.spectrum.len() * size_of::<cufft_sys::float2>()
            + workspace.mel_output.len() * size_of::<f32>()
            + workspace.first_output.len() * size_of::<f16>()
            + workspace.second_depthwise.len() * size_of::<f16>()
            + workspace.second_pointwise.len() * size_of::<f16>()
            + workspace.third_depthwise.len() * size_of::<f16>()
            + workspace.third_pointwise.len() * size_of::<f16>()
            + workspace.flattened.len() * size_of::<f16>()
            + workspace.tile_projection.len() * size_of::<f16>()
            + workspace
                .encoder
                .source
                .as_ref()
                .map_or(0, |source| source.len() * size_of::<f16>())
            + workspace.encoder.state_a.len() * size_of::<f16>()
            + workspace.encoder.state_b.len() * size_of::<f16>()
            + workspace.encoder.normalized.len() * size_of::<f16>()
            + workspace.encoder.workspace.len() * size_of::<f16>()
            + workspace.encoder.position_cache.len() * size_of::<f16>()
            + self.ff1_fp8.device_bytes()
            + self.ff2_int4.device_bytes()
            + workspace.encoder_projection.len() * size_of::<f16>()
            + workspace.decoder_input_table.len() * size_of::<f16>()
            + workspace.decoder_workspace.len() * size_of::<f32>()
            + workspace.decoder_control.len() * size_of::<i32>()
            + workspace.output_tokens.len() * size_of::<i32>()
            + workspace.output_count.len() * size_of::<i32>()
    }
}

fn validate_frontend(
    config: &FeatureExtractorConfig,
    window: &[f32],
    mel_filters: &[f32],
    vocabulary: &[String],
) -> Result<(), Box<dyn Error>> {
    if config.n_fft != frontend::FFT_SIZE
        || config.feature_size != frontend::MEL_BINS
        || config.hop_length != 160
        || window.len() != config.win_length
        || mel_filters.len() != frontend::MEL_BINS * frontend::FFT_BINS
        || vocabulary.len() != 1_024
    {
        return Err("pipeline requires the pinned v2 frontend and vocabulary dimensions".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_pipeline(
    device: usize,
    artifact_path: &Path,
    config: &FeatureExtractorConfig,
    samples: &[f32],
    window: &[f32],
    mel_filters: &[f32],
    vocabulary: &[String],
    reference_tokens: Option<&[i32]>,
    warmup_iterations: usize,
    measured_trials: usize,
) -> Result<PipelineBenchmarkReport, Box<dyn Error>> {
    if samples.is_empty() || warmup_iterations == 0 || measured_trials == 0 {
        return Err("audio and pipeline iteration counts must be nonzero".into());
    }
    let mut engine = PipelineEngine::new(
        device,
        artifact_path,
        config.clone(),
        window,
        mel_filters,
        vocabulary.to_vec(),
        samples.len(),
    )?;
    engine.prepare_samples(samples)?;

    for _ in 0..warmup_iterations {
        engine.run_prepared(samples.len())?;
    }
    engine.uploaded.stream.synchronize()?;
    let mut trial_latency_ms = Vec::with_capacity(measured_trials);
    for _ in 0..measured_trials {
        let started = engine
            .uploaded
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        engine.run_prepared(samples.len())?;
        let ended = engine
            .uploaded
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        trial_latency_ms.push(f64::from(started.elapsed_ms(&ended)?));
    }

    let token_ids = engine.read_tokens()?;
    let exact_token_match = reference_tokens.map(|expected| expected == token_ids);
    if exact_token_match == Some(false) {
        let expected = reference_tokens.expect("reference exists");
        let first = expected
            .iter()
            .zip(&token_ids)
            .position(|(left, right)| left != right)
            .unwrap_or(expected.len().min(token_ids.len()));
        return Err(format!(
            "pipeline token parity failed at token {first}: expected {:?}, observed {:?}",
            expected.get(first),
            token_ids.get(first)
        )
        .into());
    }
    let transcript = decoder::decode_tokens(vocabulary, &token_ids)?;
    let mut ordered = trial_latency_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let median_latency_ms = ordered[ordered.len() / 2];
    let audio_seconds = samples.len() as f64 / config.sampling_rate as f64;
    let mel_frames = samples.len() / config.hop_length + 1;
    let valid_mel_frames = samples.len() / config.hop_length;
    let encoder_frames = subsampling::subsample_len(mel_frames);
    let valid_encoder_frames = subsampling::subsample_len(valid_mel_frames);
    let padded_encoder_frames = encoder_frames.next_multiple_of(16);
    let workspace_device_bytes = engine.workspace_device_bytes();
    let model_and_workspace_bytes = engine
        .uploaded
        .artifact
        .header
        .payload_bytes
        .checked_add(u64::try_from(workspace_device_bytes)?)
        .ok_or("model and workspace byte count overflow")?;
    let (major, minor) = engine.uploaded.context.compute_capability()?;

    Ok(PipelineBenchmarkReport {
        schema_version: 1,
        device: engine.device,
        name: engine.uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        cufft_version: cufft_result::get_version()?,
        artifact: artifact_path.display().to_string(),
        artifact_payload_bytes: engine.uploaded.artifact.header.payload_bytes,
        artifact_load_seconds: engine.uploaded.wall_seconds,
        samples: samples.len(),
        audio_seconds,
        mel_frames,
        valid_mel_frames,
        encoder_frames,
        valid_encoder_frames,
        padded_encoder_frames,
        workspace_device_bytes,
        model_and_workspace_bytes,
        warmup_iterations,
        measured_trials,
        trial_latency_ms,
        median_latency_ms,
        realtime_factor: audio_seconds / (median_latency_ms / 1000.0),
        token_count: token_ids.len(),
        token_ids,
        transcript,
        reference_token_count: reference_tokens.map_or(0, <[i32]>::len),
        exact_token_match,
    })
}
