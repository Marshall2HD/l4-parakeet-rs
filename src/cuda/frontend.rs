use crate::config::FeatureExtractorConfig;
use crate::frontend::MelFeatures;
use cudarc::cufft::{CudaFft, result as cufft_result, sys as cufft_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use serde::Serialize;
use std::error::Error;
use std::mem::size_of;
use std::sync::Arc;

pub(super) const FRONTEND_CUBIN: &[u8] = include_bytes!(env!("PARAKEET_SM89_FRONTEND_CUBIN"));
pub(super) const FFT_SIZE: usize = 512;
pub(super) const FFT_BINS: usize = 257;
pub(super) const MEL_BINS: usize = 128;
pub(super) const MAX_CHUNK_FRAMES: usize = 8_192;

#[derive(Debug, Serialize)]
pub struct GpuFrontendReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub cufft_version: i32,
    pub samples: usize,
    pub frames: usize,
    pub valid_frames: usize,
    pub chunk_frames: usize,
    pub explicit_device_bytes: usize,
    pub warmup_iterations: usize,
    pub warm_iterations: usize,
    pub warm_latency_ms: f64,
    pub realtime_factor: f64,
    pub correctness_values: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f64,
}

pub fn benchmark_frontend(
    device: usize,
    config: &FeatureExtractorConfig,
    samples: &[f32],
    window: &[f32],
    mel_filters: &[f32],
    reference: &MelFeatures,
    warmup_iterations: usize,
    warm_iterations: usize,
) -> Result<GpuFrontendReport, Box<dyn Error>> {
    if samples.is_empty() || warmup_iterations == 0 || warm_iterations == 0 {
        return Err("audio and benchmark iteration counts must be nonzero".into());
    }
    if config.n_fft != FFT_SIZE
        || config.feature_size != MEL_BINS
        || config.hop_length != 160
        || window.len() != config.win_length
        || mel_filters.len() != MEL_BINS * FFT_BINS
    {
        return Err("GPU frontend requires the pinned v2 dimensions".into());
    }
    let frames = samples.len() / config.hop_length + 1;
    let valid_frames = samples.len() / config.hop_length;
    let chunk_frames = frames.min(MAX_CHUNK_FRAMES);
    if (reference.mel_bins, reference.frames, reference.valid_frames)
        != (MEL_BINS, frames, valid_frames)
    {
        return Err("CPU frontend reference shape does not match input".into());
    }

    let context = CudaContext::new(device)?;
    let (major, minor) = context.compute_capability()?;
    if (major, minor) != (8, 9) {
        return Err(format!(
            "device {device} is compute capability {major}.{minor}; parakeet-l4 requires an L4-class sm_89 GPU"
        )
        .into());
    }
    let module = context.load_module(Ptx::from_binary(FRONTEND_CUBIN.to_vec()))?;
    let frame_window = module.load_function("pk_sm89_frame_window")?;
    let mel_log = module.load_function("pk_sm89_mel_log_sparse")?;
    let normalize = module.load_function("pk_sm89_normalize_mel")?;
    let stream = context.default_stream();

    let samples_device = stream.clone_htod(samples)?;
    let mut centered_window = vec![0.0_f32; FFT_SIZE];
    let window_left = (FFT_SIZE - window.len()) / 2;
    centered_window[window_left..window_left + window.len()].copy_from_slice(window);
    let window_device = stream.clone_htod(&centered_window)?;
    let (filter_offsets, filter_bins, filter_values) = sparse_filters(mel_filters)?;
    let filter_offsets_device = stream.clone_htod(&filter_offsets)?;
    let filter_bins_device = stream.clone_htod(&filter_bins)?;
    let filter_values_device = stream.clone_htod(&filter_values)?;
    let mut framed = stream.alloc_zeros::<f32>(chunk_frames * FFT_SIZE)?;
    let mut spectrum = stream.alloc_zeros::<cufft_sys::float2>(chunk_frames * FFT_BINS)?;
    let mut output = stream.alloc_zeros::<f32>(frames * MEL_BINS)?;
    let fft = CudaFft::plan_1d(
        i32::try_from(FFT_SIZE)?,
        cufft_sys::cufftType::CUFFT_R2C,
        i32::try_from(chunk_frames)?,
        stream.clone(),
    )?;

    for _ in 0..warmup_iterations {
        run_frontend(
            &stream,
            &fft,
            &frame_window,
            &mel_log,
            &normalize,
            &samples_device,
            &window_device,
            &filter_offsets_device,
            &filter_bins_device,
            &filter_values_device,
            &mut framed,
            &mut spectrum,
            &mut output,
            samples.len(),
            frames,
            chunk_frames,
            valid_frames,
            config.preemphasis,
            config.log_zero_guard,
            config.normalize_epsilon,
        )?;
    }
    stream.synchronize()?;

    let started = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..warm_iterations {
        run_frontend(
            &stream,
            &fft,
            &frame_window,
            &mel_log,
            &normalize,
            &samples_device,
            &window_device,
            &filter_offsets_device,
            &filter_bins_device,
            &filter_values_device,
            &mut framed,
            &mut spectrum,
            &mut output,
            samples.len(),
            frames,
            chunk_frames,
            valid_frames,
            config.preemphasis,
            config.log_zero_guard,
            config.normalize_epsilon,
        )?;
    }
    let ended = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let warm_latency_ms = f64::from(started.elapsed_ms(&ended)?) / warm_iterations as f64;

    let actual = stream.clone_dtoh(&output)?;
    let mut max_abs_error = 0.0_f32;
    let mut sum_abs_error = 0.0_f64;
    for frame in 0..frames {
        for mel in 0..MEL_BINS {
            let expected = reference.values[mel * frames + frame];
            let observed = actual[frame * MEL_BINS + mel];
            let error = (observed - expected).abs();
            if !observed.is_finite() {
                return Err(
                    format!("GPU frontend produced a non-finite value at [{mel},{frame}]").into(),
                );
            }
            max_abs_error = max_abs_error.max(error);
            sum_abs_error += f64::from(error);
        }
    }
    if max_abs_error > 0.05 {
        return Err(format!(
            "GPU frontend parity error {max_abs_error} exceeds the 0.05 bring-up gate"
        )
        .into());
    }

    let audio_seconds = samples.len() as f64 / config.sampling_rate as f64;
    Ok(GpuFrontendReport {
        schema_version: 1,
        device,
        name: context.name()?,
        compute_capability: format!("{major}.{minor}"),
        cufft_version: cufft_result::get_version()?,
        samples: samples.len(),
        frames,
        valid_frames,
        chunk_frames,
        explicit_device_bytes: samples.len() * size_of::<f32>()
            + centered_window.len() * size_of::<f32>()
            + filter_offsets.len() * size_of::<i32>()
            + filter_bins.len() * size_of::<i32>()
            + filter_values.len() * size_of::<f32>()
            + chunk_frames * FFT_SIZE * size_of::<f32>()
            + chunk_frames * FFT_BINS * size_of::<cufft_sys::float2>()
            + frames * MEL_BINS * size_of::<f32>(),
        warmup_iterations,
        warm_iterations,
        warm_latency_ms,
        realtime_factor: audio_seconds / (warm_latency_ms / 1000.0),
        correctness_values: actual.len(),
        max_abs_error,
        mean_abs_error: sum_abs_error / actual.len() as f64,
    })
}

pub(super) fn sparse_filters(
    filters: &[f32],
) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), Box<dyn Error>> {
    let mut offsets = Vec::with_capacity(MEL_BINS + 1);
    let mut bins = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for filter in filters.chunks_exact(FFT_BINS) {
        for (bin, &value) in filter.iter().enumerate() {
            if value != 0.0 {
                bins.push(i32::try_from(bin)?);
                values.push(value);
            }
        }
        offsets.push(i32::try_from(bins.len())?);
    }
    Ok((offsets, bins, values))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_frontend(
    stream: &Arc<CudaStream>,
    fft: &CudaFft,
    frame_window: &CudaFunction,
    mel_log: &CudaFunction,
    normalize: &CudaFunction,
    samples: &CudaSlice<f32>,
    window: &CudaSlice<f32>,
    filter_offsets: &CudaSlice<i32>,
    filter_bins: &CudaSlice<i32>,
    filter_values: &CudaSlice<f32>,
    framed: &mut CudaSlice<f32>,
    spectrum: &mut CudaSlice<cufft_sys::float2>,
    output: &mut CudaSlice<f32>,
    sample_count: usize,
    frame_count: usize,
    chunk_frames: usize,
    valid_frames: usize,
    preemphasis: f32,
    log_zero_guard: f32,
    normalize_epsilon: f32,
) -> Result<(), Box<dyn Error>> {
    let threads = 256_u32;
    let sample_count = i32::try_from(sample_count)?;
    let frame_count_i32 = i32::try_from(frame_count)?;
    let valid_frames = i32::try_from(valid_frames)?;
    for frame_offset in (0..frame_count).step_by(chunk_frames) {
        let active_frames = chunk_frames.min(frame_count - frame_offset);
        let active_values = active_frames
            .checked_mul(FFT_SIZE)
            .ok_or("frame buffer size overflow")?;
        let active_frames = i32::try_from(active_frames)?;
        let frame_offset = i32::try_from(frame_offset)?;

        let mut builder = stream.launch_builder(frame_window);
        builder
            .arg(samples)
            .arg(window)
            .arg(&mut *framed)
            .arg(&sample_count)
            .arg(&active_frames)
            .arg(&frame_offset)
            .arg(&preemphasis);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (u32::try_from(active_values)?.div_ceil(threads), 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;

        fft.exec_r2c(&mut *framed, &mut *spectrum)?;

        let mut builder = stream.launch_builder(mel_log);
        builder
            .arg(&*spectrum)
            .arg(filter_offsets)
            .arg(filter_bins)
            .arg(filter_values)
            .arg(&mut *output)
            .arg(&active_frames)
            .arg(&frame_offset)
            .arg(&log_zero_guard);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (u32::try_from(active_frames)?, 1, 1),
                block_dim: (MEL_BINS as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
    }

    let mut builder = stream.launch_builder(normalize);
    builder
        .arg(&mut *output)
        .arg(&frame_count_i32)
        .arg(&valid_frames)
        .arg(&normalize_epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (MEL_BINS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}
