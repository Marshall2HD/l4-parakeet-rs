use clap::{Parser, Subcommand};
use parakeet_l4::artifact::{AotArtifactIndex, pack_fp16};
use parakeet_l4::config::{ModelBundleConfig, ModelProfile};
use parakeet_l4::weights::SafeTensorIndex;
use serde::Serialize;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(
    name = "parakeet-l4",
    version,
    about = "L4-specific Parakeet TDT engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate normalized config and SafeTensors without loading the weight payload.
    Inspect {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = ModelProfile::V2English)]
        profile: ModelProfile,
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=8))]
        exl3_bits: u8,
        #[arg(long)]
        json: bool,
    },
    /// Pack normalized F32 weights into the deterministic sm_89 FP16 artifact.
    PackFp16 {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = ModelProfile::V2English)]
        profile: ModelProfile,
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate and print a bespoke L4 AOT artifact header.
    InspectAot {
        #[arg(long)]
        artifact: PathBuf,
    },
    /// Score one complete pipeline report against the immutable optimization gates.
    EvaluateNightCircus {
        #[arg(long, default_value = "docs/benchmarks/night-circus-1h.json")]
        manifest: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
    },
    /// Run the deterministic Rust v2 log-mel parity frontend on PCM s16le mono WAV.
    FrontendCpu {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = ModelProfile::V2English)]
        profile: ModelProfile,
        /// Optionally write feature-major little-endian F32 values.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark the fused framing/cuFFT/mel/normalization path against the Rust oracle.
    BenchFrontend {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 20)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 100)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Benchmark tiled dw-striding convolution and projection against an optional F32 oracle.
    BenchSubsampling {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 20)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Benchmark one L4-specialized FastConformer layer against an optional F32 oracle.
    BenchEncoderLayer {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        rows: usize,
        #[arg(long)]
        valid_rows: usize,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 1)]
        layers: usize,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 20)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Benchmark the persistent L4 TDT decoder against optional token IDs.
    BenchDecoder {
        #[arg(long)]
        artifact: PathBuf,
        /// Row-major F32 encoder output with 1024 values per frame.
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        frames: usize,
        /// Optional little-endian I32 token oracle.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Optional SentencePiece .vocab used to render the transcript.
        #[arg(long)]
        vocabulary: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 20)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Transcribe a WAV through the complete bespoke v2 L4 pipeline.
    Transcribe {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        input: PathBuf,
        /// Optional little-endian I32 token oracle.
        #[arg(long)]
        reference_tokens: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 5)]
        measured_trials: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Transcribe an ordered short-clip batch with a packed GPU encoder.
    TranscribeBatch {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        #[arg(long, default_value_t = 1)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 1)]
        measured_trials: usize,
        /// Compare every trial's token IDs with individual inference, failing on mismatch.
        #[arg(long)]
        verify_single: bool,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Serve the persistent L4 pipeline through OpenAI-compatible Whisper endpoints.
    Serve {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 3_600)]
        max_audio_seconds: usize,
        #[arg(long, default_value_t = parakeet_l4::server::DEFAULT_MAX_UPLOAD_BYTES)]
        max_upload_bytes: usize,
    },
    /// Upload one complete AOT weight payload into a contiguous L4 allocation.
    LoadAot {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Run a packed real-model FP16 matrix through the native L4 kernel and check parity.
    BenchAotLinear {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        tensor: String,
        #[arg(long, default_value_t = 16)]
        rows: usize,
        #[arg(long, default_value_t = 20)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 200)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Verify the embedded native cubin and report its resource usage on an L4.
    CudaInfo {
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Check and benchmark the bespoke FP16 linear baseline on an NVIDIA L4.
    BenchFp16Linear {
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 20)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 200)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 20)]
        cold_iterations: usize,
    },
    /// Compare native FP8 E4M3 and INT8 Tensor Core candidates on an NVIDIA L4.
    BenchQuantizedLinear {
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 20)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 200)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 20)]
        cold_iterations: usize,
    },
    /// Benchmark direct GGUF Q4_K superblocks with the bespoke L4 kernel.
    BenchQ4KLinear {
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = 20)]
        warmup_iterations: usize,
        #[arg(long, default_value_t = 200)]
        warm_iterations: usize,
        #[arg(long, default_value_t = 20)]
        cold_iterations: usize,
    },
}

#[derive(Debug, Serialize)]
struct Inspection<'a> {
    config: &'a ModelBundleConfig,
    manifest: parakeet_l4::weights::ManifestReport,
    deployment_plan: parakeet_l4::weights::WeightPlanSummary,
}

#[derive(Debug, Serialize)]
struct FrontendReport {
    input: PathBuf,
    sample_rate: u32,
    samples: usize,
    audio_seconds: f64,
    mel_bins: usize,
    frames: usize,
    valid_frames: usize,
    layout: &'static str,
    values: usize,
    blake3: String,
    minimum: f32,
    maximum: f32,
    mean: f64,
    rms: f64,
    processing_seconds: f64,
    realtime_factor: f64,
    output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Inspect {
            model_dir,
            profile,
            exl3_bits,
            json,
        } => {
            let config = ModelBundleConfig::load(&model_dir, profile)?;
            let weights = SafeTensorIndex::open(&model_dir.join("model.safetensors"))?;
            let inspection = Inspection {
                config: &config,
                manifest: weights.validate_model(profile)?,
                deployment_plan: weights.plan(exl3_bits)?,
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&inspection)?);
            } else {
                let plan = &inspection.deployment_plan;
                println!("Parakeet TDT 0.6B {profile} contract: valid");
                println!(
                    "SafeTensors: {} tensors, {} parameters",
                    inspection.manifest.tensors, inspection.manifest.parameters
                );
                println!(
                    "Direct F32 source payload: {:.1} MiB",
                    plan.source_bytes as f64 / 1_048_576.0
                );
                println!(
                    "Fixed non-linear payload in packed candidates: {:.1} MiB",
                    plan.fixed_deployment_bytes as f64 / 1_048_576.0
                );
                for candidate in &plan.candidates {
                    println!(
                        "  {:<34} {:>7.1} MiB ({:>5.1}% of source, {} FP16 fallbacks)",
                        candidate.format,
                        candidate.deployment_bytes as f64 / 1_048_576.0,
                        candidate.fraction_of_source * 100.0,
                        candidate.fallback_tensors
                    );
                }
                println!(
                    "Payload size is not a speed ranking; select formats from L4 kernel and WER measurements."
                );
            }
        }
        Command::PackFp16 {
            model_dir,
            profile,
            output,
        } => {
            ModelBundleConfig::load(&model_dir, profile)?;
            let weights = SafeTensorIndex::open(&model_dir.join("model.safetensors"))?;
            let vocabulary = read_vocabulary(&model_dir.join("tokenizer.vocab"), profile)?;
            let report = pack_fp16(&weights, profile, &vocabulary, &output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::InspectAot { artifact } => {
            let index = AotArtifactIndex::open(&artifact)?;
            println!("{}", serde_json::to_string_pretty(&index)?);
        }
        Command::EvaluateNightCircus {
            manifest,
            reference,
            candidate,
        } => {
            let report =
                parakeet_l4::evaluation::evaluate_night_circus(&manifest, &reference, &candidate)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::FrontendCpu {
            model_dir,
            input,
            profile,
            output,
        } => frontend_cpu(&model_dir, &input, profile, output.as_deref())?,
        Command::BenchFrontend {
            artifact,
            input,
            warmup_iterations,
            warm_iterations,
            device,
        } => bench_frontend(
            &artifact,
            &input,
            warmup_iterations,
            warm_iterations,
            device,
        )?,
        Command::BenchSubsampling {
            artifact,
            input,
            reference,
            warmup_iterations,
            warm_iterations,
            device,
        } => bench_subsampling(
            &artifact,
            &input,
            reference.as_deref(),
            warmup_iterations,
            warm_iterations,
            device,
        )?,
        Command::BenchEncoderLayer {
            artifact,
            input,
            rows,
            valid_rows,
            layer,
            layers,
            reference,
            warmup_iterations,
            warm_iterations,
            device,
        } => bench_encoder_layer(
            &artifact,
            &input,
            rows,
            valid_rows,
            layer,
            layers,
            reference.as_deref(),
            warmup_iterations,
            warm_iterations,
            device,
        )?,
        Command::BenchDecoder {
            artifact,
            input,
            frames,
            reference,
            vocabulary,
            warmup_iterations,
            warm_iterations,
            device,
        } => bench_decoder(
            &artifact,
            &input,
            frames,
            reference.as_deref(),
            vocabulary.as_deref(),
            warmup_iterations,
            warm_iterations,
            device,
        )?,
        Command::Transcribe {
            artifact,
            input,
            reference_tokens,
            warmup_iterations,
            measured_trials,
            device,
        } => transcribe(
            &artifact,
            &input,
            reference_tokens.as_deref(),
            warmup_iterations,
            measured_trials,
            device,
        )?,
        Command::TranscribeBatch {
            artifact,
            input,
            warmup_iterations,
            measured_trials,
            verify_single,
            device,
        } => {
            transcribe_batch(
                &artifact,
                &input,
                warmup_iterations,
                measured_trials,
                verify_single,
                device,
            )?;
        }
        Command::Serve {
            artifact,
            host,
            port,
            device,
            max_audio_seconds,
            max_upload_bytes,
        } => parakeet_l4::server::serve(
            &artifact,
            &host,
            port,
            device,
            max_audio_seconds,
            max_upload_bytes,
        )?,
        Command::LoadAot { artifact, device } => load_aot(&artifact, device)?,
        Command::BenchAotLinear {
            artifact,
            tensor,
            rows,
            warmup_iterations,
            warm_iterations,
            device,
        } => bench_aot_linear(
            &artifact,
            &tensor,
            rows,
            warmup_iterations,
            warm_iterations,
            device,
        )?,
        Command::CudaInfo { device } => cuda_info(device)?,
        Command::BenchFp16Linear {
            device,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        } => bench_fp16_linear(device, warmup_iterations, warm_iterations, cold_iterations)?,
        Command::BenchQuantizedLinear {
            device,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        } => bench_quantized_linear(device, warmup_iterations, warm_iterations, cold_iterations)?,
        Command::BenchQ4KLinear {
            device,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        } => bench_q4_k_linear(device, warmup_iterations, warm_iterations, cold_iterations)?,
    }
    Ok(())
}

fn frontend_cpu(
    model_dir: &std::path::Path,
    input: &std::path::Path,
    profile: ModelProfile,
    output: Option<&std::path::Path>,
) -> Result<(), Box<dyn Error>> {
    let config = ModelBundleConfig::load(model_dir, profile)?;
    let weights = SafeTensorIndex::open(&model_dir.join("model.safetensors"))?;
    weights.validate_model(profile)?;
    let window = weights.read_f32("frontend.window")?;
    let mel_filters = weights.read_f32("frontend.mel_filters")?;
    let frontend = parakeet_l4::frontend::CpuMelFrontend::new(
        &config.processor.feature_extractor,
        window,
        mel_filters,
    )?;
    let audio = parakeet_l4::audio::read_pcm16_mono(input)?;
    if audio.sample_rate != config.processor.feature_extractor.sampling_rate as u32 {
        return Err(format!(
            "input sample rate is {}, expected {}",
            audio.sample_rate, config.processor.feature_extractor.sampling_rate
        )
        .into());
    }

    let started = Instant::now();
    let features = frontend.compute(&audio.samples)?;
    let processing_seconds = started.elapsed().as_secs_f64();
    let mut encoded = Vec::with_capacity(features.values.len() * 4);
    let mut hasher = blake3::Hasher::new();
    for value in &features.values {
        let bytes = value.to_le_bytes();
        hasher.update(&bytes);
        if output.is_some() {
            encoded.extend_from_slice(&bytes);
        }
    }
    if let Some(path) = output {
        std::fs::write(path, encoded)?;
    }
    let count = features.values.len() as f64;
    let sum = features
        .values
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>();
    let squares = features
        .values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    let audio_seconds = audio.samples.len() as f64 / f64::from(audio.sample_rate);
    let report = FrontendReport {
        input: input.to_owned(),
        sample_rate: audio.sample_rate,
        samples: audio.samples.len(),
        audio_seconds,
        mel_bins: features.mel_bins,
        frames: features.frames,
        valid_frames: features.valid_frames,
        layout: "feature_major_f32[mel_bin,frame]",
        values: features.values.len(),
        blake3: hasher.finalize().to_hex().to_string(),
        minimum: features
            .values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        maximum: features
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        mean: sum / count,
        rms: (squares / count).sqrt(),
        processing_seconds,
        realtime_factor: audio_seconds / processing_seconds,
        output: output.map(std::path::Path::to_owned),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_frontend(
    artifact_path: &std::path::Path,
    input: &std::path::Path,
    warmup_iterations: usize,
    warm_iterations: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let artifact = AotArtifactIndex::open(artifact_path)?;
    if artifact.header.profile != ModelProfile::V2English {
        return Err("GPU frontend benchmark currently requires the v2 English artifact".into());
    }
    let config = parakeet_l4::config::FeatureExtractorConfig::v2_english();
    let window = artifact.read_f32("frontend.window")?;
    let mel_filters = artifact.read_f32("frontend.mel_filters")?;
    let frontend =
        parakeet_l4::frontend::CpuMelFrontend::new(&config, window.clone(), mel_filters.clone())?;
    let audio = parakeet_l4::audio::read_pcm16_mono(input)?;
    if audio.sample_rate != config.sampling_rate as u32 {
        return Err(format!(
            "input sample rate is {}, expected {}",
            audio.sample_rate, config.sampling_rate
        )
        .into());
    }
    let reference = frontend.compute(&audio.samples)?;
    let report = parakeet_l4::cuda::benchmark_frontend(
        device,
        &config,
        &audio.samples,
        &window,
        &mel_filters,
        &reference,
        warmup_iterations,
        warm_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_frontend(
    _artifact: &std::path::Path,
    _input: &std::path::Path,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_subsampling(
    artifact_path: &std::path::Path,
    input: &std::path::Path,
    reference_path: Option<&std::path::Path>,
    warmup_iterations: usize,
    warm_iterations: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let artifact = AotArtifactIndex::open(artifact_path)?;
    if artifact.header.profile != ModelProfile::V2English {
        return Err("subsampling benchmark currently requires the v2 English artifact".into());
    }
    let config = parakeet_l4::config::FeatureExtractorConfig::v2_english();
    let frontend = parakeet_l4::frontend::CpuMelFrontend::new(
        &config,
        artifact.read_f32("frontend.window")?,
        artifact.read_f32("frontend.mel_filters")?,
    )?;
    let audio = parakeet_l4::audio::read_pcm16_mono(input)?;
    if audio.sample_rate != config.sampling_rate as u32 {
        return Err(format!(
            "input sample rate is {}, expected {}",
            audio.sample_rate, config.sampling_rate
        )
        .into());
    }
    let features = frontend.compute(&audio.samples)?;
    let reference_bytes = reference_path.map(std::fs::read).transpose()?;
    let reference = reference_bytes
        .as_deref()
        .map(|bytes| {
            if !bytes.len().is_multiple_of(4) {
                return Err("subsampling reference length is not divisible by four");
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>())
        })
        .transpose()?;
    let report = parakeet_l4::cuda::benchmark_subsampling(
        device,
        artifact_path,
        &features.values,
        features.frames,
        features.valid_frames,
        reference.as_deref(),
        warmup_iterations,
        warm_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_subsampling(
    _artifact: &std::path::Path,
    _input: &std::path::Path,
    _reference: Option<&std::path::Path>,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn bench_encoder_layer(
    artifact: &std::path::Path,
    input: &std::path::Path,
    rows: usize,
    valid_rows: usize,
    layer: usize,
    layers: usize,
    reference: Option<&std::path::Path>,
    warmup_iterations: usize,
    warm_iterations: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let input = read_raw_f32(input)?;
    let reference = reference.map(read_raw_f32).transpose()?;
    let report = parakeet_l4::cuda::benchmark_encoder_layer(
        device,
        artifact,
        layer,
        layers,
        &input,
        rows,
        valid_rows,
        reference.as_deref(),
        warmup_iterations,
        warm_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn read_raw_f32(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{}: raw F32 length is not divisible by four",
            path.display()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn bench_decoder(
    artifact: &std::path::Path,
    input: &std::path::Path,
    frames: usize,
    reference: Option<&std::path::Path>,
    vocabulary: Option<&std::path::Path>,
    warmup_iterations: usize,
    warm_iterations: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let input = read_raw_f32(input)?;
    let reference = reference.map(read_raw_i32).transpose()?;
    let vocabulary = vocabulary
        .map(|path| read_vocabulary(path, ModelProfile::V2English))
        .transpose()?;
    let report = parakeet_l4::cuda::benchmark_decoder(
        device,
        artifact,
        &input,
        frames,
        reference.as_deref(),
        vocabulary.as_deref(),
        warmup_iterations,
        warm_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn read_raw_i32(path: &std::path::Path) -> Result<Vec<i32>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{}: raw I32 length is not divisible by four",
            path.display()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn read_vocabulary(
    path: &std::path::Path,
    profile: ModelProfile,
) -> Result<Vec<String>, Box<dyn Error>> {
    let contents = std::fs::read_to_string(path)?;
    let pieces = contents
        .lines()
        .map(|line| {
            line.split_once('\t')
                .map_or(line, |(piece, _)| piece)
                .to_owned()
        })
        .collect::<Vec<_>>();
    let expected = profile.blank_token_id();
    if pieces.len() != expected {
        return Err(format!(
            "{}: expected {expected} SentencePiece vocabulary entries, got {}",
            path.display(),
            pieces.len()
        )
        .into());
    }
    Ok(pieces)
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn transcribe(
    artifact_path: &std::path::Path,
    input: &std::path::Path,
    reference_tokens: Option<&std::path::Path>,
    warmup_iterations: usize,
    measured_trials: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let artifact = AotArtifactIndex::open(artifact_path)?;
    if artifact.header.profile != ModelProfile::V2English {
        return Err("the complete pipeline currently requires the v2 English artifact".into());
    }
    let config = parakeet_l4::config::FeatureExtractorConfig::v2_english();
    let audio = parakeet_l4::audio::read_pcm16_mono(input)?;
    if audio.sample_rate != config.sampling_rate as u32 {
        return Err(format!(
            "input sample rate is {}, expected {}",
            audio.sample_rate, config.sampling_rate
        )
        .into());
    }
    let reference_tokens = reference_tokens.map(read_raw_i32).transpose()?;
    let report = parakeet_l4::cuda::benchmark_pipeline(
        device,
        artifact_path,
        &config,
        &audio.samples,
        &artifact.read_f32("frontend.window")?,
        &artifact.read_f32("frontend.mel_filters")?,
        &artifact.header.vocabulary,
        reference_tokens.as_deref(),
        warmup_iterations,
        measured_trials,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn transcribe_batch(
    artifact: &std::path::Path,
    paths: &[PathBuf],
    warmups: usize,
    trials: usize,
    verify: bool,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    if trials == 0 {
        return Err("measured trials must be nonzero".into());
    }
    let audio = paths
        .iter()
        .map(|path| parakeet_l4::audio::read_pcm16_mono(path))
        .collect::<Result<Vec<_>, _>>()?;
    if audio.iter().any(|item| item.sample_rate != 16_000) {
        return Err("batch requires 16 kHz mono PCM16 WAV".into());
    }
    let inputs = audio
        .iter()
        .map(|item| item.samples.as_slice())
        .collect::<Vec<_>>();
    if inputs.len() != 1 {
        parakeet_l4::transcription::BatchLayout::new(
            &inputs.iter().map(|x| x.len()).collect::<Vec<_>>(),
        )?;
    }
    let mut engine = parakeet_l4::cuda::PipelineEngine::load(
        device,
        artifact,
        inputs.iter().map(|x| x.len()).max().unwrap(),
    )?;
    let singles = if verify {
        inputs
            .iter()
            .map(|samples| engine.transcribe(samples))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for _ in 0..warmups {
        engine.transcribe_batch(&inputs)?;
    }
    let mut reports = Vec::with_capacity(trials);
    for _ in 0..trials {
        let report = engine.transcribe_batch(&inputs)?;
        for (index, (batch, single)) in report.results.iter().zip(&singles).enumerate() {
            if batch.token_ids != single.token_ids {
                return Err(format!("batch token parity failed for input {index}").into());
            }
        }
        reports.push(report);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"single_token_parity": verify.then_some(true), "trials": reports})
        )?
    );
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn transcribe_batch(
    _artifact: &std::path::Path,
    _paths: &[PathBuf],
    _warmups: usize,
    _trials: usize,
    _verify: bool,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err(
        "CUDA support requires x86_64 Linux and `cargo build --locked --release --features cuda`"
            .into(),
    )
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[allow(clippy::too_many_arguments)]
fn bench_encoder_layer(
    _artifact: &std::path::Path,
    _input: &std::path::Path,
    _rows: usize,
    _valid_rows: usize,
    _layer: usize,
    _layers: usize,
    _reference: Option<&std::path::Path>,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[allow(clippy::too_many_arguments)]
fn bench_decoder(
    _artifact: &std::path::Path,
    _input: &std::path::Path,
    _frames: usize,
    _reference: Option<&std::path::Path>,
    _vocabulary: Option<&std::path::Path>,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[allow(clippy::too_many_arguments)]
fn transcribe(
    _artifact: &std::path::Path,
    _input: &std::path::Path,
    _reference_tokens: Option<&std::path::Path>,
    _warmup_iterations: usize,
    _measured_trials: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn load_aot(artifact: &std::path::Path, device: usize) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::load_aot_weights(device, artifact)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn load_aot(_artifact: &std::path::Path, _device: usize) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_aot_linear(
    artifact: &std::path::Path,
    tensor: &str,
    rows: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    device: usize,
) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::benchmark_aot_linear(
        device,
        artifact,
        tensor,
        rows,
        warmup_iterations,
        warm_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_aot_linear(
    _artifact: &std::path::Path,
    _tensor: &str,
    _rows: usize,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _device: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_info(device: usize) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::inspect_sm89(device)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn cuda_info(_device: usize) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_fp16_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::benchmark_fp16_linear(
        device,
        warmup_iterations,
        warm_iterations,
        cold_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_fp16_linear(
    _device: usize,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_quantized_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::benchmark_quantized_linear(
        device,
        warmup_iterations,
        warm_iterations,
        cold_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_quantized_linear(
    _device: usize,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn bench_q4_k_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    let report = parakeet_l4::cuda::benchmark_q4_k_linear(
        device,
        warmup_iterations,
        warm_iterations,
        cold_iterations,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn bench_q4_k_linear(
    _device: usize,
    _warmup_iterations: usize,
    _warm_iterations: usize,
    _cold_iterations: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}
