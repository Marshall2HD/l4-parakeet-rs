use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

mod cublas;
mod decoder;
mod encoder;
mod frontend;
mod pipeline;
mod q4k_benchmark;
mod quant_benchmark;
mod subsampling;

pub use decoder::{DecoderBenchmarkReport, benchmark_decoder};
pub use encoder::{EncoderLayerBenchmarkReport, benchmark_encoder_layer};
pub use frontend::{GpuFrontendReport, benchmark_frontend};
pub use pipeline::{PipelineBenchmarkReport, PipelineEngine, benchmark_pipeline};
pub use q4k_benchmark::{Q4KBenchmarkReport, benchmark_q4_k_linear};
pub use quant_benchmark::{QuantizedBenchmarkReport, benchmark_quantized_linear};
pub use subsampling::{SubsamplingBenchmarkReport, benchmark_subsampling};

const SM89_CUBIN: &[u8] = include_bytes!(env!("PARAKEET_SM89_CUBIN"));
const M_BUCKETS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const L2_SCRUB_BYTES: usize = 96 * 1024 * 1024;
const WARM_SAMPLES: usize = 7;

#[derive(Clone, Copy)]
struct LinearShape {
    family: &'static str,
    k: usize,
    n: usize,
}

const LINEAR_SHAPES: [LinearShape; 7] = [
    LinearShape {
        family: "encoder_ffn_expand",
        k: 1024,
        n: 4096,
    },
    LinearShape {
        family: "encoder_ffn_contract",
        k: 4096,
        n: 1024,
    },
    LinearShape {
        family: "attention_projection",
        k: 1024,
        n: 1024,
    },
    LinearShape {
        family: "pointwise_conv_expand",
        k: 1024,
        n: 2048,
    },
    LinearShape {
        family: "predictor_lstm",
        k: 640,
        n: 2560,
    },
    LinearShape {
        family: "joint_output_v2",
        k: 640,
        n: 1030,
    },
    LinearShape {
        family: "joint_output",
        k: 640,
        n: 8198,
    },
];

#[derive(Debug, Serialize)]
pub struct CudaReport {
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub kernel_binary_version: String,
    pub kernel_registers_per_thread: i32,
    pub kernel_static_shared_bytes: i32,
    pub kernel_max_threads_per_block: i32,
}

#[derive(Debug, Serialize)]
pub struct Fp16BenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub device_memory_bytes: usize,
    pub kernel: &'static str,
    pub kernel_binary_version: String,
    pub kernel_registers_per_thread: i32,
    pub kernel_static_shared_bytes: i32,
    pub operation: &'static str,
    pub warmup_iterations: usize,
    pub warm_samples: usize,
    pub warm_iterations: usize,
    pub cold_iterations: usize,
    pub l2_scrub_bytes: usize,
    pub measurements: Vec<Fp16Measurement>,
}

#[derive(Debug, Serialize)]
pub struct Fp16Measurement {
    pub family: &'static str,
    pub logical_m: usize,
    pub logical_n: usize,
    pub logical_k: usize,
    pub padded_m: usize,
    pub padded_n: usize,
    pub padded_k: usize,
    pub weight_bytes: usize,
    pub warm_latency_ms: f64,
    pub cold_p50_latency_ms: f64,
    pub cold_p95_latency_ms: f64,
    pub warm_effective_weight_gbps: f64,
    pub cold_p50_effective_weight_gbps: f64,
    pub warm_physical_tflops: f64,
    pub cold_p50_physical_tflops: f64,
    pub correctness_samples: usize,
    pub max_abs_error: f32,
}

#[derive(Debug, Serialize)]
pub struct AotLoadReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub artifact: String,
    pub profile: crate::config::ModelProfile,
    pub precision_plan: String,
    pub tensors: usize,
    pub payload_bytes: u64,
    pub device_memory_bytes: usize,
    pub wall_seconds: f64,
    pub effective_gib_per_second: f64,
}

#[derive(Debug, Serialize)]
pub struct AotLinearReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub artifact: String,
    pub tensor: String,
    pub bias: Option<String>,
    pub logical_m: usize,
    pub logical_n: usize,
    pub logical_k: usize,
    pub padded_m: usize,
    pub padded_n: usize,
    pub padded_k: usize,
    pub warmup_iterations: usize,
    pub warm_iterations: usize,
    pub warm_latency_ms: f64,
    pub correctness_samples: usize,
    pub max_abs_error: f32,
}

pub fn load_aot_weights(device: usize, path: &Path) -> Result<AotLoadReport, Box<dyn Error>> {
    let uploaded = upload_aot(device, path)?;
    let (major, minor) = uploaded.context.compute_capability()?;

    Ok(AotLoadReport {
        schema_version: 1,
        device,
        name: uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        artifact: path.display().to_string(),
        profile: uploaded.artifact.header.profile,
        precision_plan: uploaded.artifact.header.precision_plan.clone(),
        tensors: uploaded.artifact.header.tensors.len(),
        payload_bytes: uploaded.artifact.header.payload_bytes,
        device_memory_bytes: uploaded.context.total_mem()?,
        wall_seconds: uploaded.wall_seconds,
        effective_gib_per_second: uploaded.artifact.header.payload_bytes as f64
            / (uploaded.wall_seconds * 1024.0 * 1024.0 * 1024.0),
    })
}

pub fn benchmark_aot_linear(
    device: usize,
    path: &Path,
    tensor_name: &str,
    logical_m: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
) -> Result<AotLinearReport, Box<dyn Error>> {
    if logical_m == 0 || warmup_iterations == 0 || warm_iterations == 0 {
        return Err("rows and benchmark iteration counts must be greater than zero".into());
    }

    let uploaded = upload_aot(device, path)?;
    let tensor = uploaded
        .artifact
        .header
        .tensors
        .iter()
        .find(|tensor| tensor.name == tensor_name)
        .ok_or_else(|| format!("artifact does not contain tensor {tensor_name:?}"))?;
    if tensor.storage != crate::artifact::AotStorage::Sm89Fp16Linear
        || tensor.logical_shape.len() != 2
        || tensor.physical_shape.len() != 2
    {
        return Err(format!("{tensor_name}: tensor is not an sm_89 FP16 linear weight").into());
    }

    let logical_n = tensor.logical_shape[0];
    let logical_k = tensor.logical_shape[1];
    let padded_m = logical_m.next_multiple_of(16);
    let padded_n = tensor.physical_shape[0];
    let padded_k = tensor.physical_shape[1];
    let bias_name = tensor_name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.bias"));
    let bias_tensor = bias_name.as_ref().and_then(|name| {
        uploaded
            .artifact
            .header
            .tensors
            .iter()
            .find(|tensor| tensor.name == *name)
    });
    if let Some(bias) = bias_tensor
        && (bias.storage != crate::artifact::AotStorage::Sm89Fp32Bias
            || bias.logical_shape != [logical_n]
            || bias.physical_shape != [padded_n])
    {
        return Err(format!("{}: incompatible AOT linear bias layout", bias.name).into());
    }
    let input_host = make_input(logical_m, logical_k, padded_m, padded_k);
    let input = uploaded.stream.clone_htod(&input_host)?;
    let zero_bias = if bias_tensor.is_none() {
        Some(uploaded.stream.alloc_zeros::<f32>(padded_n)?)
    } else {
        None
    };
    let mut output = uploaded.stream.alloc_zeros::<f16>(padded_m * padded_n)?;

    let start = usize::try_from(tensor.offset)?;
    let end = start
        .checked_add(usize::try_from(tensor.bytes)?)
        .ok_or("tensor range overflow")?;
    let weight_bytes = uploaded.weights.slice(start..end);
    // SAFETY: the artifact validator requires this storage class to be 16-bit FP16,
    // and every tensor offset is at least 256-byte aligned.
    let weights = unsafe { weight_bytes.transmute::<f16>(usize::try_from(tensor.bytes / 2)?) }
        .ok_or("FP16 tensor view exceeds its artifact range")?;
    let bias = if let Some(record) = bias_tensor {
        let start = usize::try_from(record.offset)?;
        let end = start
            .checked_add(usize::try_from(record.bytes)?)
            .ok_or("bias range overflow")?;
        let bytes = uploaded.weights.slice(start..end);
        // SAFETY: the artifact validator requires this storage class to be
        // 32-bit FP32 and every tensor offset is at least 256-byte aligned.
        unsafe { bytes.transmute::<f32>(usize::try_from(record.bytes / 4)?) }
            .ok_or("FP32 bias view exceeds its artifact range")?
    } else {
        zero_bias.as_ref().expect("zero bias allocation").as_view()
    };

    let module = uploaded
        .context
        .load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
    let function = module.load_function("pk_sm89_fp16_linear_epilogue")?;
    let config = linear_launch_config(padded_m, padded_n);

    for _ in 0..warmup_iterations {
        launch_linear_view(
            &uploaded.stream,
            &function,
            config,
            &input,
            &weights,
            Some(&bias),
            &mut output.as_view_mut(),
            padded_m,
            padded_n,
            padded_k,
            None,
            1.0,
            0.0,
            0,
        )?;
    }
    uploaded.stream.synchronize()?;

    let started = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..warm_iterations {
        launch_linear_view(
            &uploaded.stream,
            &function,
            config,
            &input,
            &weights,
            Some(&bias),
            &mut output.as_view_mut(),
            padded_m,
            padded_n,
            padded_k,
            None,
            1.0,
            0.0,
            0,
        )?;
    }
    let ended = uploaded
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let warm_latency_ms = f64::from(started.elapsed_ms(&ended)?) / warm_iterations as f64;

    let output_host = uploaded.stream.clone_dtoh(&output)?;
    let packed_weights = read_artifact_fp16_tensor(path, &uploaded.artifact, tensor)?;
    let bias_host = match bias_tensor {
        Some(tensor) => read_artifact_fp32_tensor(path, &uploaded.artifact, tensor)?,
        None => vec![0.0_f32; padded_n],
    };
    let (correctness_samples, max_abs_error) = check_samples(
        &input_host,
        &packed_weights,
        &bias_host,
        &output_host,
        logical_m,
        logical_n,
        padded_n,
        padded_k,
    )?;

    let (major, minor) = uploaded.context.compute_capability()?;
    Ok(AotLinearReport {
        schema_version: 1,
        device,
        name: uploaded.context.name()?,
        compute_capability: format!("{major}.{minor}"),
        artifact: path.display().to_string(),
        tensor: tensor_name.into(),
        bias: bias_tensor.map(|tensor| tensor.name.clone()),
        logical_m,
        logical_n,
        logical_k,
        padded_m,
        padded_n,
        padded_k,
        warmup_iterations,
        warm_iterations,
        warm_latency_ms,
        correctness_samples,
        max_abs_error,
    })
}

pub(super) struct UploadedAot {
    artifact: crate::artifact::AotArtifactIndex,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    weights: CudaSlice<u8>,
    wall_seconds: f64,
}

pub(super) fn upload_aot(device: usize, path: &Path) -> Result<UploadedAot, Box<dyn Error>> {
    let artifact = crate::artifact::AotArtifactIndex::open(path)?;
    let context = CudaContext::new(device)?;
    let (major, minor) = context.compute_capability()?;
    if (major, minor) != (8, 9) {
        return Err(format!(
            "device {device} is compute capability {major}.{minor}; parakeet-l4 requires an L4-class sm_89 GPU"
        )
        .into());
    }

    let payload_bytes = usize::try_from(artifact.header.payload_bytes)?;
    let stream = context.default_stream();
    // SAFETY: every byte is initialized by the complete artifact read below
    // before the allocation can be observed by a kernel.
    let mut weights = unsafe { stream.alloc::<u8>(payload_bytes) }?;
    let mut source = File::open(path)?;
    source.seek(SeekFrom::Start(artifact.data_start))?;
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    let started = Instant::now();
    let mut offset = 0usize;
    while offset < payload_bytes {
        let requested = buffer.len().min(payload_bytes - offset);
        source.read_exact(&mut buffer[..requested])?;
        let mut destination = weights.slice_mut(offset..offset + requested);
        stream.memcpy_htod(&buffer[..requested], &mut destination)?;
        offset += requested;
    }
    stream.synchronize()?;

    Ok(UploadedAot {
        artifact,
        context,
        stream,
        weights,
        wall_seconds: started.elapsed().as_secs_f64(),
    })
}

fn read_artifact_fp16_tensor(
    path: &Path,
    artifact: &crate::artifact::AotArtifactIndex,
    tensor: &crate::artifact::AotTensor,
) -> Result<Vec<f16>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(artifact.data_start + tensor.offset))?;
    let mut encoded = vec![0u8; usize::try_from(tensor.bytes)?];
    file.read_exact(&mut encoded)?;
    Ok(encoded
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

fn read_artifact_fp32_tensor(
    path: &Path,
    artifact: &crate::artifact::AotArtifactIndex,
    tensor: &crate::artifact::AotTensor,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(artifact.data_start + tensor.offset))?;
    let mut encoded = vec![0u8; usize::try_from(tensor.bytes)?];
    file.read_exact(&mut encoded)?;
    Ok(encoded
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

pub fn inspect_sm89(device: usize) -> Result<CudaReport, Box<dyn Error>> {
    let context = CudaContext::new(device)?;
    let (major, minor) = context.compute_capability()?;
    if (major, minor) != (8, 9) {
        return Err(format!(
            "device {device} is compute capability {major}.{minor}; parakeet-l4 requires an L4-class sm_89 GPU"
        )
        .into());
    }

    let module = context.load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
    let function = module.load_function("pk_sm89_fp16_linear_epilogue")?;
    let binary = function.binary_version()?;

    Ok(CudaReport {
        device,
        name: context.name()?,
        compute_capability: format!("{major}.{minor}"),
        kernel_binary_version: format!("{}.{}", binary / 10, binary % 10),
        kernel_registers_per_thread: function.num_regs()?,
        kernel_static_shared_bytes: function.shared_size_bytes()?,
        kernel_max_threads_per_block: function.max_threads_per_block()?,
    })
}

pub fn benchmark_fp16_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Fp16BenchmarkReport, Box<dyn Error>> {
    if warmup_iterations == 0 || warm_iterations == 0 || cold_iterations == 0 {
        return Err("benchmark iteration counts must all be greater than zero".into());
    }

    let context = CudaContext::new(device)?;
    let (major, minor) = context.compute_capability()?;
    if (major, minor) != (8, 9) {
        return Err(format!(
            "device {device} is compute capability {major}.{minor}; parakeet-l4 requires an L4-class sm_89 GPU"
        )
        .into());
    }

    let module = context.load_module(Ptx::from_binary(SM89_CUBIN.to_vec()))?;
    let linear = module.load_function("pk_sm89_fp16_linear_epilogue")?;
    let scrub = module.load_function("pk_sm89_l2_scrub")?;
    let stream = context.default_stream();
    let scrub_elements = L2_SCRUB_BYTES / std::mem::size_of::<u32>();
    let mut scrub_buffer = stream.alloc_zeros::<u32>(scrub_elements)?;
    let mut measurements = Vec::with_capacity(LINEAR_SHAPES.len() * M_BUCKETS.len());

    for shape in LINEAR_SHAPES {
        measurements.extend(benchmark_shape(
            &stream,
            &linear,
            &scrub,
            &mut scrub_buffer,
            shape,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        )?);
    }

    let binary = linear.binary_version()?;
    Ok(Fp16BenchmarkReport {
        schema_version: 1,
        device,
        name: context.name()?,
        compute_capability: format!("{major}.{minor}"),
        device_memory_bytes: context.total_mem()?,
        kernel: "pk_sm89_fp16_linear_epilogue",
        kernel_binary_version: format!("{}.{}", binary / 10, binary % 10),
        kernel_registers_per_thread: linear.num_regs()?,
        kernel_static_shared_bytes: linear.shared_size_bytes()?,
        operation: "FP16 input and weights, FP32 accumulation, fused FP16 bias, FP16 output",
        warmup_iterations,
        warm_samples: WARM_SAMPLES,
        warm_iterations,
        cold_iterations,
        l2_scrub_bytes: L2_SCRUB_BYTES,
        measurements,
    })
}

#[allow(clippy::too_many_arguments)]
fn benchmark_shape(
    stream: &Arc<CudaStream>,
    linear: &CudaFunction,
    scrub: &CudaFunction,
    scrub_buffer: &mut CudaSlice<u32>,
    shape: LinearShape,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Vec<Fp16Measurement>, Box<dyn Error>> {
    let padded_k = round_up(shape.k, 128);
    let padded_n = round_up(shape.n, 128);
    let weights_host = make_weights(shape.n, shape.k, padded_n, padded_k);
    let bias_host = make_bias(shape.n, padded_n);
    let weights = stream.clone_htod(&weights_host)?;
    let bias = stream.clone_htod(&bias_host)?;
    let weight_bytes = weights_host.len() * std::mem::size_of::<f16>();
    let mut measurements = Vec::with_capacity(M_BUCKETS.len());

    for logical_m in M_BUCKETS {
        let padded_m = round_up(logical_m, 16);
        let input_host = make_input(logical_m, shape.k, padded_m, padded_k);
        let input = stream.clone_htod(&input_host)?;
        let mut output = stream.alloc_zeros::<f16>(padded_m * padded_n)?;
        let launch = linear_launch_config(padded_m, padded_n);

        stream.synchronize()?;
        for _ in 0..warmup_iterations {
            launch_linear(
                stream,
                linear,
                launch,
                &input,
                &weights,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
            )?;
        }
        stream.synchronize()?;

        let mut warm_latencies = Vec::with_capacity(WARM_SAMPLES);
        for _ in 0..WARM_SAMPLES {
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..warm_iterations {
                launch_linear(
                    stream,
                    linear,
                    launch,
                    &input,
                    &weights,
                    &bias,
                    &mut output,
                    padded_m,
                    padded_n,
                    padded_k,
                )?;
            }
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            warm_latencies.push(f64::from(start.elapsed_ms(&end)?) / warm_iterations as f64);
        }
        warm_latencies.sort_by(f64::total_cmp);
        let warm_latency_ms = percentile(&warm_latencies, 50);

        let mut cold_latencies = Vec::with_capacity(cold_iterations);
        for _ in 0..cold_iterations {
            launch_l2_scrub(stream, scrub, scrub_buffer)?;
            let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            launch_linear(
                stream,
                linear,
                launch,
                &input,
                &weights,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
            )?;
            let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            cold_latencies.push(f64::from(start.elapsed_ms(&end)?));
        }
        cold_latencies.sort_by(f64::total_cmp);
        let cold_p50_latency_ms = percentile(&cold_latencies, 50);
        let cold_p95_latency_ms = percentile(&cold_latencies, 95);

        let output_host = stream.clone_dtoh(&output)?;
        let (correctness_samples, max_abs_error) = check_samples(
            &input_host,
            &weights_host,
            &bias_host,
            &output_host,
            logical_m,
            shape.n,
            padded_n,
            padded_k,
        )?;
        let physical_flops = 2.0 * padded_m as f64 * padded_n as f64 * padded_k as f64;

        measurements.push(Fp16Measurement {
            family: shape.family,
            logical_m,
            logical_n: shape.n,
            logical_k: shape.k,
            padded_m,
            padded_n,
            padded_k,
            weight_bytes,
            warm_latency_ms,
            cold_p50_latency_ms,
            cold_p95_latency_ms,
            warm_effective_weight_gbps: effective_weight_gbps(weight_bytes, warm_latency_ms),
            cold_p50_effective_weight_gbps: effective_weight_gbps(
                weight_bytes,
                cold_p50_latency_ms,
            ),
            warm_physical_tflops: physical_flops / (warm_latency_ms * 1.0e9),
            cold_p50_physical_tflops: physical_flops / (cold_p50_latency_ms * 1.0e9),
            correctness_samples,
            max_abs_error,
        });
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn launch_linear(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<f16>,
    weights: &CudaSlice<f16>,
    bias: &CudaSlice<f32>,
    output: &mut CudaSlice<f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let null_residual = 0_u64;
    let output_scale = 1.0_f32;
    let residual_scale = 0.0_f32;
    let activation = 0_i32;
    let m = i32::try_from(m)?;
    let n = i32::try_from(n)?;
    let k = i32::try_from(k)?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weights)
        .arg(bias)
        .arg(&null_residual)
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

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_linear_view(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<f16>,
    weights: &CudaView<'_, f16>,
    bias: Option<&CudaView<'_, f32>>,
    output: &mut CudaViewMut<'_, f16>,
    m: usize,
    n: usize,
    k: usize,
    residual: Option<&CudaSlice<f16>>,
    output_scale: f32,
    residual_scale: f32,
    activation: i32,
) -> Result<(), Box<dyn Error>> {
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

fn launch_l2_scrub(
    stream: &CudaStream,
    function: &CudaFunction,
    buffer: &mut CudaSlice<u32>,
) -> Result<(), Box<dyn Error>> {
    let elements = i32::try_from(buffer.len())?;
    let threads = 256_u32;
    let config = LaunchConfig {
        grid_dim: ((u32::try_from(buffer.len())?).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(function);
    builder.arg(buffer).arg(&elements);
    unsafe { builder.launch(config) }?;
    Ok(())
}

pub(super) fn linear_launch_config(m: usize, n: usize) -> LaunchConfig {
    debug_assert_eq!(m % 16, 0);
    debug_assert_eq!(n % 32, 0);
    LaunchConfig {
        grid_dim: ((n / 32) as u32, (m.div_ceil(32)) as u32, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn linear_large_launch_config(m: usize, n: usize) -> LaunchConfig {
    debug_assert_eq!(m % 16, 0);
    debug_assert_eq!(n % 128, 0);
    LaunchConfig {
        grid_dim: ((n / 128) as u32, (m.div_ceil(64)) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn make_input(logical_m: usize, logical_k: usize, padded_m: usize, padded_k: usize) -> Vec<f16> {
    let mut input = vec![f16::ZERO; padded_m * padded_k];
    for row in 0..logical_m {
        for col in 0..logical_k {
            input[row * padded_k + col] = f16::from_f32(pattern(row * logical_k + col, 17, 0.125));
        }
    }
    input
}

fn make_weights(logical_n: usize, logical_k: usize, padded_n: usize, padded_k: usize) -> Vec<f16> {
    let mut weights = vec![f16::ZERO; padded_n * padded_k];
    for output in 0..logical_n {
        for input in 0..logical_k {
            weights[output * padded_k + input] =
                f16::from_f32(pattern(output * logical_k + input, 29, 0.03125));
        }
    }
    weights
}

fn make_bias(logical_n: usize, padded_n: usize) -> Vec<f32> {
    let mut bias = vec![0.0_f32; padded_n];
    for (index, value) in bias.iter_mut().take(logical_n).enumerate() {
        *value = pattern(index, 43, 0.0625);
    }
    bias
}

fn pattern(index: usize, salt: u64, amplitude: f32) -> f32 {
    let mut value = (index as u64).wrapping_add(salt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = (value & 0xffff) as f32 / 65_535.0;
    (unit * 2.0 - 1.0) * amplitude
}

#[allow(clippy::too_many_arguments)]
fn check_samples(
    input: &[f16],
    weights: &[f16],
    bias: &[f32],
    output: &[f16],
    logical_m: usize,
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
) -> Result<(usize, f32), Box<dyn Error>> {
    let rows = sample_indices(logical_m);
    let columns = sample_indices(logical_n);
    let mut samples = 0;
    let mut max_abs_error = 0.0_f32;

    for row in rows {
        for &column in &columns {
            let input_row = &input[row * padded_k..(row + 1) * padded_k];
            let weight_row = &weights[column * padded_k..(column + 1) * padded_k];
            let expected = input_row
                .iter()
                .zip(weight_row)
                .fold(bias[column], |sum, (left, right)| {
                    sum + left.to_f32() * right.to_f32()
                });
            let expected = f16::from_f32(expected).to_f32();
            let actual = output[row * padded_n + column].to_f32();
            let error = (actual - expected).abs();
            if !actual.is_finite() || error > 0.02 {
                return Err(format!(
                    "FP16 parity failed at [{row},{column}]: actual={actual}, expected={expected}, abs_error={error}"
                )
                .into());
            }
            max_abs_error = max_abs_error.max(error);
            samples += 1;
        }
    }

    Ok((samples, max_abs_error))
}

fn sample_indices(length: usize) -> Vec<usize> {
    let mut indices = vec![0, length / 3, (2 * length) / 3, length - 1];
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn effective_weight_gbps(weight_bytes: usize, latency_ms: f64) -> f64 {
    weight_bytes as f64 / (latency_ms * 1.0e6)
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}
