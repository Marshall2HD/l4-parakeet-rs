use super::{
    L2_SCRUB_BYTES, LINEAR_SHAPES, LinearShape, M_BUCKETS, SM89_CUBIN, WARM_SAMPLES, make_bias,
    make_input, make_weights, percentile, round_up, sample_indices,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use float8::F8E4M3;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::sync::Arc;

const FP8_MAX: f32 = 448.0;
const INT8_MAX: f32 = 127.0;

#[derive(Debug, Serialize)]
pub struct QuantizedBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub device_memory_bytes: usize,
    pub input_scaling: &'static str,
    pub weight_scaling: &'static str,
    pub warmup_iterations: usize,
    pub warm_samples: usize,
    pub warm_iterations: usize,
    pub cold_iterations: usize,
    pub l2_scrub_bytes: usize,
    pub kernels: Vec<KernelReport>,
    pub measurements: Vec<QuantizedMeasurement>,
}

#[derive(Debug, Serialize)]
pub struct KernelReport {
    pub format: &'static str,
    pub kernel: &'static str,
    pub binary_version: String,
    pub registers_per_thread: i32,
    pub static_shared_bytes: i32,
}

#[derive(Debug, Serialize)]
pub struct QuantizedMeasurement {
    pub format: &'static str,
    pub family: &'static str,
    pub logical_m: usize,
    pub logical_n: usize,
    pub logical_k: usize,
    pub padded_m: usize,
    pub padded_n: usize,
    pub padded_k: usize,
    pub packed_weight_bytes: usize,
    pub warm_packed_latency_ms: f64,
    pub warm_quantize_latency_ms: f64,
    pub warm_end_to_end_latency_ms: f64,
    pub cold_p50_packed_latency_ms: f64,
    pub cold_p95_packed_latency_ms: f64,
    pub cold_p50_end_to_end_latency_ms: f64,
    pub cold_p95_end_to_end_latency_ms: f64,
    pub warm_packed_effective_weight_gbps: f64,
    pub cold_p50_packed_effective_weight_gbps: f64,
    pub warm_packed_physical_tflops: f64,
    pub correctness_samples: usize,
    pub max_abs_kernel_error: f32,
    pub max_abs_quantization_error: f32,
}

pub fn benchmark_quantized_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<QuantizedBenchmarkReport, Box<dyn Error>> {
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
    let fp8_linear = module.load_function("pk_sm89_fp8_linear_epilogue")?;
    let int8_linear = module.load_function("pk_sm89_int8_linear_epilogue")?;
    let fp8_quantize = module.load_function("pk_sm89_quantize_fp8")?;
    let int8_quantize = module.load_function("pk_sm89_quantize_int8")?;
    let scrub = module.load_function("pk_sm89_l2_scrub")?;
    let stream = context.default_stream();
    let scrub_elements = L2_SCRUB_BYTES / std::mem::size_of::<u32>();
    let mut scrub_buffer = stream.alloc_zeros::<u32>(scrub_elements)?;
    let mut measurements = Vec::with_capacity(LINEAR_SHAPES.len() * M_BUCKETS.len() * 2);

    for shape in LINEAR_SHAPES {
        let padded_k = round_up(shape.k, 128);
        let padded_n = round_up(shape.n, 128);
        let weights = make_weights(shape.n, shape.k, padded_n, padded_k);
        let bias = make_bias(shape.n, padded_n);
        measurements.extend(benchmark_fp8_shape(
            &stream,
            &fp8_linear,
            &fp8_quantize,
            &scrub,
            &mut scrub_buffer,
            shape,
            &weights,
            &bias,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        )?);
        measurements.extend(benchmark_int8_shape(
            &stream,
            &int8_linear,
            &int8_quantize,
            &scrub,
            &mut scrub_buffer,
            shape,
            &weights,
            &bias,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        )?);
    }

    Ok(QuantizedBenchmarkReport {
        schema_version: 1,
        device,
        name: context.name()?,
        compute_capability: format!("{major}.{minor}"),
        device_memory_bytes: context.total_mem()?,
        input_scaling: "fixed symmetric per tensor; supplied scale; calibration/reduction excluded",
        weight_scaling: "symmetric per output row; scales included in packed byte counts",
        warmup_iterations,
        warm_samples: WARM_SAMPLES,
        warm_iterations,
        cold_iterations,
        l2_scrub_bytes: L2_SCRUB_BYTES,
        kernels: vec![
            kernel_report("fp8_e4m3", "pk_sm89_fp8_linear_epilogue", &fp8_linear)?,
            kernel_report("int8", "pk_sm89_int8_linear_epilogue", &int8_linear)?,
        ],
        measurements,
    })
}

#[allow(clippy::too_many_arguments)]
fn benchmark_fp8_shape(
    stream: &Arc<CudaStream>,
    linear: &CudaFunction,
    quantize: &CudaFunction,
    scrub: &CudaFunction,
    scrub_buffer: &mut CudaSlice<u32>,
    shape: LinearShape,
    source_weights: &[f16],
    bias_host: &[f32],
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Vec<QuantizedMeasurement>, Box<dyn Error>> {
    let padded_k = round_up(shape.k, 128);
    let padded_n = round_up(shape.n, 128);
    let (weights_host, weight_scales_host) =
        quantize_fp8_weights(source_weights, shape.n, padded_n, padded_k);
    let weights = stream.clone_htod(&weights_host)?;
    let weight_scales = stream.clone_htod(&weight_scales_host)?;
    let bias = stream.clone_htod(bias_host)?;
    let packed_weight_bytes = weights_host.len() + weight_scales_host.len() * 4;
    let mut measurements = Vec::with_capacity(M_BUCKETS.len());

    for logical_m in M_BUCKETS {
        let padded_m = round_up(logical_m, 16);
        let source_input = make_input(logical_m, shape.k, padded_m, padded_k);
        let input_scale = 0.125 / FP8_MAX;
        let inverse_scale = input_scale.recip();
        let input_host: Vec<F8E4M3> = source_input
            .iter()
            .map(|value| F8E4M3::from_f32(value.to_f32() * inverse_scale))
            .collect();
        let source_input_device = stream.clone_htod(&source_input)?;
        let mut input = stream.clone_htod(&input_host)?;
        let mut output = stream.alloc_zeros::<f16>(padded_m * padded_n)?;
        let linear_config = quantized_linear_launch_config(padded_m, padded_n);
        let quantize_config = elementwise_launch_config(source_input.len());

        stream.synchronize()?;
        for _ in 0..warmup_iterations {
            launch_fp8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )?;
        }
        stream.synchronize()?;

        let warm_packed_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_fp8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;
        let warm_quantize_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_fp8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )
        })?;
        let warm_end_to_end_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_fp8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )?;
            launch_fp8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;

        let cold_packed = time_cold(cold_iterations, stream, scrub, scrub_buffer, || {
            launch_fp8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;
        let cold_end_to_end = time_cold(cold_iterations, stream, scrub, scrub_buffer, || {
            launch_fp8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )?;
            launch_fp8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;

        let actual_input = stream.clone_dtoh(&input)?;
        if actual_input != input_host {
            return Err("GPU FP8 activation quantization disagrees with the host packer".into());
        }
        let output_host = stream.clone_dtoh(&output)?;
        let (samples, kernel_error, quantization_error) = check_fp8_samples(
            &source_input,
            source_weights,
            &input_host,
            &weights_host,
            &weight_scales_host,
            bias_host,
            &output_host,
            logical_m,
            shape.n,
            padded_n,
            padded_k,
            input_scale,
        )?;
        measurements.push(measurement(
            "fp8_e4m3",
            shape,
            logical_m,
            padded_m,
            padded_n,
            padded_k,
            packed_weight_bytes,
            warm_packed_latency_ms,
            warm_quantize_latency_ms,
            warm_end_to_end_latency_ms,
            &cold_packed,
            &cold_end_to_end,
            samples,
            kernel_error,
            quantization_error,
        ));
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn benchmark_int8_shape(
    stream: &Arc<CudaStream>,
    linear: &CudaFunction,
    quantize: &CudaFunction,
    scrub: &CudaFunction,
    scrub_buffer: &mut CudaSlice<u32>,
    shape: LinearShape,
    source_weights: &[f16],
    bias_host: &[f32],
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Vec<QuantizedMeasurement>, Box<dyn Error>> {
    let padded_k = round_up(shape.k, 128);
    let padded_n = round_up(shape.n, 128);
    let (weights_host, weight_scales_host) =
        quantize_int8_weights(source_weights, shape.n, padded_n, padded_k);
    let weights = stream.clone_htod(&weights_host)?;
    let weight_scales = stream.clone_htod(&weight_scales_host)?;
    let bias = stream.clone_htod(bias_host)?;
    let packed_weight_bytes = weights_host.len() + weight_scales_host.len() * 4;
    let mut measurements = Vec::with_capacity(M_BUCKETS.len());

    for logical_m in M_BUCKETS {
        let padded_m = round_up(logical_m, 16);
        let source_input = make_input(logical_m, shape.k, padded_m, padded_k);
        let input_scale = 0.125 / INT8_MAX;
        let inverse_scale = input_scale.recip();
        let input_host: Vec<i8> = source_input
            .iter()
            .map(|value| quantize_i8(value.to_f32() * inverse_scale))
            .collect();
        let source_input_device = stream.clone_htod(&source_input)?;
        let mut input = stream.clone_htod(&input_host)?;
        let mut output = stream.alloc_zeros::<f16>(padded_m * padded_n)?;
        let linear_config = quantized_linear_launch_config(padded_m, padded_n);
        let quantize_config = elementwise_launch_config(source_input.len());

        stream.synchronize()?;
        for _ in 0..warmup_iterations {
            launch_int8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )?;
        }
        stream.synchronize()?;

        let warm_packed_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_int8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;
        let warm_quantize_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_int8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )
        })?;
        let warm_end_to_end_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_int8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )?;
            launch_int8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;

        let cold_packed = time_cold(cold_iterations, stream, scrub, scrub_buffer, || {
            launch_int8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;
        let cold_end_to_end = time_cold(cold_iterations, stream, scrub, scrub_buffer, || {
            launch_int8_quantize(
                stream,
                quantize,
                quantize_config,
                &source_input_device,
                &mut input,
                inverse_scale,
            )?;
            launch_int8_linear(
                stream,
                linear,
                linear_config,
                &input,
                &weights,
                &weight_scales,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                padded_k,
                input_scale,
            )
        })?;

        let actual_input = stream.clone_dtoh(&input)?;
        if actual_input != input_host {
            return Err("GPU INT8 activation quantization disagrees with the host packer".into());
        }
        let output_host = stream.clone_dtoh(&output)?;
        let (samples, kernel_error, quantization_error) = check_int8_samples(
            &source_input,
            source_weights,
            &input_host,
            &weights_host,
            &weight_scales_host,
            bias_host,
            &output_host,
            logical_m,
            shape.n,
            padded_n,
            padded_k,
            input_scale,
        )?;
        measurements.push(measurement(
            "int8",
            shape,
            logical_m,
            padded_m,
            padded_n,
            padded_k,
            packed_weight_bytes,
            warm_packed_latency_ms,
            warm_quantize_latency_ms,
            warm_end_to_end_latency_ms,
            &cold_packed,
            &cold_end_to_end,
            samples,
            kernel_error,
            quantization_error,
        ));
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn measurement(
    format: &'static str,
    shape: LinearShape,
    logical_m: usize,
    padded_m: usize,
    padded_n: usize,
    padded_k: usize,
    packed_weight_bytes: usize,
    warm_packed_latency_ms: f64,
    warm_quantize_latency_ms: f64,
    warm_end_to_end_latency_ms: f64,
    cold_packed: &[f64],
    cold_end_to_end: &[f64],
    correctness_samples: usize,
    max_abs_kernel_error: f32,
    max_abs_quantization_error: f32,
) -> QuantizedMeasurement {
    let cold_p50_packed_latency_ms = percentile(cold_packed, 50);
    let physical_flops = 2.0 * padded_m as f64 * padded_n as f64 * padded_k as f64;
    QuantizedMeasurement {
        format,
        family: shape.family,
        logical_m,
        logical_n: shape.n,
        logical_k: shape.k,
        padded_m,
        padded_n,
        padded_k,
        packed_weight_bytes,
        warm_packed_latency_ms,
        warm_quantize_latency_ms,
        warm_end_to_end_latency_ms,
        cold_p50_packed_latency_ms,
        cold_p95_packed_latency_ms: percentile(cold_packed, 95),
        cold_p50_end_to_end_latency_ms: percentile(cold_end_to_end, 50),
        cold_p95_end_to_end_latency_ms: percentile(cold_end_to_end, 95),
        warm_packed_effective_weight_gbps: packed_weight_bytes as f64
            / (warm_packed_latency_ms * 1.0e6),
        cold_p50_packed_effective_weight_gbps: packed_weight_bytes as f64
            / (cold_p50_packed_latency_ms * 1.0e6),
        warm_packed_physical_tflops: physical_flops / (warm_packed_latency_ms * 1.0e9),
        correctness_samples,
        max_abs_kernel_error,
        max_abs_quantization_error,
    }
}

fn kernel_report(
    format: &'static str,
    kernel: &'static str,
    function: &CudaFunction,
) -> Result<KernelReport, Box<dyn Error>> {
    let binary = function.binary_version()?;
    Ok(KernelReport {
        format,
        kernel,
        binary_version: format!("{}.{}", binary / 10, binary % 10),
        registers_per_thread: function.num_regs()?,
        static_shared_bytes: function.shared_size_bytes()?,
    })
}

fn time_repeated<F>(
    iterations: usize,
    stream: &CudaStream,
    mut launch: F,
) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Result<(), Box<dyn Error>>,
{
    let mut latencies = Vec::with_capacity(WARM_SAMPLES);
    for _ in 0..WARM_SAMPLES {
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..iterations {
            launch()?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        latencies.push(f64::from(start.elapsed_ms(&end)?) / iterations as f64);
    }
    latencies.sort_by(f64::total_cmp);
    Ok(percentile(&latencies, 50))
}

fn time_cold<F>(
    iterations: usize,
    stream: &CudaStream,
    scrub: &CudaFunction,
    scrub_buffer: &mut CudaSlice<u32>,
    mut launch: F,
) -> Result<Vec<f64>, Box<dyn Error>>
where
    F: FnMut() -> Result<(), Box<dyn Error>>,
{
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        launch_l2_scrub(stream, scrub, scrub_buffer)?;
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        launch()?;
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        latencies.push(f64::from(start.elapsed_ms(&end)?));
    }
    latencies.sort_by(f64::total_cmp);
    Ok(latencies)
}

#[allow(clippy::too_many_arguments)]
fn launch_fp8_linear(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<F8E4M3>,
    weights: &CudaSlice<F8E4M3>,
    weight_scales: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    output: &mut CudaSlice<f16>,
    m: usize,
    n: usize,
    k: usize,
    input_scale: f32,
) -> Result<(), Box<dyn Error>> {
    let activation = 0_i32;
    let (m, n, k) = (i32::try_from(m)?, i32::try_from(n)?, i32::try_from(k)?);
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weights)
        .arg(weight_scales)
        .arg(bias)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&input_scale)
        .arg(&activation);
    unsafe { builder.launch(config) }?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_int8_linear(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<i8>,
    weights: &CudaSlice<i8>,
    weight_scales: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    output: &mut CudaSlice<f16>,
    m: usize,
    n: usize,
    k: usize,
    input_scale: f32,
) -> Result<(), Box<dyn Error>> {
    let activation = 0_i32;
    let (m, n, k) = (i32::try_from(m)?, i32::try_from(n)?, i32::try_from(k)?);
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weights)
        .arg(weight_scales)
        .arg(bias)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&input_scale)
        .arg(&activation);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn launch_fp8_quantize(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<f16>,
    output: &mut CudaSlice<F8E4M3>,
    inverse_scale: f32,
) -> Result<(), Box<dyn Error>> {
    let elements = i32::try_from(input.len())?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&inverse_scale)
        .arg(&elements);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn launch_int8_quantize(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<f16>,
    output: &mut CudaSlice<i8>,
    inverse_scale: f32,
) -> Result<(), Box<dyn Error>> {
    let elements = i32::try_from(input.len())?;
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(output)
        .arg(&inverse_scale)
        .arg(&elements);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn launch_l2_scrub(
    stream: &CudaStream,
    function: &CudaFunction,
    buffer: &mut CudaSlice<u32>,
) -> Result<(), Box<dyn Error>> {
    let elements = i32::try_from(buffer.len())?;
    let config = elementwise_launch_config(buffer.len());
    let mut builder = stream.launch_builder(function);
    builder.arg(buffer).arg(&elements);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn quantized_linear_launch_config(m: usize, n: usize) -> LaunchConfig {
    debug_assert_eq!(m % 16, 0);
    debug_assert_eq!(n % 32, 0);
    LaunchConfig {
        grid_dim: ((n / 32) as u32, (m / 16) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn elementwise_launch_config(elements: usize) -> LaunchConfig {
    let threads = 256_u32;
    LaunchConfig {
        grid_dim: ((elements as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn quantize_fp8_weights(
    source: &[f16],
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
) -> (Vec<F8E4M3>, Vec<f32>) {
    let mut quantized = vec![F8E4M3::ZERO; source.len()];
    let mut scales = vec![1.0_f32; padded_n];
    for (row, output_scale) in scales.iter_mut().take(logical_n).enumerate() {
        let start = row * padded_k;
        let values = &source[start..start + padded_k];
        let scale = values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0_f32, f32::max)
            / FP8_MAX;
        *output_scale = scale;
        for (output, value) in quantized[start..start + padded_k].iter_mut().zip(values) {
            *output = F8E4M3::from_f32(value.to_f32() / scale);
        }
    }
    (quantized, scales)
}

fn quantize_int8_weights(
    source: &[f16],
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut quantized = vec![0_i8; source.len()];
    let mut scales = vec![1.0_f32; padded_n];
    for (row, output_scale) in scales.iter_mut().take(logical_n).enumerate() {
        let start = row * padded_k;
        let values = &source[start..start + padded_k];
        let scale = values
            .iter()
            .map(|value| value.to_f32().abs())
            .fold(0.0_f32, f32::max)
            / INT8_MAX;
        *output_scale = scale;
        for (output, value) in quantized[start..start + padded_k].iter_mut().zip(values) {
            *output = quantize_i8(value.to_f32() / scale);
        }
    }
    (quantized, scales)
}

fn quantize_i8(value: f32) -> i8 {
    value.round_ties_even().clamp(-INT8_MAX, INT8_MAX) as i8
}

#[allow(clippy::too_many_arguments)]
fn check_fp8_samples(
    source_input: &[f16],
    source_weights: &[f16],
    input: &[F8E4M3],
    weights: &[F8E4M3],
    weight_scales: &[f32],
    bias: &[f32],
    output: &[f16],
    logical_m: usize,
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
    input_scale: f32,
) -> Result<(usize, f32, f32), Box<dyn Error>> {
    check_samples(
        source_input,
        source_weights,
        bias,
        output,
        logical_m,
        logical_n,
        padded_n,
        padded_k,
        |row, column| {
            let input_row = &input[row * padded_k..(row + 1) * padded_k];
            let weight_row = &weights[column * padded_k..(column + 1) * padded_k];
            input_row
                .iter()
                .zip(weight_row)
                .fold(0.0_f32, |sum, (left, right)| {
                    sum + left.to_f32() * right.to_f32()
                })
                * input_scale
                * weight_scales[column]
                + bias[column]
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn check_int8_samples(
    source_input: &[f16],
    source_weights: &[f16],
    input: &[i8],
    weights: &[i8],
    weight_scales: &[f32],
    bias: &[f32],
    output: &[f16],
    logical_m: usize,
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
    input_scale: f32,
) -> Result<(usize, f32, f32), Box<dyn Error>> {
    check_samples(
        source_input,
        source_weights,
        bias,
        output,
        logical_m,
        logical_n,
        padded_n,
        padded_k,
        |row, column| {
            let input_row = &input[row * padded_k..(row + 1) * padded_k];
            let weight_row = &weights[column * padded_k..(column + 1) * padded_k];
            input_row
                .iter()
                .zip(weight_row)
                .fold(0_i32, |sum, (left, right)| {
                    sum + i32::from(*left) * i32::from(*right)
                }) as f32
                * input_scale
                * weight_scales[column]
                + bias[column]
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn check_samples<F>(
    source_input: &[f16],
    source_weights: &[f16],
    bias: &[f32],
    output: &[f16],
    logical_m: usize,
    logical_n: usize,
    padded_n: usize,
    padded_k: usize,
    mut quantized_reference: F,
) -> Result<(usize, f32, f32), Box<dyn Error>>
where
    F: FnMut(usize, usize) -> f32,
{
    let rows = sample_indices(logical_m);
    let columns = sample_indices(logical_n);
    let mut samples = 0;
    let mut max_kernel_error = 0.0_f32;
    let mut max_quantization_error = 0.0_f32;
    for row in rows {
        for &column in &columns {
            let quantized = f16::from_f32(quantized_reference(row, column)).to_f32();
            let actual = output[row * padded_n + column].to_f32();
            let source = source_input[row * padded_k..(row + 1) * padded_k]
                .iter()
                .zip(&source_weights[column * padded_k..(column + 1) * padded_k])
                .fold(bias[column], |sum, (left, right)| {
                    sum + left.to_f32() * right.to_f32()
                });
            let source = f16::from_f32(source).to_f32();
            let kernel_error = (actual - quantized).abs();
            if !actual.is_finite() || kernel_error > 0.02 {
                return Err(format!(
                    "quantized kernel parity failed at [{row},{column}]: actual={actual}, expected={quantized}, abs_error={kernel_error}"
                )
                .into());
            }
            max_kernel_error = max_kernel_error.max(kernel_error);
            max_quantization_error = max_quantization_error.max((quantized - source).abs());
            samples += 1;
        }
    }
    Ok((samples, max_kernel_error, max_quantization_error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_fp8_encoding_matches_nvidia_e4m3_examples() {
        assert_eq!(F8E4M3::from_f32(1.0).to_bits(), 0x38);
        assert_eq!(F8E4M3::from_f32(-1.0).to_bits(), 0xb8);
        assert_eq!(F8E4M3::from_f32(448.0).to_bits(), 0x7e);
    }

    #[test]
    fn int8_quantizer_is_symmetric_and_saturating() {
        assert_eq!(quantize_i8(0.5), 0);
        assert_eq!(quantize_i8(1.5), 2);
        assert_eq!(quantize_i8(200.0), 127);
        assert_eq!(quantize_i8(-200.0), -127);
    }

    #[test]
    fn generated_values_span_both_signs() {
        let values: Vec<_> = (0..100)
            .map(|index| super::super::pattern(index, 17, 1.0))
            .collect();
        assert!(values.iter().any(|value| *value < 0.0));
        assert!(values.iter().any(|value| *value > 0.0));
    }
}
