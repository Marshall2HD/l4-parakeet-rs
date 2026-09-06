use super::{
    L2_SCRUB_BYTES, LINEAR_SHAPES, LinearShape, M_BUCKETS, SM89_CUBIN, WARM_SAMPLES, make_bias,
    make_input, make_weights, percentile, round_up, sample_indices,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use serde::Serialize;
use std::error::Error;
use std::sync::Arc;

const Q4_K_ELEMENTS: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q4_K_GROUP: usize = 32;
const Q4_K_GROUPS: usize = Q4_K_ELEMENTS / Q4_K_GROUP;

#[derive(Debug, Serialize)]
pub struct Q4KBenchmarkReport {
    pub schema_version: u32,
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub device_memory_bytes: usize,
    pub format: &'static str,
    pub kernel: &'static str,
    pub kernel_binary_version: String,
    pub kernel_registers_per_thread: i32,
    pub kernel_static_shared_bytes: i32,
    pub operation: &'static str,
    pub unsupported_families: Vec<&'static str>,
    pub warmup_iterations: usize,
    pub warm_samples: usize,
    pub warm_iterations: usize,
    pub cold_iterations: usize,
    pub l2_scrub_bytes: usize,
    pub measurements: Vec<Q4KMeasurement>,
}

#[derive(Debug, Serialize)]
pub struct Q4KMeasurement {
    pub family: &'static str,
    pub logical_m: usize,
    pub logical_n: usize,
    pub logical_k: usize,
    pub padded_m: usize,
    pub padded_n: usize,
    pub packed_weight_bytes: usize,
    pub warm_latency_ms: f64,
    pub cold_p50_latency_ms: f64,
    pub cold_p95_latency_ms: f64,
    pub warm_effective_weight_gbps: f64,
    pub cold_p50_effective_weight_gbps: f64,
    pub correctness_samples: usize,
    pub max_abs_kernel_error: f32,
    pub max_abs_quantization_error: f32,
}

pub fn benchmark_q4_k_linear(
    device: usize,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Q4KBenchmarkReport, Box<dyn Error>> {
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
    let linear = module.load_function("pk_sm89_q4_k_linear_epilogue")?;
    let dequantize = module.load_function("pk_sm89_q4_k_dequantize")?;
    let scrub = module.load_function("pk_sm89_l2_scrub")?;
    let stream = context.default_stream();
    let scrub_elements = L2_SCRUB_BYTES / std::mem::size_of::<u32>();
    let mut scrub_buffer = stream.alloc_zeros::<u32>(scrub_elements)?;
    let mut unsupported_families = Vec::new();
    let mut measurements = Vec::with_capacity(LINEAR_SHAPES.len() * M_BUCKETS.len());

    for shape in LINEAR_SHAPES {
        if !shape.k.is_multiple_of(Q4_K_ELEMENTS) {
            unsupported_families.push(shape.family);
            continue;
        }
        measurements.extend(benchmark_shape(
            &stream,
            &linear,
            &dequantize,
            &scrub,
            &mut scrub_buffer,
            shape,
            warmup_iterations,
            warm_iterations,
            cold_iterations,
        )?);
    }

    let binary = linear.binary_version()?;
    Ok(Q4KBenchmarkReport {
        schema_version: 1,
        device,
        name: context.name()?,
        compute_capability: format!("{major}.{minor}"),
        device_memory_bytes: context.total_mem()?,
        format: "ggml_q4_k",
        kernel: "pk_sm89_q4_k_linear_epilogue",
        kernel_binary_version: format!("{}.{}", binary / 10, binary % 10),
        kernel_registers_per_thread: linear.num_regs()?,
        kernel_static_shared_bytes: linear.shared_size_bytes()?,
        operation: "direct 144-byte GGUF Q4_K superblocks, tile-local FP16 dequantization, FP32 Tensor Core accumulation, fused FP16 bias and output",
        unsupported_families,
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
    dequantize: &CudaFunction,
    scrub: &CudaFunction,
    scrub_buffer: &mut CudaSlice<u32>,
    shape: LinearShape,
    warmup_iterations: usize,
    warm_iterations: usize,
    cold_iterations: usize,
) -> Result<Vec<Q4KMeasurement>, Box<dyn Error>> {
    let padded_n = round_up(shape.n, 128);
    let source_weights = make_weights(shape.n, shape.k, padded_n, shape.k);
    let bias_host = make_bias(shape.n, padded_n);
    let weights_host = quantize_q4_k_weights(&source_weights, padded_n, shape.k);
    let packed_weight_bytes = weights_host.len();
    let weights = stream.clone_htod(&weights_host)?;
    let bias = stream.clone_htod(&bias_host)?;
    validate_device_dequantization(
        stream,
        dequantize,
        &weights,
        &weights_host,
        padded_n,
        shape.k,
    )?;
    let mut measurements = Vec::with_capacity(M_BUCKETS.len());

    for logical_m in M_BUCKETS {
        let padded_m = round_up(logical_m, 16);
        let input_host = make_input(logical_m, shape.k, padded_m, shape.k);
        let input = stream.clone_htod(&input_host)?;
        let mut output = stream.alloc_zeros::<f16>(padded_m * padded_n)?;
        let launch = q4_k_launch_config(padded_m, padded_n);

        stream.synchronize()?;
        for _ in 0..warmup_iterations {
            launch_q4_k(
                stream,
                linear,
                launch,
                &input,
                &weights,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                shape.k,
            )?;
        }
        stream.synchronize()?;

        let warm_latency_ms = time_repeated(warm_iterations, stream, || {
            launch_q4_k(
                stream,
                linear,
                launch,
                &input,
                &weights,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                shape.k,
            )
        })?;
        let cold_latencies = time_cold(cold_iterations, stream, scrub, scrub_buffer, || {
            launch_q4_k(
                stream,
                linear,
                launch,
                &input,
                &weights,
                &bias,
                &mut output,
                padded_m,
                padded_n,
                shape.k,
            )
        })?;

        let output_host = stream.clone_dtoh(&output)?;
        let (correctness_samples, max_abs_kernel_error, max_abs_quantization_error) =
            check_samples(
                &input_host,
                &source_weights,
                &weights_host,
                &bias_host,
                &output_host,
                logical_m,
                shape.n,
                padded_n,
                shape.k,
            )?;
        let cold_p50_latency_ms = percentile(&cold_latencies, 50);
        measurements.push(Q4KMeasurement {
            family: shape.family,
            logical_m,
            logical_n: shape.n,
            logical_k: shape.k,
            padded_m,
            padded_n,
            packed_weight_bytes,
            warm_latency_ms,
            cold_p50_latency_ms,
            cold_p95_latency_ms: percentile(&cold_latencies, 95),
            warm_effective_weight_gbps: packed_weight_bytes as f64 / (warm_latency_ms * 1.0e6),
            cold_p50_effective_weight_gbps: packed_weight_bytes as f64
                / (cold_p50_latency_ms * 1.0e6),
            correctness_samples,
            max_abs_kernel_error,
            max_abs_quantization_error,
        });
    }

    Ok(measurements)
}

#[allow(clippy::too_many_arguments)]
fn launch_q4_k(
    stream: &CudaStream,
    function: &CudaFunction,
    config: LaunchConfig,
    input: &CudaSlice<f16>,
    weights: &CudaSlice<u8>,
    bias: &CudaSlice<f32>,
    output: &mut CudaSlice<f16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let activation = 0_i32;
    let (m, n, k) = (i32::try_from(m)?, i32::try_from(n)?, i32::try_from(k)?);
    let mut builder = stream.launch_builder(function);
    builder
        .arg(input)
        .arg(weights)
        .arg(bias)
        .arg(output)
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&activation);
    unsafe { builder.launch(config) }?;
    Ok(())
}

fn q4_k_launch_config(m: usize, n: usize) -> LaunchConfig {
    debug_assert_eq!(m % 16, 0);
    debug_assert_eq!(n % 64, 0);
    LaunchConfig {
        grid_dim: ((n / 64) as u32, (m / 16) as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn validate_device_dequantization(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    input: &CudaSlice<u8>,
    packed: &[u8],
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let mut output = stream.alloc_zeros::<f16>(n * k)?;
    let elements = n.checked_mul(k).ok_or("Q4_K dequantized size overflow")?;
    let threads = 256_u32;
    let config = LaunchConfig {
        grid_dim: ((u32::try_from(elements)?).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i32, k_i32) = (i32::try_from(n)?, i32::try_from(k)?);
    let mut builder = stream.launch_builder(function);
    builder.arg(input).arg(&mut output).arg(&n_i32).arg(&k_i32);
    unsafe { builder.launch(config) }?;
    let output = stream.clone_dtoh(&output)?;
    for row in sample_indices(n) {
        for column in sample_indices(k) {
            let expected = f16::from_f32(dequantize_q4_k_value(packed, row, column, k)).to_f32();
            let actual = output[row * k + column].to_f32();
            if actual != expected {
                return Err(format!(
                    "device Q4_K dequantization failed at [{row},{column}]: actual={actual}, expected={expected}"
                )
                .into());
            }
        }
    }
    Ok(())
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

fn quantize_q4_k_weights(source: &[f16], rows: usize, columns: usize) -> Vec<u8> {
    assert!(columns.is_multiple_of(Q4_K_ELEMENTS));
    assert_eq!(source.len(), rows * columns);
    let blocks_per_row = columns / Q4_K_ELEMENTS;
    let mut output = vec![0_u8; rows * blocks_per_row * Q4_K_BYTES];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let source_offset = row * columns + block * Q4_K_ELEMENTS;
            let output_offset = (row * blocks_per_row + block) * Q4_K_BYTES;
            encode_q4_k_block(
                &source[source_offset..source_offset + Q4_K_ELEMENTS],
                &mut output[output_offset..output_offset + Q4_K_BYTES],
            );
        }
    }
    output
}

fn encode_q4_k_block(source: &[f16], output: &mut [u8]) {
    assert_eq!(source.len(), Q4_K_ELEMENTS);
    assert_eq!(output.len(), Q4_K_BYTES);
    output.fill(0);

    let mut levels = [0_u8; Q4_K_ELEMENTS];
    let mut scales = [0.0_f32; Q4_K_GROUPS];
    let mut minimums = [0.0_f32; Q4_K_GROUPS];
    let mut max_scale = 0.0_f32;
    let mut max_minimum = 0.0_f32;
    for group in 0..Q4_K_GROUPS {
        let start = group * Q4_K_GROUP;
        let values = &source[start..start + Q4_K_GROUP];
        let mean_square = values
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>()
            / Q4_K_GROUP as f32;
        let rms = mean_square.sqrt();
        let weights: Vec<_> = values
            .iter()
            .map(|value| rms + value.to_f32().abs())
            .collect();
        let (scale, minimum) =
            make_qkx2_quants(values, &weights, &mut levels[start..start + Q4_K_GROUP]);
        scales[group] = scale;
        minimums[group] = minimum;
        max_scale = max_scale.max(scale);
        max_minimum = max_minimum.max(minimum);
    }

    let inverse_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inverse_minimum = if max_minimum > 0.0 {
        63.0 / max_minimum
    } else {
        0.0
    };
    let mut packed_scales = [0_u8; 12];
    for group in 0..Q4_K_GROUPS {
        let scale = nearest_int(inverse_scale * scales[group]).clamp(0, 63) as u8;
        let minimum = nearest_int(inverse_minimum * minimums[group]).clamp(0, 63) as u8;
        if group < 4 {
            packed_scales[group] = scale;
            packed_scales[group + 4] = minimum;
        } else {
            packed_scales[group + 4] = (scale & 0x0f) | ((minimum & 0x0f) << 4);
            packed_scales[group - 4] |= (scale >> 4) << 6;
            packed_scales[group] |= (minimum >> 4) << 6;
        }
    }

    let super_scale = f16::from_f32(max_scale / 63.0);
    let super_minimum = f16::from_f32(max_minimum / 63.0);
    output[0..2].copy_from_slice(&super_scale.to_bits().to_le_bytes());
    output[2..4].copy_from_slice(&super_minimum.to_bits().to_le_bytes());
    output[4..16].copy_from_slice(&packed_scales);

    for group in 0..Q4_K_GROUPS {
        let (scale, minimum) = q4_k_scale_min(&packed_scales, group);
        let scale = super_scale.to_f32() * f32::from(scale);
        if scale == 0.0 {
            continue;
        }
        let minimum = super_minimum.to_f32() * f32::from(minimum);
        let start = group * Q4_K_GROUP;
        for index in 0..Q4_K_GROUP {
            levels[start + index] =
                nearest_int((source[start + index].to_f32() + minimum) / scale).clamp(0, 15) as u8;
        }
    }

    for pair in 0..(Q4_K_GROUPS / 2) {
        let first = pair * 2 * Q4_K_GROUP;
        let second = first + Q4_K_GROUP;
        for index in 0..Q4_K_GROUP {
            output[16 + pair * Q4_K_GROUP + index] =
                levels[first + index] | (levels[second + index] << 4);
        }
    }
}

fn make_qkx2_quants(source: &[f16], weights: &[f32], levels: &mut [u8]) -> (f32, f32) {
    debug_assert_eq!(source.len(), Q4_K_GROUP);
    debug_assert_eq!(weights.len(), source.len());
    debug_assert_eq!(levels.len(), source.len());

    let mut minimum = source[0].to_f32();
    let mut maximum = minimum;
    let mut sum_weight = weights[0];
    let mut sum_source = weights[0] * minimum;
    for (value, weight) in source.iter().zip(weights).skip(1) {
        minimum = minimum.min(value.to_f32());
        maximum = maximum.max(value.to_f32());
        sum_weight += weight;
        sum_source += weight * value.to_f32();
    }
    minimum = minimum.min(0.0);
    if maximum == minimum {
        levels.fill(0);
        return (0.0, -minimum);
    }

    let mut inverse_scale = 15.0 / (maximum - minimum);
    let mut scale = inverse_scale.recip();
    let mut best_error = 0.0_f32;
    for (index, (value, weight)) in source.iter().zip(weights).enumerate() {
        let level = nearest_int(inverse_scale * (value.to_f32() - minimum)).clamp(0, 15);
        levels[index] = level as u8;
        let difference = scale * level as f32 + minimum - value.to_f32();
        best_error += weight * difference * difference;
    }

    let mut auxiliary = [0_u8; Q4_K_GROUP];
    for step in 0..=20 {
        inverse_scale = (-1.0 + 0.1 * step as f32 + 15.0) / (maximum - minimum);
        let mut sum_level = 0.0_f32;
        let mut sum_level_squared = 0.0_f32;
        let mut sum_source_level = 0.0_f32;
        for (index, (value, weight)) in source.iter().zip(weights).enumerate() {
            let level = nearest_int(inverse_scale * (value.to_f32() - minimum)).clamp(0, 15);
            auxiliary[index] = level as u8;
            sum_level += weight * level as f32;
            sum_level_squared += weight * (level * level) as f32;
            sum_source_level += weight * level as f32 * value.to_f32();
        }
        let determinant = sum_weight * sum_level_squared - sum_level * sum_level;
        if determinant <= 0.0 {
            continue;
        }
        let mut candidate_scale =
            (sum_weight * sum_source_level - sum_source * sum_level) / determinant;
        let mut candidate_minimum =
            (sum_level_squared * sum_source - sum_level * sum_source_level) / determinant;
        if candidate_minimum > 0.0 {
            candidate_minimum = 0.0;
            candidate_scale = sum_source_level / sum_level_squared;
        }
        let error = source
            .iter()
            .zip(weights)
            .zip(auxiliary)
            .map(|((value, weight), level)| {
                let difference =
                    candidate_scale * f32::from(level) + candidate_minimum - value.to_f32();
                weight * difference * difference
            })
            .sum::<f32>();
        if error < best_error {
            levels.copy_from_slice(&auxiliary);
            best_error = error;
            scale = candidate_scale;
            minimum = candidate_minimum;
        }
    }
    (scale, -minimum)
}

fn nearest_int(value: f32) -> i32 {
    value.round_ties_even() as i32
}

fn q4_k_scale_min(packed: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (packed[group] & 63, packed[group + 4] & 63)
    } else {
        (
            (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4),
            (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4),
        )
    }
}

fn dequantize_q4_k_value(weights: &[u8], row: usize, column: usize, row_size: usize) -> f32 {
    let blocks_per_row = row_size / Q4_K_ELEMENTS;
    let block = column / Q4_K_ELEMENTS;
    let offset = (row * blocks_per_row + block) * Q4_K_BYTES;
    let data = &weights[offset..offset + Q4_K_BYTES];
    let super_scale = f16::from_bits(u16::from_le_bytes([data[0], data[1]])).to_f32();
    let super_minimum = f16::from_bits(u16::from_le_bytes([data[2], data[3]])).to_f32();
    let input_in_block = column % Q4_K_ELEMENTS;
    let group = input_in_block / Q4_K_GROUP;
    let value_in_group = input_in_block % Q4_K_GROUP;
    let packed = data[16 + (group / 2) * Q4_K_GROUP + value_in_group];
    let quant = if group.is_multiple_of(2) {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let (scale, minimum) = q4_k_scale_min(&data[4..16], group);
    super_scale * f32::from(scale) * f32::from(quant) - super_minimum * f32::from(minimum)
}

#[allow(clippy::too_many_arguments)]
fn check_samples(
    input: &[f16],
    source_weights: &[f16],
    weights: &[u8],
    bias: &[f32],
    output: &[f16],
    logical_m: usize,
    logical_n: usize,
    padded_n: usize,
    k: usize,
) -> Result<(usize, f32, f32), Box<dyn Error>> {
    let rows = sample_indices(logical_m);
    let columns = sample_indices(logical_n);
    let mut samples = 0;
    let mut max_kernel_error = 0.0_f32;
    let mut max_quantization_error = 0.0_f32;
    for row in rows {
        for &column in &columns {
            let input_row = &input[row * k..(row + 1) * k];
            let quantized =
                input_row
                    .iter()
                    .enumerate()
                    .fold(bias[column], |sum, (input_index, value)| {
                        let weight =
                            f16::from_f32(dequantize_q4_k_value(weights, column, input_index, k));
                        sum + value.to_f32() * weight.to_f32()
                    });
            let quantized = f16::from_f32(quantized).to_f32();
            let source = input_row
                .iter()
                .zip(&source_weights[column * k..(column + 1) * k])
                .fold(bias[column], |sum, (left, right)| {
                    sum + left.to_f32() * right.to_f32()
                });
            let source = f16::from_f32(source).to_f32();
            let actual = output[row * padded_n + column].to_f32();
            let kernel_error = (actual - quantized).abs();
            if !actual.is_finite() || kernel_error > 0.02 {
                return Err(format!(
                    "Q4_K kernel parity failed at [{row},{column}]: actual={actual}, expected={quantized}, abs_error={kernel_error}"
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
    fn q4_k_block_has_the_gguf_layout_size() {
        let source: Vec<_> = (0..Q4_K_ELEMENTS)
            .map(|index| f16::from_f32(super::super::pattern(index, 29, 0.03125)))
            .collect();
        let mut block = vec![0_u8; Q4_K_BYTES];
        encode_q4_k_block(&source, &mut block);
        assert_eq!(block.len(), 144);
        assert!(block.iter().any(|value| *value != 0));
    }

    #[test]
    fn q4_k_zero_block_round_trips_to_zero() {
        let source = vec![f16::ZERO; Q4_K_ELEMENTS];
        let packed = quantize_q4_k_weights(&source, 1, Q4_K_ELEMENTS);
        for column in 0..Q4_K_ELEMENTS {
            assert_eq!(
                dequantize_q4_k_value(&packed, 0, column, Q4_K_ELEMENTS),
                0.0
            );
        }
    }
}
