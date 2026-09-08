#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace wmma = nvcuda::wmma;

constexpr int kModelWidth = 1024;
constexpr int kHeadWidth = 128;
constexpr int kHeads = 8;
constexpr int kMaxAttention = 257;
constexpr int kPositionRowsPadded = 272;
constexpr int kFp8ContextRowsPadded = 288;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_sum_tree_tail(float value, float* shared) {
    const int lane = static_cast<int>(threadIdx.x);
    shared[lane] = value;
    __syncthreads();
    if (lane < 32) {
        // Warp zero owns eight coalesced leaf vectors. Preserve the original
        // widths128/64/32 tree exactly before the unchanged shuffle tail.
        const float s0 = shared[lane] + shared[lane + 128];
        const float s1 = shared[lane + 64] + shared[lane + 192];
        const float s2 = shared[lane + 32] + shared[lane + 160];
        const float s3 = shared[lane + 96] + shared[lane + 224];
        value = warp_sum((s0 + s1) + (s2 + s3));
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_max_tree_tail(float value, float* shared) {
    const int lane = static_cast<int>(threadIdx.x);
    shared[lane] = value;
    __syncthreads();
    if (lane < 32) {
        const float s0 = fmaxf(shared[lane], shared[lane + 128]);
        const float s1 = fmaxf(shared[lane + 64], shared[lane + 192]);
        const float s2 = fmaxf(shared[lane + 32], shared[lane + 160]);
        const float s3 = fmaxf(shared[lane + 96], shared[lane + 224]);
        value = warp_max(fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float half_warp_sum(float value) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset, 16);
    }
    return value;
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_layer_norm(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int rows,
    int padded_rows,
    float epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= padded_rows) {
        return;
    }
    if (row >= rows) {
        for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
            output[row * kModelWidth + feature] = __float2half_rn(0.0f);
        }
        return;
    }

    __shared__ float reductions[256];
    float sum = 0.0f;
    for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
        sum += __half2float(input[row * kModelWidth + feature]);
    }
    const float mean = block_sum_tree_tail(sum, reductions) / kModelWidth;
    // All warps must consume the broadcast before variance reuses its slot.
    __syncthreads();

    float square_sum = 0.0f;
    for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
        const float difference =
            __half2float(input[row * kModelWidth + feature]) - mean;
        square_sum += difference * difference;
    }
    const float inverse_std =
        rsqrtf(block_sum_tree_tail(square_sum, reductions) / kModelWidth + epsilon);
    for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
        const int index = row * kModelWidth + feature;
        const float normalized =
            (__half2float(input[index]) - mean) * inverse_std;
        output[index] = __float2half_rn(
            normalized * __half2float(weight[feature]) + __half2float(bias[feature]));
    }
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_layer_norm_pair(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const __half* __restrict__ next_weight,
    const __half* __restrict__ next_bias,
    __half* __restrict__ next_output,
    int rows,
    int padded_rows,
    float epsilon,
    float inverse_scale) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= padded_rows) return;
    if (row >= rows) {
        for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
            output[row * kModelWidth + feature] = __float2half_rn(0.0f);
            if (inverse_scale != 0.0f) {
                reinterpret_cast<uint8_t*>(next_output)[row * kModelWidth + feature] = 0;
            } else {
                next_output[row * kModelWidth + feature] = __float2half_rn(0.0f);
            }
        }
        return;
    }
    __shared__ float reductions[256];
    float sum = 0.0f;
    for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
        sum += __half2float(input[row * kModelWidth + feature]);
    }
    const float mean = block_sum_tree_tail(sum, reductions) / kModelWidth;
    __syncthreads();
    float square_sum = 0.0f;
    for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
        const float difference = __half2float(input[row * kModelWidth + feature]) - mean;
        square_sum += difference * difference;
    }
    const float inverse_std =
        rsqrtf(block_sum_tree_tail(square_sum, reductions) / kModelWidth + epsilon);
    float rounded[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * 256;
        const int index = row * kModelWidth + feature;
        const float normalized = (__half2float(input[index]) - mean) * inverse_std;
        const __half first = __float2half_rn(
            normalized * __half2float(weight[feature]) + __half2float(bias[feature]));
        // This intermediate is still required by the next FF1 residual.
        output[index] = first;
        rounded[item] = __half2float(first);
    }
    // Retire the first variance broadcast before reusing reduction storage.
    __syncthreads();
    float next_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) next_sum += rounded[item];
    const float next_mean = block_sum_tree_tail(next_sum, reductions) / kModelWidth;
    __syncthreads();
    float next_square_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float difference = rounded[item] - next_mean;
        next_square_sum += difference * difference;
    }
    const float next_inverse_std =
        rsqrtf(block_sum_tree_tail(next_square_sum, reductions) / kModelWidth + epsilon);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * 256;
        const float normalized = (rounded[item] - next_mean) * next_inverse_std;
        const __half second = __float2half_rn(
            normalized * __half2float(next_weight[feature]) + __half2float(next_bias[feature]));
        if (inverse_scale != 0.0f) {
            reinterpret_cast<uint8_t*>(next_output)[row * kModelWidth + feature] =
                __nv_fp8_e4m3(__half2float(second) * inverse_scale).__x;
        } else {
            next_output[row * kModelWidth + feature] = second;
        }
    }
}

template<bool kInt4>
__device__ __forceinline__ void layer_norm_quantize_dynamic(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int padded_rows,
    float epsilon) {
    const int row = static_cast<int>(gridDim.x - 1 - blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= padded_rows) {
        return;
    }
    if (row >= rows) {
        for (int feature = lane; feature < kModelWidth; feature += blockDim.x) {
            if constexpr (kInt4) {
                if ((lane & 1) == 0) output[(row * kModelWidth + feature) / 2] = 0;
            } else {
                output[row * kModelWidth + feature] = __nv_fp8_e4m3(0.0f).__x;
            }
        }
        if (lane == 0) {
            scales[row] = 1.0f;
        }
        return;
    }

    __shared__ float reductions[256];
    float values[4];
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        values[item] = __half2float(input[row * kModelWidth + lane + item * blockDim.x]);
        sum += values[item];
    }
    const float mean = block_sum_tree_tail(sum, reductions) / kModelWidth;
    __syncthreads();

    float square_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float difference = values[item] - mean;
        square_sum += difference * difference;
    }
    const float inverse_std =
        rsqrtf(block_sum_tree_tail(square_sum, reductions) / kModelWidth + epsilon);
    __syncthreads();
    float maximum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * blockDim.x;
        values[item] = __half2float(__float2half_rn(
            (values[item] - mean) * inverse_std * __half2float(weight[feature]) +
            __half2float(bias[feature])));
        if constexpr (kInt4) {
            // Use every signed hardware code while retaining a zero anchor.
            maximum = fmaxf(maximum, fmaxf(values[item] / 7.0f, -values[item] / 8.0f));
        } else {
            maximum = fmaxf(maximum, fabsf(values[item]));
        }
    }
    maximum = block_max_tree_tail(maximum, reductions);
    const float scale = maximum > 0.0f ? (kInt4 ? maximum : maximum / 448.0f) : 1.0f;
    if (lane == 0) {
        scales[row] = scale;
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int feature = lane + item * blockDim.x;
        if constexpr (kInt4) {
            const int q = max(-8, min(7, __float2int_rn(values[item] / scale)));
            const int other = __shfl_xor_sync(0xffffffff, q, 1);
            if ((lane & 1) == 0) {
                output[(row * kModelWidth + feature) / 2] = (q & 15) | ((other & 15) << 4);
            }
        } else {
            output[row * kModelWidth + feature] = __nv_fp8_e4m3(values[item] / scale).__x;
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_layer_norm_quantize_fp8_dynamic(
    const __half* input, const __half* weight, const __half* bias,
    uint8_t* output, float* scales, int rows, int padded_rows, float epsilon) {
    layer_norm_quantize_dynamic<false>(input, weight, bias, output, scales, rows, padded_rows, epsilon);
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_layer_norm_quantize_int4_dynamic(
    const __half* input, const __half* weight, const __half* bias,
    uint8_t* output, float* scales, int rows, int padded_rows, float epsilon) {
    layer_norm_quantize_dynamic<true>(input, weight, bias, output, scales, rows, padded_rows, epsilon);
}

extern "C" __global__
void pk_sm89_glu_masked(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int padded_rows,
    int valid_rows,
    const int32_t* __restrict__ bounds) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = padded_rows * kModelWidth;
    if (index >= total) {
        return;
    }
    const int row = index / kModelWidth;
    const int feature = index - row * kModelWidth;
    if (row >= rows || row >= valid_rows || (bounds && row >= bounds[2 * row + 1])) {
        output[index] = __float2half_rn(0.0f);
        return;
    }
    const int expanded = row * (2 * kModelWidth);
    const float value = __half2float(input[expanded + feature]);
    const float gate = __half2float(input[expanded + kModelWidth + feature]);
    output[index] = __float2half_rn(value / (1.0f + expf(-gate)));
}

extern "C" __global__
void pk_sm89_silu_in_place(__half* values, int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        const float value = __half2float(values[index]);
        values[index] = __float2half_rn(value / (1.0f + expf(-value)));
    }
}

__device__ __forceinline__ void copy_16_async(__half* destination, const __half* source) {
    const unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared), "l"(source));
}

__device__ __forceinline__ void copy_16_bytes_async(uint8_t* destination, const uint8_t* source) {
    const unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared), "l"(source));
}

__device__ __forceinline__ uint32_t load_shared_u32(const uint8_t* pointer) {
    return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__device__ __forceinline__ void mma_m16n8k32_fp8(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void copy_ffn_expand_tile_async(
    __half* shared,
    const __half* input,
    const __half* weight,
    int row,
    int inner,
    int output,
    int rows) {
    constexpr int kTileRows = 128;
    constexpr int kTileColumns = 128;
    constexpr int kTileInner = 32;
    constexpr int kInputWidth = 1024;
    constexpr int kInputValues = kTileRows * kTileInner;

    #pragma unroll
    for (int load = 0; load < 2; ++load) {
        const int input_index = (static_cast<int>(threadIdx.x) + load * blockDim.x) * 8;
        const int input_row = input_index / kTileInner;
        const int input_inner = input_index - input_row * kTileInner;
        __half* shared_input = shared + input_index;
        if (row + input_row < rows) {
            copy_16_async(
                shared_input,
                input + (row + input_row) * kInputWidth + inner + input_inner);
        } else {
            *reinterpret_cast<uint4*>(shared_input) = make_uint4(0, 0, 0, 0);
        }
    }

    __half* shared_weight = shared + kInputValues;
#pragma unroll
    for (int load = 0; load < 2; ++load) {
        const int weight_index = (static_cast<int>(threadIdx.x) + load * blockDim.x) * 8;
        const int weight_row = weight_index / kTileInner;
        const int weight_inner = weight_index - weight_row * kTileInner;
        copy_16_async(
            shared_weight + weight_index,
            weight + (output + weight_row) * kInputWidth + inner + weight_inner);
    }
}

// Fixed-shape L4 FFN expansion. Two cp.async stages keep the next 128x32
// activation and 128x32 weight tiles in flight while eight warps compute the
// current 128x128 output tile. Each loaded fragment feeds eight WMMA operations;
// SiLU is fused into the final FP16 store.
extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_expand_async(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int rows) {
    constexpr int kTileRows = 128;
    constexpr int kTileColumns = 128;
    constexpr int kTileInner = 32;
    constexpr int kInputWidth = 1024;
    constexpr int kOutputWidth = 4096;
    constexpr int kInputValues = kTileRows * kTileInner;
    constexpr int kWeightValues = kTileColumns * kTileInner;
    constexpr int kStageValues = kInputValues + kWeightValues;

    extern __shared__ __half shared_values[];
    __shared__ float scratch[8][16 * 16];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp_row = (warp >> 1) * 32;
    const int warp_column = (warp & 1) * 64;
    const int row = static_cast<int>(blockIdx.y) * kTileRows;
    const int column = static_cast<int>(blockIdx.x) * kTileColumns;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[2][4];
#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < 4; ++column_tile) {
            wmma::fill_fragment(accumulators[row_tile][column_tile], 0.0f);
        }
    }

    copy_ffn_expand_tile_async(
        shared_values, input, weight, row, 0, column, rows);
    asm volatile("cp.async.commit_group;\n");

    for (int inner = 0; inner < kInputWidth; inner += kTileInner) {
        const int stage = (inner / kTileInner) & 1;
        __half* stage_values = shared_values + stage * kStageValues;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        if (inner + kTileInner < kInputWidth) {
            const int next_stage = stage ^ 1;
            copy_ffn_expand_tile_async(
                shared_values + next_stage * kStageValues,
                input,
                weight,
                row,
                inner + kTileInner,
                column,
                rows);
            asm volatile("cp.async.commit_group;\n");
        }

        __half* shared_input = stage_values;
        __half* shared_weight = stage_values + kInputValues;
#pragma unroll
        for (int tile_inner = 0; tile_inner < kTileInner; tile_inner += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
                input_fragments[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
                weight_fragments[4];
#pragma unroll
            for (int row_tile = 0; row_tile < 2; ++row_tile) {
                wmma::load_matrix_sync(
                    input_fragments[row_tile],
                    shared_input + (warp_row + row_tile * 16) * kTileInner + tile_inner,
                    kTileInner);
            }
#pragma unroll
            for (int column_tile = 0; column_tile < 4; ++column_tile) {
                wmma::load_matrix_sync(
                    weight_fragments[column_tile],
                    shared_weight + (warp_column + column_tile * 16) * kTileInner + tile_inner,
                    kTileInner);
            }
#pragma unroll
            for (int row_tile = 0; row_tile < 2; ++row_tile) {
#pragma unroll
                for (int column_tile = 0; column_tile < 4; ++column_tile) {
                    wmma::mma_sync(
                        accumulators[row_tile][column_tile],
                        input_fragments[row_tile],
                        weight_fragments[column_tile],
                        accumulators[row_tile][column_tile]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        const int output_row = row + warp_row + row_tile * 16;
        if (output_row >= rows) {
            continue;
        }
#pragma unroll
        for (int column_tile = 0; column_tile < 4; ++column_tile) {
            const int output_column = column + warp_column + column_tile * 16;
            wmma::store_matrix_sync(
                scratch[warp],
                accumulators[row_tile][column_tile],
                16,
                wmma::mem_row_major);
            __syncwarp();
            for (int index = lane; index < 16 * 16; index += 32) {
                const int tile_row = index / 16;
                const int tile_column = index - tile_row * 16;
                if (output_row + tile_row < rows) {
                    const float value = scratch[warp][index];
                    output[(output_row + tile_row) * kOutputWidth + output_column + tile_column] =
                        __float2half_rn(value / (1.0f + expf(-value)));
                }
            }
            __syncwarp();
        }
    }
}

extern "C" __global__
void pk_sm89_quantize_fp8_fixed(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int elements,
    float inverse_scale) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        output[index] = __nv_fp8_e4m3(__half2float(input[index]) * inverse_scale).__x;
    }
}

// Feature-major layout: original FP8 V, packed S4 V, then 1024 FP32 scales.
// The two-pass fallback also supports rows beyond the register strip's capacity.
extern "C" __global__ void pk_sm89_quantize_value_int4(
    const __half* input, uint8_t* output, int elements, float inverse_scale) {
    const int feature = blockIdx.x, tid = threadIdx.x;
    const int rows = elements / kModelWidth;
    __shared__ float maxima[8];
    __shared__ float scale;
    float maximum = 0.0f;
    for (int row = 2 * tid; row < rows; row += 512) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2*>(input + feature * rows + row));
        maximum = fmaxf(maximum, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    maximum = warp_max(maximum);
    if ((tid & 31) == 0) maxima[tid >> 5] = maximum;
    __syncthreads();
    if (tid < 32) {
        maximum = warp_max(tid < 8 ? maxima[tid] : 0.0f);
        if (tid == 0) {
            scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
            reinterpret_cast<float*>(output + elements + elements / 2)[feature] = scale;
        }
    }
    __syncthreads();
    for (int row = 2 * tid; row < rows; row += 512) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2*>(input + feature * rows + row));
        const uint16_t original = uint16_t(__nv_fp8_e4m3(v.x * inverse_scale).__x) |
            (uint16_t(__nv_fp8_e4m3(v.y * inverse_scale).__x) << 8);
        *reinterpret_cast<uint16_t*>(output + feature * rows + row) = original;
        const int a = max(-7, min(7, __float2int_rn(v.x / scale)));
        const int b = max(-7, min(7, __float2int_rn(v.y / scale)));
        output[elements + (feature * rows + row) / 2] = (a & 15) | ((b & 15) << 4);
    }
}

// 1024 lanes retain 24 half2 values each: at most 49152 rows, one global read.
extern "C" __global__ __launch_bounds__(1024, 1)
void pk_sm89_quantize_value_int4_register(
    const __half* input, uint8_t* output, int elements, float inverse_scale, int valid_rows) {
    const int feature = blockIdx.x, tid = threadIdx.x;
    const int rows = elements / kModelWidth;
    __shared__ float maxima[32];
    __shared__ float scale;
    __half2 saved[24];
    float maximum = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < 24; ++chunk) {
        const int row = chunk * 2048 + 2 * tid;
        saved[chunk] = row < rows ? *reinterpret_cast<const __half2*>(input + feature * rows + row)
            : __float2half2_rn(0.0f);
        const float2 v = __half22float2(saved[chunk]);
        maximum = fmaxf(maximum, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    maximum = warp_max(maximum);
    if ((tid & 31) == 0) maxima[tid >> 5] = maximum;
    __syncthreads();
    if (tid < 32) {
        maximum = warp_max(maxima[tid]);
        if (tid == 0) {
            scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
            reinterpret_cast<float*>(output + elements + elements / 2)[feature] = scale;
        }
    }
    __syncthreads();
#pragma unroll
    for (int chunk = 0; chunk < 24; ++chunk) {
        const int row = chunk * 2048 + 2 * tid;
        if (row < rows) {
            const float2 v = __half22float2(saved[chunk]);
            // Only scalar boundary tiles consume original FP8 V. Their key
            // windows lie inside these conservative 272-row edge regions.
            if (row < 272 || row + 1 >= valid_rows - 272) {
                const uint16_t original = uint16_t(__nv_fp8_e4m3(v.x * inverse_scale).__x) |
                    (uint16_t(__nv_fp8_e4m3(v.y * inverse_scale).__x) << 8);
                *reinterpret_cast<uint16_t*>(output + feature * rows + row) = original;
            }
            const int a = max(-7, min(7, __float2int_rn(v.x / scale)));
            const int b = max(-7, min(7, __float2int_rn(v.y / scale)));
            output[elements + (feature * rows + row) / 2] = (a & 15) | ((b & 15) << 4);
        }
    }
}

extern "C" __global__
void pk_sm89_quantize_fp8_fixed_transpose(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows,
    float inverse_scale) {
    __shared__ uint8_t tile[32][33];
    const int input_feature = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
    const int first_input_row = static_cast<int>(blockIdx.y) * 32 + threadIdx.y;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int input_row = first_input_row + offset;
        if (input_row < rows) {
            tile[threadIdx.y + offset][threadIdx.x] = __nv_fp8_e4m3(
                __half2float(input[input_row * kModelWidth + input_feature]) * inverse_scale).__x;
        }
    }
    __syncthreads();

    const int output_feature = static_cast<int>(blockIdx.x) * 32 + threadIdx.y;
    const int output_row = static_cast<int>(blockIdx.y) * 32 + threadIdx.x;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        if (output_row < rows) {
            output[(output_feature + offset) * rows + output_row] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

// Bijective 16-byte segment swizzle within each 64-byte shared operand row.
// Both asynchronous stores and ldmatrix row addresses use this mapping.
__device__ __forceinline__ int fp8_shared_index(int index) {
    return index ^ (((index / 64) & 3) * 16);
}

// Compressed 32-byte rows need a different bijective segment permutation.
__device__ __forceinline__ int fp8_sparse_input_index(int index) {
    return index ^ (((index / 32) & 7) * 16);
}

__device__ __forceinline__ uint32_t compress_fp8_quartet(uint32_t packed) {
    uint32_t data = 0, metadata = 0;
    int count = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned magnitude = (packed >> (8 * i)) & 127;
        int rank = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned other = (packed >> (8 * j)) & 127;
            rank += other > magnitude || (other == magnitude && j < i);
        }
        if (rank < 2) {
            data |= ((packed >> (8 * i)) & 255) << (8 * count);
            metadata |= i << (2 * count);
            ++count;
        }
    }
    return data | (metadata << 16);
}

__device__ __forceinline__ void mma_m16n8k64_sparse_fp8(
    float* d, const uint32_t* a, const uint32_t* b, uint32_t metadata) {
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(metadata));
}

template<int kInputWidth, bool kPairedGlu, bool kSparse>
__device__ __forceinline__ void copy_fp8_ffn_tile_async(
    uint8_t* shared,
    const uint8_t* input,
    const uint8_t* weight,
    int row,
    int inner,
    int output,
    int rows) {
    constexpr int kTileRows = 128;
    constexpr int kTileColumns = 128;
    constexpr int kTileInner = 64;
    constexpr int kInputStride = kSparse ? 32 : kTileInner;
    constexpr int kInputValues = kTileRows * kInputStride;

#pragma unroll
    for (int load = 0; load < (kSparse ? 1 : 2); ++load) {
        const int input_index = (static_cast<int>(threadIdx.x) + load * blockDim.x) * 16;
        const int input_row = input_index / kInputStride;
        const int input_inner = input_index - input_row * kInputStride;
        uint8_t* shared_input = shared + (kSparse
            ? fp8_sparse_input_index(input_index) : fp8_shared_index(input_index));
        if (row + input_row < rows) {
            copy_16_bytes_async(
                shared_input,
                input + (row + input_row) * (kInputWidth / (kSparse ? 2 : 1)) +
                    inner / (kSparse ? 2 : 1) + input_inner);
        } else {
            *reinterpret_cast<uint4*>(shared_input) = make_uint4(0, 0, 0, 0);
        }
    }

    uint8_t* shared_weight = shared + kInputValues;
#pragma unroll
    for (int load = 0; load < 2; ++load) {
        const int weight_index = (static_cast<int>(threadIdx.x) + load * blockDim.x) * 16;
        const int weight_row = weight_index / kTileInner;
        const int weight_inner = weight_index - weight_row * kTileInner;
        const int source_row = output + (kPairedGlu
            ? weight_row % 64 + (weight_row / 64) * 1024 : weight_row);
        copy_16_bytes_async(
            shared_weight + fp8_shared_index(weight_index),
            weight + source_row * kInputWidth + inner + weight_inner);
    }
    if constexpr (kSparse) {
        if (threadIdx.x < 128) {
            auto* destination = shared + kInputValues + kTileColumns * kTileInner + threadIdx.x * 8;
            if (row + threadIdx.x < rows) {
                const auto* source = input + size_t(rows) * (kInputWidth / 2) +
                    size_t(row + threadIdx.x) * (kInputWidth / 8) + inner / 8;
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                    :: "r"(address), "l"(source));
            } else {
                *reinterpret_cast<uint64_t*>(destination) = 0x4444444444444444ull;
            }
        }
    }
}

__device__ __forceinline__ void mma_m16n8k64_int4(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Native E4M3 or packed S4 operands. kInputWidth counts stored bytes: the
// same 32-byte operand fragments feed K32 FP8 or K64 INT4 instructions.
// The 128x128 tile and two 64-byte cp.async stages retain two CTAs per L4 SM.
template<int kInputWidth, int kOutputWidth, int kEpilogue, bool kInt4 = false, bool kReverse = false>
__device__ __forceinline__ void ffn_fp8_async(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    const float* __restrict__ input_scales,
    __half* __restrict__ output,
    int rows,
    float input_scale,
    int dynamic_scale,
    const float* bias = nullptr,
    int valid_rows = 0,
    int inner_width = kInputWidth,
    uint8_t* __restrict__ packed_key = nullptr,
    float* __restrict__ key_scales = nullptr,
    __half* __restrict__ value_t = nullptr) {
    constexpr int kTileRows = 128;
    constexpr int kTileColumns = 128;
    constexpr int kTileInner = 64;
    constexpr bool kSparse = kEpilogue == 8;
    constexpr int kInputStride = kSparse ? 32 : kTileInner;
    constexpr int kInputValues = kTileRows * kInputStride;
    constexpr int kWeightValues = kTileColumns * kTileInner;
    constexpr int kStageValues = kInputValues + kWeightValues + (kSparse ? 1024 : 0);

    extern __shared__ uint8_t shared_bytes[];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int warp_row = (warp >> 1) * 32;
    const int warp_column = (warp & 1) * 64;
    const int row = static_cast<int>(kReverse ? gridDim.y - 1 - blockIdx.y : blockIdx.y) * kTileRows;
    const int column = static_cast<int>(blockIdx.x) * (kEpilogue == 5 ? 64 : kTileColumns);

    using Accumulator = std::conditional_t<kInt4, int32_t, float>;
    Accumulator accumulators[2][8][4] = {};
    copy_fp8_ffn_tile_async<kInputWidth, kEpilogue == 5, kSparse>(
        shared_bytes, input, weight, row, 0, column, rows);
    asm volatile("cp.async.commit_group;\n");

    // Automatic full unrolling at K=256 spills the pointwise accumulator state.
#pragma unroll 1
    for (int inner = 0; inner < inner_width; inner += kTileInner) {
        const int stage = (inner / kTileInner) & 1;
        uint8_t* stage_values = shared_bytes + stage * kStageValues;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        if (inner + kTileInner < inner_width) {
            copy_fp8_ffn_tile_async<kInputWidth, kEpilogue == 5, kSparse>(
                shared_bytes + (stage ^ 1) * kStageValues,
                input,
                weight,
                row,
                inner + kTileInner,
                column,
                rows);
            asm volatile("cp.async.commit_group;\n");
        }

        const uint8_t* shared_input = stage_values;
        const uint8_t* shared_weight = stage_values + kInputValues;
#pragma unroll
        for (int tile_inner = 0; tile_inner < kTileInner; tile_inner += (kSparse ? 64 : 32)) {
            uint32_t input_fragments[2][4];
            uint32_t metadata[2] = {};
#pragma unroll
            for (int row_tile = 0; row_tile < 2; ++row_tile) {
                const int load_row = warp_row + row_tile * 16 + (lane & 15);
                const int load_column = tile_inner / (kSparse ? 2 : 1) + (lane >> 4) * 16;
                const int input_index = load_row * kInputStride + load_column;
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
                    shared_input + (kSparse
                        ? fp8_sparse_input_index(input_index) : fp8_shared_index(input_index))));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(input_fragments[row_tile][0]), "=r"(input_fragments[row_tile][1]),
                      "=r"(input_fragments[row_tile][2]), "=r"(input_fragments[row_tile][3])
                    : "r"(address));
                if constexpr (kSparse) {
                    // SM89 selector 0: alternating lanes own the upper eight
                    // rows; the other lane bit selects one 32-K metadata word.
                    const int meta_row = warp_row + row_tile * 16 + group + 8 * (thread_in_group & 1);
                    metadata[row_tile] = *reinterpret_cast<const uint32_t*>(
                        stage_values + kInputValues + kWeightValues + meta_row * 8 +
                        (thread_in_group >> 1) * 4);
                }
            }
#pragma unroll
            for (int column_tile = 0; column_tile < 8; ++column_tile) {
                const int load_row = warp_column + column_tile * 8 + (lane & 7);
                const int load_column = tile_inner + ((lane >> 3) & (kSparse ? 3 : 1)) * 16;
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
                    shared_weight + fp8_shared_index(load_row * kTileInner + load_column)));
                uint32_t weights[4];
                if constexpr (kSparse) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                        : "=r"(weights[0]), "=r"(weights[1]), "=r"(weights[2]), "=r"(weights[3])
                        : "r"(address));
                } else {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                        : "=r"(weights[0]), "=r"(weights[1]) : "r"(address));
                }
#pragma unroll
                for (int row_tile = 0; row_tile < 2; ++row_tile) {
                    Accumulator* accumulator = accumulators[row_tile][column_tile];
                    if constexpr (kInt4) {
                        mma_m16n8k64_int4(
                            accumulator[0], accumulator[1], accumulator[2], accumulator[3],
                            input_fragments[row_tile][0], input_fragments[row_tile][1],
                            input_fragments[row_tile][2], input_fragments[row_tile][3],
                            weights[0], weights[1]);
                    } else if constexpr (kSparse) {
                        mma_m16n8k64_sparse_fp8(
                            accumulator, input_fragments[row_tile], weights, metadata[row_tile]);
                    } else {
                        mma_m16n8k32_fp8(
                            accumulator[0], accumulator[1], accumulator[2], accumulator[3],
                            input_fragments[row_tile][0], input_fragments[row_tile][1],
                            input_fragments[row_tile][2], input_fragments[row_tile][3],
                            weights[0], weights[1]);
                    }
                }
            }
        }
    }

    if constexpr (kEpilogue == 5) {
        // All operand readers finish before reusing the arena for paired FP16
        // value/gate outputs. Keep both FP16 boundaries of projection -> GLU.
        __syncthreads();
        auto* expanded = reinterpret_cast<__half*>(shared_bytes);
#pragma unroll
        for (int row_tile = 0; row_tile < 2; ++row_tile) {
            const int local_row = warp_row + row_tile * 16;
            if (row + local_row >= rows) continue;
            const float scale0 = input_scales[row + local_row + group];
            const float scale1 = input_scales[row + local_row + group + 8];
#pragma unroll
            for (int column_tile = 0; column_tile < 8; ++column_tile) {
                const int local_column = warp_column + column_tile * 8 + thread_in_group * 2;
                const int weight_column = column + local_column % 64 + (local_column / 64) * 1024;
                const float weight0 = weight_scales[weight_column];
                const float weight1 = weight_scales[weight_column + 1];
                const float* a = accumulators[row_tile][column_tile];
                *reinterpret_cast<__half2*>(expanded + (local_row + group) * 128 + local_column) =
                    __floats2half2_rn(a[0] * scale0 * weight0, a[1] * scale0 * weight1);
                *reinterpret_cast<__half2*>(expanded + (local_row + group + 8) * 128 + local_column) =
                    __floats2half2_rn(a[2] * scale1 * weight0, a[3] * scale1 * weight1);
            }
        }
        __syncthreads();
        for (int index = threadIdx.x; index < 128 * 64; index += 256) {
            const int local_row = index / 64;
            const int feature = index % 64;
            if (row + local_row < rows) {
                const float value = __half2float(expanded[local_row * 128 + feature]);
                const float gate = __half2float(expanded[local_row * 128 + feature + 64]);
                output[(row + local_row) * 1024 + column + feature] = row + local_row < valid_rows
                    ? __float2half_rn(value / (1.0f + expf(-gate))) : __float2half_rn(0.0f);
            }
        }
        return;
    }

    if constexpr (kEpilogue == 9) {
        // Fused Q/K/V projection. Each 128-column tile is exactly one head of
        // one projection: blockIdx.x 0-7 query, 8-15 key, 16-23 value. Query
        // keeps its FP16 row-major layout. Key and value tiles are FP16-rounded
        // into shared storage first; a 136-half row pitch spreads the eight
        // fragment rows of every store across distinct banks.
        const int projection = static_cast<int>(blockIdx.x) >> 3;
        const int head = static_cast<int>(blockIdx.x) & 7;
        constexpr int kStagePitch = 136;
        auto* staged = reinterpret_cast<__half*>(shared_bytes);
        if (projection != 0) {
            // All cp.async and ldmatrix readers finish before the arena is reused.
            __syncthreads();
        }
#pragma unroll
        for (int row_tile = 0; row_tile < 2; ++row_tile) {
            const int local_row = warp_row + row_tile * 16;
            const int output_row = row + local_row;
            if (output_row >= rows) continue;
            const float row_scale0 = input_scales[output_row + group];
            const float row_scale1 = input_scales[output_row + group + 8];
#pragma unroll
            for (int column_tile = 0; column_tile < 8; ++column_tile) {
                const int local_column = warp_column + column_tile * 8 + thread_in_group * 2;
                const int weight_column = column + local_column;
                const float weight_scale0 = weight_scales[weight_column];
                const float weight_scale1 = weight_scales[weight_column + 1];
                const float* a = accumulators[row_tile][column_tile];
                const __half2 rounded0 = __floats2half2_rn(
                    a[0] * row_scale0 * weight_scale0, a[1] * row_scale0 * weight_scale1);
                const __half2 rounded1 = __floats2half2_rn(
                    a[2] * row_scale1 * weight_scale0, a[3] * row_scale1 * weight_scale1);
                if (projection == 0) {
                    const int feature = head * kHeadWidth + local_column;
                    *reinterpret_cast<__half2*>(
                        output + (output_row + group) * kModelWidth + feature) = rounded0;
                    *reinterpret_cast<__half2*>(
                        output + (output_row + group + 8) * kModelWidth + feature) = rounded1;
                } else {
                    *reinterpret_cast<__half2*>(
                        staged + (local_row + group) * kStagePitch + local_column) = rounded0;
                    *reinterpret_cast<__half2*>(
                        staged + (local_row + group + 8) * kStagePitch + local_column) = rounded1;
                }
            }
        }
        if (projection == 0) return;
        __syncthreads();
        if (projection == 1) {
            // Signed INT4 key pack, identical arithmetic and byte layout to
            // pk_sm89_pack_key_int4: two lanes share one row, one row max per
            // (row, head) vector, scale max/7, nibbles in K64 MMA operand order.
            const int local_row = static_cast<int>(threadIdx.x) >> 1;
            const int half = static_cast<int>(threadIdx.x) & 1;
            const __half2* source =
                reinterpret_cast<const __half2*>(staged + local_row * kStagePitch + half * 64);
            float maximum = 0.0f;
            // Rows past the end of a partial tile were never staged; both lanes
            // of such a row keep zero and the shuffle below stays warp-uniform.
            if (row + local_row < rows) {
#pragma unroll
                for (int pair = 0; pair < 32; ++pair) {
                    const float2 v = __half22float2(source[pair]);
                    maximum = fmaxf(maximum, fmaxf(fabsf(v.x), fabsf(v.y)));
                }
            }
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
            const float scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
            if (row + local_row < rows) {
                const int vector = (row + local_row) * kHeads + head;
                if (half == 0) key_scales[vector] = scale;
                uint32_t words[8];
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    uint32_t low = 0, high = 0;
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 lo = __half22float2(source[quad * 4 + pair]);
                        const float2 hi = __half22float2(source[16 + quad * 4 + pair]);
                        low |= (uint32_t(__float2int_rn(lo.x / scale)) & 15u) << (8 * pair);
                        low |= (uint32_t(__float2int_rn(lo.y / scale)) & 15u) << (8 * pair + 4);
                        high |= (uint32_t(__float2int_rn(hi.x / scale)) & 15u) << (8 * pair);
                        high |= (uint32_t(__float2int_rn(hi.y / scale)) & 15u) << (8 * pair + 4);
                    }
                    words[quad * 2] = low;
                    words[quad * 2 + 1] = high;
                }
                auto* destination = reinterpret_cast<uint4*>(packed_key + vector * 64 + half * 32);
                destination[0] = make_uint4(words[0], words[1], words[2], words[3]);
                destination[1] = make_uint4(words[4], words[5], words[6], words[7]);
            }
            return;
        }
        // Feature-major FP16 value rows, the layout the value quantizer and
        // scalar boundary tiles already consume. Lanes own rows l, l+32, ...
        // so each warp store instruction covers 64 contiguous bytes.
        const int tile_rows = min(kTileRows, rows - row);
#pragma unroll 4
        for (int item = 0; item < 16; ++item) {
            const int local_column = warp * 16 + item;
            __half* destination = value_t + size_t(head * kHeadWidth + local_column) * rows + row;
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int local_row = lane + part * 32;
                if (local_row < tile_rows) {
                    destination[local_row] = staged[local_row * kStagePitch + local_column];
                }
            }
        }
        return;
    }

    if constexpr (kEpilogue == 1 || kEpilogue == 7) {
        // All MMA operand readers finish before the packed epilogue reuses
        // their arena. A 144-byte row keeps the later vector loads aligned
        // while offsetting neighboring rows across shared-memory banks.
        __syncthreads();
    }
#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
        const int output_row = row + warp_row + row_tile * 16;
        if (output_row >= rows) {
            continue;
        }
        const float row_scale0 =
            dynamic_scale != 0 ? input_scales[output_row + group] : input_scale;
        const float row_scale1 =
            dynamic_scale != 0 ? input_scales[output_row + group + 8] : input_scale;
#pragma unroll
        for (int column_tile = 0; column_tile < 8; ++column_tile) {
            const int output_column =
                column + warp_column + column_tile * 8 + thread_in_group * 2;
            const float weight_scale0 = weight_scales[output_column];
            const float weight_scale1 = weight_scales[output_column + 1];
            const Accumulator* accumulator = accumulators[row_tile][column_tile];
            const float value0 = accumulator[0] * row_scale0 * weight_scale0;
            const float value1 = accumulator[1] * row_scale0 * weight_scale1;
            const float value2 = accumulator[2] * row_scale1 * weight_scale0;
            const float value3 = accumulator[3] * row_scale1 * weight_scale1;
            const int offset0 = (output_row + group) * kOutputWidth + output_column;
            const int offset1 = (output_row + group + 8) * kOutputWidth + output_column;
            if constexpr (kEpilogue == 3 || kEpilogue == 4) {
                float biased0 = value0 + bias[output_column];
                float biased1 = value1 + bias[output_column + 1];
                float biased2 = value2 + bias[output_column];
                float biased3 = value3 + bias[output_column + 1];
                if constexpr (kEpilogue == 4) {
                    biased0 = fmaxf(biased0, 0.0f);
                    biased1 = fmaxf(biased1, 0.0f);
                    biased2 = fmaxf(biased2, 0.0f);
                    biased3 = fmaxf(biased3, 0.0f);
                }
                *reinterpret_cast<__half2*>(output + offset0) = __floats2half2_rn(
                    biased0, biased1);
                *reinterpret_cast<__half2*>(output + offset1) = __floats2half2_rn(
                    biased2, biased3);
            } else if constexpr (kEpilogue == 2 || kEpilogue == 6 || kSparse) {
                *reinterpret_cast<__half2*>(output + offset0) = __floats2half2_rn(
                    fmaf(kEpilogue == 6 ? 1.0f : 0.5f, value0, __half2float(output[offset0])),
                    fmaf(kEpilogue == 6 ? 1.0f : 0.5f, value1, __half2float(output[offset0 + 1])));
                *reinterpret_cast<__half2*>(output + offset1) = __floats2half2_rn(
                    fmaf(kEpilogue == 6 ? 1.0f : 0.5f, value2, __half2float(output[offset1])),
                    fmaf(kEpilogue == 6 ? 1.0f : 0.5f, value3, __half2float(output[offset1 + 1])));
            } else {
                const __half2 rounded0 = __floats2half2_rn(
                    value0 / (1.0f + expf(-value0)),
                    value1 / (1.0f + expf(-value1)));
                const __half2 rounded1 = __floats2half2_rn(
                    value2 / (1.0f + expf(-value2)),
                    value3 / (1.0f + expf(-value3)));
                if constexpr (kEpilogue == 1 || kEpilogue == 7) {
                    // Preserve the original FP16 boundary before fixed E4M3 quantization.
                    const float2 a = __half22float2(rounded0);
                    const float2 b = __half22float2(rounded1);
                    const int shared_offset0 =
                        (output_row + group - row) * 144 + output_column - column;
                    const int shared_offset1 =
                        (output_row + group + 8 - row) * 144 + output_column - column;
                    *reinterpret_cast<uint16_t*>(shared_bytes + shared_offset0) =
                        uint16_t(__nv_fp8_e4m3(a.x * 16.0f).__x) |
                        (uint16_t(__nv_fp8_e4m3(a.y * 16.0f).__x) << 8);
                    *reinterpret_cast<uint16_t*>(shared_bytes + shared_offset1) =
                        uint16_t(__nv_fp8_e4m3(b.x * 16.0f).__x) |
                        (uint16_t(__nv_fp8_e4m3(b.y * 16.0f).__x) << 8);
                } else {
                    *reinterpret_cast<__half2*>(output + offset0) = rounded0;
                    *reinterpret_cast<__half2*>(output + offset1) = rounded1;
                }
            }
        }
    }
    if constexpr (kEpilogue == 1 || kEpilogue == 7) {
        __syncthreads();
        auto* bytes = reinterpret_cast<uint8_t*>(output);
        for (int index = threadIdx.x * 16; index < 128 * 128; index += 256 * 16) {
            const int local_row = index / 128;
            const int local_column = index % 128;
            if (row + local_row < rows) {
                const uint4 values = *reinterpret_cast<const uint4*>(
                    shared_bytes + local_row * 144 + local_column);
                if constexpr (kEpilogue == 7) {
                    // Top two E4M3 magnitudes per quartet, ties by lower index.
                    // The existing activation arena holds both data and metadata.
                    const uint32_t words[4] = {values.x, values.y, values.z, values.w};
                    uint64_t compressed = 0;
                    uint16_t metadata = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t packed = compress_fp8_quartet(words[i]);
                        compressed |= uint64_t(packed & 65535) << (16 * i);
                        metadata |= (packed >> 16) << (4 * i);
                    }
                    *reinterpret_cast<uint64_t*>(bytes + size_t(row + local_row) * (kOutputWidth / 2) +
                        (column + local_column) / 2) = compressed;
                    *reinterpret_cast<uint16_t*>(bytes + size_t(rows) * (kOutputWidth / 2) +
                        size_t(row + local_row) * (kOutputWidth / 8) +
                        (column + local_column) / 8) = metadata;
                } else {
                    *reinterpret_cast<uint4*>(
                        bytes + (row + local_row) * kOutputWidth + column + local_column) = values;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_expand_fp8_async(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* output, int rows, float input_scale, int dynamic_scale) {
    ffn_fp8_async<1024, 4096, 0>(
        input, weight, weight_scales, input_scales, output, rows, input_scale, dynamic_scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_expand_fp8_packed(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* output, int rows, float input_scale, int dynamic_scale) {
    ffn_fp8_async<1024, 4096, 1, false, true>(
        input, weight, weight_scales, input_scales, output, rows, input_scale, dynamic_scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_expand_int4_sparse(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* output, int rows, float input_scale, int dynamic_scale) {
    ffn_fp8_async<512, 4096, 7, true>(
        input, weight, weight_scales, input_scales, output, rows, input_scale, dynamic_scale);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_contract_fp8_sparse(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    __half* output, int rows, int inner_width) {
    // The host supplies 4096. A runtime loop bound prevents ptxas from fully
    // unrolling the sparse mainloop into a spilling 1024-instruction MMA body.
    ffn_fp8_async<4096, 1024, 8, false, true>(input, weight, weight_scales, nullptr, output,
        rows, 1.0f / 16.0f, 0, nullptr, 0, inner_width);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_ffn_contract_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    __half* output, int rows) {
    ffn_fp8_async<4096, 1024, 2>(
        input, weight, weight_scales, nullptr, output, rows, 1.0f / 16.0f, 0);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_subsample_projection_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, const float* bias, __half* output, int rows) {
    ffn_fp8_async<4096, 1024, 3>(
        input, weight, weight_scales, input_scales, output, rows, 1.0f, 1, bias);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_subsample_pointwise_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, const float* bias, __half* output, int rows) {
    ffn_fp8_async<256, 256, 4>(
        input, weight, weight_scales, input_scales, output, rows, 1.0f, 1, bias);
}

// One launch projects query, key and value from the FP8 attention input.
// Dynamic shared memory is 128 x 136 halves so the staged tile has bank slack.
extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_qkv_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* query, uint8_t* packed_key, float* key_scales,
    __half* value_t, int rows) {
    ffn_fp8_async<1024, 3072, 9>(
        input, weight, weight_scales, input_scales, query, rows, 1.0f, 1, nullptr, 0, 1024,
        packed_key, key_scales, value_t);
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_conv_glu_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* output, int rows, int valid_rows) {
    ffn_fp8_async<1024, 2048, 5>(
        input, weight, weight_scales, input_scales, output, rows, 1.0f, 1, nullptr, valid_rows);
}

extern "C" __global__
void pk_sm89_quantize_rows1024(const __half* input, uint8_t* output, float* scales) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float values[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = __half2float(input[row * 1024 + lane + i * 256]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    __shared__ float scratch[256];
    maximum = block_max_tree_tail(maximum, scratch);
    const float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
    if (lane == 0) scales[row] = scale;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        output[row * 1024 + lane + i * 256] = __nv_fp8_e4m3(values[i] / scale).__x;
    }
}

extern "C" __global__
void pk_sm89_quantize_rows256(const __half* input, uint8_t* output, float* scales) {
    const int index = blockIdx.x * 256 + threadIdx.x;
    const float value = __half2float(input[index]);
    __shared__ float scratch[256];
    const float maximum = block_max_tree_tail(fabsf(value), scratch);
    const float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
    if (threadIdx.x == 0) scales[blockIdx.x] = scale;
    output[index] = __nv_fp8_e4m3(value / scale).__x;
}

extern "C" __global__
void pk_sm89_quantize_ffn_contract(const __half* input, uint8_t* output, float* scales) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = __half2float(input[row * 4096 + lane + i * 256]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    __shared__ float scratch[256];
    maximum = block_max_tree_tail(maximum, scratch);
    const float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
    if (lane == 0) scales[row] = scale;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        output[row * 4096 + lane + i * 256] = __nv_fp8_e4m3(values[i] / scale).__x;
    }
}

extern "C" __global__
void pk_sm89_depthwise_batchnorm_silu(
    const __half* __restrict__ input,
    const __half* __restrict__ depthwise_weight,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_variance,
    __half* __restrict__ output,
    int rows,
    int padded_rows,
    float epsilon) {
    const int channel = static_cast<int>((blockIdx.x % 4) * blockDim.x + threadIdx.x);
    const int start = static_cast<int>(blockIdx.x / 4) * 8;
    // One thread owns an eight-row strip, reusing its halo and channel weights.
    float samples[16];
    float weights[9];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int row = start + i - 4;
        samples[i] = row >= 0 && row < rows
            ? __half2float(input[row * kModelWidth + channel]) : 0.0f;
    }
    #pragma unroll
    for (int kernel = 0; kernel < 9; ++kernel) {
        weights[kernel] = __half2float(depthwise_weight[channel * 9 + kernel]);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = start + i;
        if (row >= padded_rows) {
            continue;
        }
        const int index = row * kModelWidth + channel;
        if (row >= rows) {
            output[index] = __float2half_rn(0.0f);
            continue;
        }
        float value = 0.0f;
        #pragma unroll
        for (int kernel = 0; kernel < 9; ++kernel) {
            const int input_row = row + kernel - 4;
            if (input_row >= 0 && input_row < rows) {
                value = fmaf(samples[i + kernel], weights[kernel], value);
            }
        }
        value = (value - running_mean[channel]) *
            rsqrtf(running_variance[channel] + epsilon) * norm_weight[channel] +
            norm_bias[channel];
        output[index] = __float2half_rn(value / (1.0f + expf(-value)));
    }
}

extern "C" __global__
void pk_sm89_depthwise_pack(
    const __half* __restrict__ input,
    const __half* __restrict__ depthwise_weight,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_variance,
    uint8_t* __restrict__ output,
    float* scales,
    int rows,
    int padded_rows,
    float epsilon) {
    const int start = static_cast<int>(gridDim.x - 1 - blockIdx.x) * 8;
    float values[4][8] = {};
    // Own complete rows so packing can consume the rounded depthwise values
    // without materializing FP16 or changing the channel's FMA/BN/SiLU order.
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int channel = threadIdx.x + c * 256;
        float samples[16];
        float weights[9];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row = start + i - 4;
            samples[i] = row >= 0 && row < rows
                ? __half2float(input[row * kModelWidth + channel]) : 0.0f;
        }
#pragma unroll
        for (int kernel = 0; kernel < 9; ++kernel) {
            weights[kernel] = __half2float(depthwise_weight[channel * 9 + kernel]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = start + i;
            if (row >= padded_rows || row >= rows) continue;
            float value = 0.0f;
#pragma unroll
            for (int kernel = 0; kernel < 9; ++kernel) {
                const int input_row = row + kernel - 4;
                if (input_row >= 0 && input_row < rows) {
                    value = fmaf(samples[i + kernel], weights[kernel], value);
                }
            }
            value = (value - running_mean[channel]) *
                rsqrtf(running_variance[channel] + epsilon) * norm_weight[channel] +
                norm_bias[channel];
            values[c][i] = __half2float(__float2half_rn(value / (1.0f + expf(-value))));
        }
    }
    __shared__ float maxima[8][8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float maximum = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) maximum = fmaxf(maximum, fabsf(values[c][i]));
        maximum = warp_max(maximum);
        if (lane == 0) maxima[i][warp] = maximum;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = start + i;
        if (row >= padded_rows) continue;
        float maximum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) maximum = fmaxf(maximum, maxima[i][w]);
        const float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
        if (threadIdx.x == 0) scales[row] = scale;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            output[row * 1024 + threadIdx.x + c * 256] = __nv_fp8_e4m3(values[c][i] / scale).__x;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_conv_residual_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    const float* input_scales, __half* output, int rows) {
    ffn_fp8_async<1024, 1024, 6>(
        input, weight, weight_scales, input_scales, output, rows, 1.0f, 1);
}

extern "C" __global__
void pk_sm89_pack_position_heads(
    const __half* __restrict__ input,
    __half* __restrict__ output) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = kPositionRowsPadded * kModelWidth;
    if (index >= total) {
        return;
    }
    const int position = index / kModelWidth;
    const int feature = index - position * kModelWidth;
    const int head = feature / kHeadWidth;
    const int dimension = feature - head * kHeadWidth;
    output[(head * kPositionRowsPadded + position) * kHeadWidth + dimension] =
        input[index];
}

extern "C" __global__ __launch_bounds__(32, 8)
void pk_sm89_local_relpos_attention(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const __half* __restrict__ position,
    const __half* __restrict__ bias_u,
    const __half* __restrict__ bias_v,
    __half* __restrict__ output,
    int rows,
    int padded_rows,
    int valid_rows,
    int attention_left,
    int attention_right,
    const int32_t* __restrict__ bounds) {
    const int query_row = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & 31;
    const int half = lane >> 4;
    const int half_lane = lane & 15;
    if (query_row >= padded_rows || head >= kHeads) {
        return;
    }
    const int first_valid = bounds ? bounds[2 * query_row] : 0;
    const int end_valid = bounds ? bounds[2 * query_row + 1] : valid_rows;
    if (query_row >= rows || query_row >= end_valid) {
        for (int dimension = lane; dimension < kHeadWidth; dimension += 32) {
            output[query_row * kModelWidth + head * kHeadWidth + dimension] =
                __float2half_rn(0.0f);
        }
        return;
    }

    const int first_key = max(first_valid, query_row - attention_left);
    const int last_key = min(end_valid - 1, query_row + attention_right);
    const int key_count = last_key - first_key + 1;
    __shared__ float scores[kMaxAttention];

    const int query_head = query_row * kModelWidth + head * kHeadWidth;
    const int bias_head = head * kHeadWidth;
    float2 query_u[4];
    float2 query_v[4];
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        const int dimension = 2 * half_lane + quarter * 32;
        const float2 q = __half22float2(
            *reinterpret_cast<const __half2*>(query + query_head + dimension));
        const float2 u = __half22float2(
            *reinterpret_cast<const __half2*>(bias_u + bias_head + dimension));
        const float2 v = __half22float2(
            *reinterpret_cast<const __half2*>(bias_v + bias_head + dimension));
        query_u[quarter] = make_float2(q.x + u.x, q.y + u.y);
        query_v[quarter] = make_float2(q.x + v.x, q.y + v.y);
    }

    for (int slot_pair = 0; slot_pair < key_count; slot_pair += 2) {
        const int slot = slot_pair + half;
        const int load_slot = min(slot, key_count - 1);
        const int key_row = first_key + load_slot;
        const int position_row = attention_left - query_row + key_row;
        float score = 0.0f;
#pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
            const int dimension = 2 * half_lane + quarter * 32;
            const int key_index =
                key_row * kModelWidth + head * kHeadWidth + dimension;
            const int position_index =
                (head * kPositionRowsPadded + position_row) * kHeadWidth + dimension;
            const float2 key_pair = __half22float2(
                *reinterpret_cast<const __half2*>(key + key_index));
            const float2 position_pair = __half22float2(
                *reinterpret_cast<const __half2*>(position + position_index));
            score = fmaf(query_u[quarter].x, key_pair.x, score);
            score = fmaf(query_v[quarter].x, position_pair.x, score);
            score = fmaf(query_u[quarter].y, key_pair.y, score);
            score = fmaf(query_v[quarter].y, position_pair.y, score);
        }
        score = half_warp_sum(score);
        if (half_lane == 0 && slot < key_count) {
            scores[slot] = score * 0.08838834764831845f;
        }
    }
    __syncwarp();

    float maximum = -FLT_MAX;
    for (int slot = lane; slot < key_count; slot += 32) {
        maximum = fmaxf(maximum, scores[slot]);
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffff, maximum, 0);

    float denominator = 0.0f;
    for (int slot = lane; slot < key_count; slot += 32) {
        scores[slot] = expf(scores[slot] - maximum);
        denominator += scores[slot];
    }
    denominator = warp_sum(denominator);
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    __syncwarp();

    const int dimension = 4 * lane;
    float context0 = 0.0f;
    float context1 = 0.0f;
    float context2 = 0.0f;
    float context3 = 0.0f;
    for (int slot = 0; slot < key_count; ++slot) {
        const int key_row = first_key + slot;
        const int value_index =
            key_row * kModelWidth + head * kHeadWidth + dimension;
        const float weight = scores[slot] / denominator;
        const float2 value01 = __half22float2(
            *reinterpret_cast<const __half2*>(value + value_index));
        const float2 value23 = __half22float2(
            *reinterpret_cast<const __half2*>(value + value_index + 2));
        context0 = fmaf(weight, value01.x, context0);
        context1 = fmaf(weight, value01.y, context1);
        context2 = fmaf(weight, value23.x, context2);
        context3 = fmaf(weight, value23.y, context3);
    }
    const int output_index =
        query_row * kModelWidth + head * kHeadWidth + dimension;
    *reinterpret_cast<__half2*>(output + output_index) = __halves2half2(
        __float2half_rn(context0), __float2half_rn(context1));
    *reinterpret_cast<__half2*>(output + output_index + 2) = __halves2half2(
        __float2half_rn(context2), __float2half_rn(context3));
}

template<bool kPackedOutput>
__device__ __forceinline__ void store_attention_pair(
    __half* output, int index, __half2 rounded) {
    if constexpr (kPackedOutput) {
        const float2 values = __half22float2(rounded);
        const uint16_t packed = uint16_t(__nv_fp8_e4m3(values.x * 28.0f).__x) |
            (uint16_t(__nv_fp8_e4m3(values.y * 28.0f).__x) << 8);
        *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(output) + index) = packed;
    } else {
        *reinterpret_cast<__half2*>(output + index) = rounded;
    }
}

// Long-form L4 specialization. Sixteen adjacent queries share the same
// 272-key union. Eight warps form content and relative-position scores on
// Tensor Cores and normalize in FP32. Packed output uses row-scaled U4
// probabilities and feature-scaled S4 values; the smaller path retains E4M3.
// Results are scaled and written from registers. Scalar boundary tiles always
// consume the original transposed E4M3 values.
__device__ __forceinline__ void mma_context_u4_s4(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_m16n8k32_int8(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Scalar boundary tiles of the packed kernel consume the same signed INT4 key
// vectors as the interior Tensor Core tiles; the FP16 key is never materialized.
__device__ __forceinline__ float2 unpack_key_pair(
    const uint8_t* __restrict__ packed_key, int vector, int dimension, float scale) {
    const int within = dimension & 63;
    const int byte = (dimension >> 6) * 32 + ((within & 31) >> 3) * 8 + (within >> 5) * 4 +
        ((within & 7) >> 1);
    const unsigned packed = packed_key[vector * 64 + byte];
    const int low = static_cast<int>((packed & 15u) ^ 8u) - 8;
    const int high = static_cast<int>(((packed >> 4) & 15u) ^ 8u) - 8;
    return make_float2(static_cast<float>(low) * scale, static_cast<float>(high) * scale);
}

template<bool kPackedOutput>
__device__ __forceinline__ void local_relpos_attention_tc_scores(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const uint8_t* __restrict__ value_fp8,
    const __half* __restrict__ position,
    const __half* __restrict__ bias_u,
    const __half* __restrict__ bias_v,
    __half* __restrict__ output,
    int rows,
    int padded_rows,
    int valid_rows,
    int attention_left,
    int attention_right,
    int value_stride,
    const uint8_t* packed_key = nullptr, const float* key_scales = nullptr,
    const uint8_t* packed_position = nullptr, const float* position_scales = nullptr) {
    constexpr int kQueryTile = 16;
    constexpr int kWarps = 8;
    const int query_base = static_cast<int>(kPackedOutput ? gridDim.x - 1 - blockIdx.x : blockIdx.x) * kQueryTile;
    const int head = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    if (query_base >= padded_rows || head >= kHeads) {
        return;
    }

    // Offset successive rows across shared banks without changing score arithmetic.
    __shared__ __align__(32) __half query_u[kQueryTile][kHeadWidth + 8];
    __shared__ __align__(32) __half query_v[kQueryTile][kHeadWidth + 8];
    union __align__(32) ContentStorage {
        float scores[kPackedOutput ? kWarps : kQueryTile][kPositionRowsPadded];
        __nv_bfloat16 packed_scores[kQueryTile][kPositionRowsPadded];
        float context[kQueryTile][kHeadWidth];
    };
    union __align__(32) PositionStorage {
        float scores[kPackedOutput ? 1 : kQueryTile][kPositionRowsPadded];
        __nv_bfloat16 packed_scores[kQueryTile][kPositionRowsPadded];
        uint8_t probabilities[kQueryTile][kPackedOutput ? 160 : kFp8ContextRowsPadded];
    };
    __shared__ ContentStorage content_storage;
    __shared__ PositionStorage position_storage;
    __shared__ float probability_scale[kQueryTile];

    const bool interior = query_base >= attention_left &&
        query_base + kQueryTile - 1 + attention_right < valid_rows &&
        query_base + kQueryTile <= rows;
    if (!interior) {
        for (int local_query = warp; local_query < kQueryTile; local_query += kWarps) {
            const int query_row = query_base + local_query;
            if (query_row >= rows || query_row >= valid_rows) {
                if (query_row < padded_rows) {
                    for (int dimension = lane; dimension < kHeadWidth; dimension += 32) {
                        const int index = query_row * kModelWidth + head * kHeadWidth + dimension;
                        if constexpr (kPackedOutput) {
                            reinterpret_cast<uint8_t*>(output)[index] = 0;
                        } else {
                            output[index] = __float2half_rn(0.0f);
                        }
                    }
                }
                continue;
            }

            const int first_key = max(0, query_row - attention_left);
            const int last_key = min(valid_rows - 1, query_row + attention_right);
            const int key_count = last_key - first_key + 1;
            float* scores = content_storage.scores[kPackedOutput ? warp : local_query];
            const int query_head = query_row * kModelWidth + head * kHeadWidth;
            const int bias_head = head * kHeadWidth;
            float2 scalar_query_u[4];
            float2 scalar_query_v[4];
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const int dimension = 2 * (lane & 15) + quarter * 32;
                const float2 q = __half22float2(
                    *reinterpret_cast<const __half2*>(query + query_head + dimension));
                const float2 u = __half22float2(
                    *reinterpret_cast<const __half2*>(bias_u + bias_head + dimension));
                const float2 v = __half22float2(
                    *reinterpret_cast<const __half2*>(bias_v + bias_head + dimension));
                scalar_query_u[quarter] = make_float2(q.x + u.x, q.y + u.y);
                scalar_query_v[quarter] = make_float2(q.x + v.x, q.y + v.y);
            }

            const int half = lane >> 4;
            const int half_lane = lane & 15;
            for (int slot_pair = 0; slot_pair < key_count; slot_pair += 2) {
                const int slot = slot_pair + half;
                const int load_slot = min(slot, key_count - 1);
                const int key_row = first_key + load_slot;
                const int position_row = attention_left - query_row + key_row;
                const int key_vector = key_row * kHeads + head;
                const float key_scale = kPackedOutput ? key_scales[key_vector] : 0.0f;
                float score = 0.0f;
#pragma unroll
                for (int quarter = 0; quarter < 4; ++quarter) {
                    const int dimension = 2 * half_lane + quarter * 32;
                    const int key_index =
                        key_row * kModelWidth + head * kHeadWidth + dimension;
                    const int position_index =
                        (head * kPositionRowsPadded + position_row) * kHeadWidth + dimension;
                    const float2 key_pair = kPackedOutput
                        ? unpack_key_pair(packed_key, key_vector, dimension, key_scale)
                        : __half22float2(*reinterpret_cast<const __half2*>(key + key_index));
                    const float2 position_pair = __half22float2(
                        *reinterpret_cast<const __half2*>(position + position_index));
                    score = fmaf(scalar_query_u[quarter].x, key_pair.x, score);
                    score = fmaf(scalar_query_v[quarter].x, position_pair.x, score);
                    score = fmaf(scalar_query_u[quarter].y, key_pair.y, score);
                    score = fmaf(scalar_query_v[quarter].y, position_pair.y, score);
                }
                score = half_warp_sum(score);
                if (half_lane == 0 && slot < key_count) {
                    scores[slot] = score * 0.08838834764831845f;
                }
            }
            __syncwarp();

            float maximum = -FLT_MAX;
            for (int slot = lane; slot < key_count; slot += 32) {
                maximum = fmaxf(maximum, scores[slot]);
            }
            maximum = warp_max(maximum);
            maximum = __shfl_sync(0xffffffff, maximum, 0);

            float denominator = 0.0f;
            for (int slot = lane; slot < key_count; slot += 32) {
                scores[slot] = expf(scores[slot] - maximum);
                denominator += scores[slot];
            }
            denominator = warp_sum(denominator);
            denominator = __shfl_sync(0xffffffff, denominator, 0);
            __syncwarp();

            const int dimension = 4 * lane;
            float context0 = 0.0f;
            float context1 = 0.0f;
            float context2 = 0.0f;
            float context3 = 0.0f;
            const int value_feature = head * kHeadWidth + dimension;
            constexpr float kValueScale = 8.0f / 448.0f;
            // The feature-major stride is padded to 16 rows. Fetch four aligned
            // row bytes once, but consume only in-window keys in original order.
            for (int chunk = first_key & ~3; chunk <= last_key; chunk += 4) {
                const uint32_t words0 = *reinterpret_cast<const uint32_t*>(
                    value_fp8 + value_feature * value_stride + chunk);
                const uint32_t words1 = *reinterpret_cast<const uint32_t*>(
                    value_fp8 + (value_feature + 1) * value_stride + chunk);
                const uint32_t words2 = *reinterpret_cast<const uint32_t*>(
                    value_fp8 + (value_feature + 2) * value_stride + chunk);
                const uint32_t words3 = *reinterpret_cast<const uint32_t*>(
                    value_fp8 + (value_feature + 3) * value_stride + chunk);
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const int key_row = chunk + item;
                    if (key_row < first_key || key_row > last_key) continue;
                    const float weight = scores[key_row - first_key] / denominator;
                    const float value0 =
                        fp8_e4m3_to_float(uint8_t(words0 >> (item * 8))) * kValueScale;
                    const float value1 =
                        fp8_e4m3_to_float(uint8_t(words1 >> (item * 8))) * kValueScale;
                    const float value2 =
                        fp8_e4m3_to_float(uint8_t(words2 >> (item * 8))) * kValueScale;
                    const float value3 =
                        fp8_e4m3_to_float(uint8_t(words3 >> (item * 8))) * kValueScale;
                    context0 = fmaf(weight, value0, context0);
                    context1 = fmaf(weight, value1, context1);
                    context2 = fmaf(weight, value2, context2);
                    context3 = fmaf(weight, value3, context3);
                }
            }
            const int output_index =
                query_row * kModelWidth + head * kHeadWidth + dimension;
            store_attention_pair<kPackedOutput>(output, output_index, __halves2half2(
                __float2half_rn(context0), __float2half_rn(context1)));
            store_attention_pair<kPackedOutput>(output, output_index + 2, __halves2half2(
                __float2half_rn(context2), __float2half_rn(context3)));
            // The next query reuses this warp's score row. All lanes must
            // finish their scalar context reads before any lane overwrites it.
            if constexpr (kPackedOutput) __syncwarp();
        }
        return;
    }

    const int first_key = query_base - attention_left;
    if constexpr (kPackedOutput) {
        // Pair MMA operand words for 64-bit loads. A 160-byte row rotates
        // each four-lane group onto disjoint banks within a half-warp load.
        __shared__ __align__(32) uint8_t q8[2][kQueryTile][160];
        __shared__ float query_scales[2][kQueryTile];
        for (int row = warp; row < kQueryTile; row += kWarps) {
            float values[2][4];
            float maximum[2] = {};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int dim = 4 * lane + i;
                const float q = __half2float(query[(query_base + row) * kModelWidth + head * kHeadWidth + dim]);
                values[0][i] = __half2float(__float2half_rn(q + __half2float(bias_u[head * kHeadWidth + dim])));
                values[1][i] = __half2float(__float2half_rn(q + __half2float(bias_v[head * kHeadWidth + dim])));
                maximum[0] = fmaxf(maximum[0], fabsf(values[0][i]));
                maximum[1] = fmaxf(maximum[1], fabsf(values[1][i]));
            }
#pragma unroll
            for (int plane = 0; plane < 2; ++plane) {
                float m = warp_max(maximum[plane]);
                m = __shfl_sync(0xffffffff, m, 0);
                const float scale = m > 0.0f ? m / (plane == 0 ? 7.0f : 127.0f) : 1.0f;
                if (lane == 0) query_scales[plane][row] = scale;
                uint32_t word = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    word |= (uint32_t(__float2int_rn(values[plane][i] / scale)) &
                        (plane == 0 ? 15u : 255u)) << ((plane == 0 ? 4 : 8) * i);
                if (plane == 0) {
                    // Two neighboring lanes pack eight signed nibbles per MMA word.
                    *reinterpret_cast<uint16_t*>(&q8[plane][row][(lane / 16) * 32 +
                        ((lane / 2) % 4) * 8 + ((lane / 2) & 4) + (lane & 1) * 2]) = word;
                } else {
                    *reinterpret_cast<uint32_t*>(&q8[plane][row][(lane / 8) * 32 +
                        (lane % 4) * 8 + (lane & 4)]) = word;
                }
            }
        }
        __syncthreads();
        for (int tile = warp; tile < kPositionRowsPadded / 16; tile += kWarps) {
#pragma unroll
            for (int plane = 0; plane < 2; ++plane) {
                // 128 * 127^2 < 2^21: integer accumulation and FP32 conversion
                // are exact; approximation is confined to quantization.
                int32_t acc[2][4] = {};
                const uint8_t* qa = &q8[plane][0][0];
                const uint8_t* kb = plane == 0 ? packed_key : packed_position;
                for (int inner = 0; inner < kHeadWidth; inner += (plane == 0 ? 64 : 32)) {
                    const int offset = group * 160 + (plane == 0 ? inner / 2 : inner) +
                        8 * thread_in_group;
                    const uint2 a02 = *reinterpret_cast<const uint2*>(qa + offset);
                    const uint2 a13 = *reinterpret_cast<const uint2*>(qa + offset + 8 * 160);
#pragma unroll
                    for (int part = 0; part < 2; ++part) {
                        const int col = tile * 16 + part * 8 + group;
                        const int vector = plane == 0 ? (first_key + col) * kHeads + head : head * kPositionRowsPadded + col;
                        const uint8_t* b = kb + vector * (plane == 0 ? kHeadWidth / 2 : kHeadWidth) +
                            (plane == 0 ? inner / 2 : inner) + 8 * thread_in_group;
                        const uint2 words = *reinterpret_cast<const uint2*>(b);
                        if (plane == 0) {
                            mma_m16n8k64_int4(acc[part][0], acc[part][1], acc[part][2], acc[part][3],
                                a02.x, a13.x, a02.y, a13.y, words.x, words.y);
                        } else {
                            mma_m16n8k32_int8(acc[part][0], acc[part][1], acc[part][2], acc[part][3],
                                a02.x, a13.x, a02.y, a13.y, words.x, words.y);
                        }
                    }
                }
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = group + (i / 2) * 8;
                        const int col = tile * 16 + part * 8 + thread_in_group * 2 + i % 2;
                        const int vector = plane == 0 ? (first_key + col) * kHeads + head : head * kPositionRowsPadded + col;
                        const float scale = query_scales[plane][row] * (plane == 0 ? key_scales[vector] : position_scales[vector]);
                        if (plane == 0) content_storage.packed_scores[row][col] = __float2bfloat16_rn(acc[part][i] * scale);
                        else position_storage.packed_scores[row][col] = __float2bfloat16_rn(acc[part][i] * scale);
                    }
                }
            }
        }
    } else {
    for (int index = static_cast<int>(threadIdx.x);
         index < kQueryTile * kHeadWidth;
         index += static_cast<int>(blockDim.x)) {
        const int local_query = index / kHeadWidth;
        const int dimension = index - local_query * kHeadWidth;
        const int query_index =
            (query_base + local_query) * kModelWidth + head * kHeadWidth + dimension;
        const int bias_index = head * kHeadWidth + dimension;
        const float query_value = __half2float(query[query_index]);
        query_u[local_query][dimension] = __float2half_rn(
            query_value + __half2float(bias_u[bias_index]));
        query_v[local_query][dimension] = __float2half_rn(
            query_value + __half2float(bias_v[bias_index]));
    }
    __syncthreads();

    for (int tile = warp; tile < kPositionRowsPadded / 16; tile += kWarps) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;

        wmma::fill_fragment(accumulator, 0.0f);
        for (int inner = 0; inner < kHeadWidth; inner += 16) {
            wmma::load_matrix_sync(a, &query_u[0][inner], kHeadWidth + 8);
            wmma::load_matrix_sync(
                b,
                key + (first_key + 16 * tile) * kModelWidth +
                    head * kHeadWidth + inner,
                kModelWidth);
            wmma::mma_sync(accumulator, a, b, accumulator);
        }
        wmma::store_matrix_sync(
            &content_storage.scores[0][16 * tile],
            accumulator,
            kPositionRowsPadded,
            wmma::mem_row_major);

        wmma::fill_fragment(accumulator, 0.0f);
        for (int inner = 0; inner < kHeadWidth; inner += 16) {
            wmma::load_matrix_sync(a, &query_v[0][inner], kHeadWidth + 8);
            wmma::load_matrix_sync(
                b,
                position + (head * kPositionRowsPadded + 16 * tile) * kHeadWidth + inner,
                kHeadWidth);
            wmma::mma_sync(accumulator, a, b, accumulator);
        }
        wmma::store_matrix_sync(
            &position_storage.scores[0][16 * tile],
            accumulator,
            kPositionRowsPadded,
            wmma::mem_row_major);
    }
    }
    __syncthreads();

    if constexpr (kPackedOutput) {
        // Compact score planes permit three L4 CTAs/SM. BF16 retains the
        // FP32 exponent range; all subsequent softmax arithmetic stays FP32.
        float logits[2][9];
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int local_query = warp + item * kWarps;
#pragma unroll
            for (int part = 0; part < 9; ++part) {
                const int slot = lane + part * 32;
                logits[item][part] = slot < kMaxAttention
                    ? (__bfloat162float(content_storage.packed_scores[local_query][local_query + slot]) +
                       __bfloat162float(position_storage.packed_scores[local_query][slot])) * 0.08838834764831845f
                    : -FLT_MAX;
            }
        }
        // All position-score reads must finish before probability writes
        // reuse the union. Retain the final context-consumption barrier too.
        __syncthreads();
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int local_query = warp + item * kWarps;
            float maximum = -FLT_MAX;
#pragma unroll
            for (int part = 0; part < 9; ++part) {
                if (lane + part * 32 < kMaxAttention)
                    maximum = fmaxf(maximum, logits[item][part]);
            }
            maximum = warp_max(maximum);
            maximum = __shfl_sync(0xffffffff, maximum, 0);
            float denominator = 0.0f;
#pragma unroll
            for (int part = 0; part < 9; ++part) {
                if (lane + part * 32 < kMaxAttention) {
                    logits[item][part] = expf(logits[item][part] - maximum);
                    denominator += logits[item][part];
                }
            }
            denominator = warp_sum(denominator);
            denominator = __shfl_sync(0xffffffff, denominator, 0);
            if (lane == 0) probability_scale[local_query] = 1.0f / (15.0f * denominator);
#pragma unroll
            for (int part = 0; part < 9; ++part) {
                const int slot = lane + part * 32;
                const uint32_t q = slot < kMaxAttention ? __float2int_rn(logits[item][part] * 15.0f) : 0;
                const uint32_t next_group = part < 8
                    ? __shfl_sync(0xffffffff, __float2int_rn(logits[item][part + 1] * 15.0f), 0) : 0;
                const uint32_t next_lane = __shfl_down_sync(0xffffffff, q, 1);
                const uint32_t next = lane == 31 ? next_group : next_lane;
                const int physical = local_query + slot;
                // One lane writes each complete byte, including cross-part
                // pairs and the odd-row leading half-byte. No RMW races.
                if (slot < kMaxAttention && (physical & 1) == 0)
                    position_storage.probabilities[local_query][physical / 2] = q | (next << 4);
                if (part == 0 && lane == 0 && (local_query & 1))
                    position_storage.probabilities[local_query][local_query / 2] = q << 4;
            }
            // Five K64 MMAs consume 320 nibbles; the tail beyond row271 is zero.
            for (int byte = lane; byte < 160; byte += 32) {
                if (2 * byte + 1 < local_query || 2 * byte >= local_query + kMaxAttention)
                    position_storage.probabilities[local_query][byte] = 0;
            }
        }
        __syncthreads();
    } else {
    for (int local_query = warp; local_query < kQueryTile; local_query += kWarps) {
        float* scores = content_storage.scores[local_query];
        for (int slot = lane; slot < kMaxAttention; slot += 32) {
            const int union_slot = local_query + slot;
            scores[union_slot] =
                (scores[union_slot] + position_storage.scores[local_query][slot]) *
                0.08838834764831845f;
        }
    }
    __syncthreads();

    for (int local_query = warp; local_query < kQueryTile; local_query += kWarps) {
        float* scores = content_storage.scores[local_query];
        float maximum = -FLT_MAX;
        for (int slot = lane; slot < kMaxAttention; slot += 32) {
            maximum = fmaxf(maximum, scores[local_query + slot]);
        }
        maximum = warp_max(maximum);
        maximum = __shfl_sync(0xffffffff, maximum, 0);

        float denominator = 0.0f;
        for (int slot = lane; slot < kMaxAttention; slot += 32) {
            const int union_slot = local_query + slot;
            scores[union_slot] = expf(scores[union_slot] - maximum);
            denominator += scores[union_slot];
        }
        denominator = warp_sum(denominator);
        denominator = __shfl_sync(0xffffffff, denominator, 0);
        __syncwarp();

        for (int union_slot = lane; union_slot < kFp8ContextRowsPadded; union_slot += 32) {
            const bool in_window = union_slot >= local_query &&
                union_slot < local_query + kMaxAttention;
            const float probability = in_window
                ? scores[union_slot] / denominator
                : 0.0f;
            position_storage.probabilities[local_query][union_slot] =
                __nv_fp8_e4m3(probability * 448.0f).__x;
        }
    }
    __syncthreads();

    }
    if constexpr (kPackedOutput) {
        int32_t context[2][4] = {};
        for (int inner = 0; inner < 320; inner += 64) {
            const uint8_t* probabilities = &position_storage.probabilities[group][inner / 2];
            const int offset = thread_in_group * 4;
            const uint32_t a0 = load_shared_u32(probabilities + offset);
            const uint32_t a1 = load_shared_u32(probabilities + 8 * 160 + offset);
            const uint32_t a2 = load_shared_u32(probabilities + 16 + offset);
            const uint32_t a3 = load_shared_u32(probabilities + 8 * 160 + 16 + offset);
#pragma unroll
            for (int column_tile = 0; column_tile < 2; ++column_tile) {
                const int feature = head * kHeadWidth + 16 * warp + 8 * column_tile + group;
                const uint8_t* values = value_fp8 + kModelWidth * value_stride +
                    (feature * value_stride + first_key + inner) / 2;
                const uint32_t b0 = inner + thread_in_group * 8 < kPositionRowsPadded
                    ? *reinterpret_cast<const uint32_t*>(values + offset) : 0;
                const uint32_t b1 = inner + 32 + thread_in_group * 8 < kPositionRowsPadded
                    ? *reinterpret_cast<const uint32_t*>(values + 16 + offset) : 0;
                mma_context_u4_s4(context[column_tile][0], context[column_tile][1],
                    context[column_tile][2], context[column_tile][3], a0, a1, a2, a3, b0, b1);
            }
        }
        const float* scales = reinterpret_cast<const float*>(value_fp8 + kModelWidth * value_stride * 3 / 2);
#pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            const int column = head * kHeadWidth + 16 * warp + 8 * column_tile + thread_in_group * 2;
            const float scale00 = probability_scale[group] * scales[column];
            const float scale01 = probability_scale[group] * scales[column + 1];
            const float scale10 = probability_scale[group + 8] * scales[column];
            const float scale11 = probability_scale[group + 8] * scales[column + 1];
            store_attention_pair<true>(output, (query_base + group) * kModelWidth + column,
                __floats2half2_rn(context[column_tile][0] * scale00, context[column_tile][1] * scale01));
            store_attention_pair<true>(output, (query_base + group + 8) * kModelWidth + column,
                __floats2half2_rn(context[column_tile][2] * scale10, context[column_tile][3] * scale11));
        }
        return;
    }
    float context[2][4] = {};
    for (int inner = 0; inner < kFp8ContextRowsPadded; inner += 32) {
        const int lane_inner = thread_in_group * 4;
        const uint8_t* probability_tile =
            &position_storage.probabilities[group][inner];
        const uint32_t probability0 = load_shared_u32(probability_tile + lane_inner);
        const uint32_t probability1 = load_shared_u32(
            probability_tile + 8 * kFp8ContextRowsPadded + lane_inner);
        const uint32_t probability2 = load_shared_u32(probability_tile + 16 + lane_inner);
        const uint32_t probability3 = load_shared_u32(
            probability_tile + 8 * kFp8ContextRowsPadded + 16 + lane_inner);
#pragma unroll
        for (int column_tile = 0; column_tile < 2; ++column_tile) {
            const int value_feature =
                head * kHeadWidth + 16 * warp + 8 * column_tile + group;
            const uint8_t* value_tile =
                value_fp8 + value_feature * value_stride + first_key + inner;
            const uint32_t value0 = *reinterpret_cast<const uint32_t*>(
                value_tile + lane_inner);
            const uint32_t value1 = inner + 16 < kPositionRowsPadded
                ? *reinterpret_cast<const uint32_t*>(value_tile + 16 + lane_inner)
                : 0;
            mma_m16n8k32_fp8(
                context[column_tile][0],
                context[column_tile][1],
                context[column_tile][2],
                context[column_tile][3],
                probability0,
                probability1,
                probability2,
                probability3,
                value0,
                value1);
        }
    }

    constexpr float kContextScale = (1.0f / 448.0f) * (8.0f / 448.0f);
#pragma unroll
    for (int column_tile = 0; column_tile < 2; ++column_tile) {
        const int output_column =
            head * kHeadWidth + 16 * warp + 8 * column_tile + thread_in_group * 2;
        const float* accumulator = context[column_tile];
        store_attention_pair<kPackedOutput>(output,
            (query_base + group) * kModelWidth + output_column,
            __floats2half2_rn(
                accumulator[0] * kContextScale,
                accumulator[1] * kContextScale));
        store_attention_pair<kPackedOutput>(output,
            (query_base + group + 8) * kModelWidth + output_column,
            __floats2half2_rn(
                accumulator[2] * kContextScale,
                accumulator[3] * kContextScale));
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_local_relpos_attention_tc_scores(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, const uint8_t* __restrict__ value_fp8,
    const __half* __restrict__ position, const __half* __restrict__ bias_u,
    const __half* __restrict__ bias_v, __half* __restrict__ output,
    int rows, int padded_rows, int valid_rows,
    int attention_left, int attention_right, int value_stride) {
    local_relpos_attention_tc_scores<false>(query, key, value, value_fp8,
        position, bias_u, bias_v, output, rows, padded_rows, valid_rows,
        attention_left, attention_right, value_stride);
}

// INT4 content operands free enough registers for four resident L4 CTAs.
extern "C" __global__ __launch_bounds__(256, 4)
void pk_sm89_local_relpos_attention_packed(
    const __half* __restrict__ query, const __half* __restrict__ key,
    const __half* __restrict__ value, const uint8_t* __restrict__ value_fp8,
    const __half* __restrict__ position, const __half* __restrict__ bias_u,
    const __half* __restrict__ bias_v, __half* __restrict__ output,
    int rows, int padded_rows, int valid_rows,
    int attention_left, int attention_right, int value_stride,
    const uint8_t* packed_key, const float* key_scales,
    const uint8_t* packed_position, const float* position_scales) {
    local_relpos_attention_tc_scores<true>(query, key, value, value_fp8,
        position, bias_u, bias_v, output, rows, padded_rows, valid_rows,
        attention_left, attention_right, value_stride,
        packed_key, key_scales, packed_position, position_scales);
}

extern "C" __global__ void pk_sm89_pack_score_vectors(
    const __half* input, uint8_t* output, float* scales, int vectors) {
    const int lane = threadIdx.x & 31;
    const int vector = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (vector >= vectors) return;
    float values[4], maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = __half2float(input[vector * 128 + 4 * lane + i]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lane == 0) scales[vector] = scale;
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        word |= uint32_t(uint8_t(__float2int_rn(values[i] / scale))) << (8 * i);
    *reinterpret_cast<uint32_t*>(output + vector * 128 + (lane / 8) * 32 + (lane % 4) * 8 + (lane & 4)) = word;
}

extern "C" __global__ void pk_sm89_pack_key_int4(
    const __half* input, uint8_t* output, float* scales, int vectors) {
    const int lane = threadIdx.x & 31;
    const int vector = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (vector >= vectors) return;
    float values[4], maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = __half2float(input[vector * 128 + 4 * lane + i]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    maximum = warp_max(maximum);
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
    if (lane == 0) scales[vector] = scale;
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        word |= (uint32_t(__float2int_rn(values[i] / scale)) & 15u) << (4 * i);
    *reinterpret_cast<uint16_t*>(output + vector * 64 + (lane / 16) * 32 +
        ((lane / 2) % 4) * 8 + ((lane / 2) & 4) + (lane & 1) * 2) = word;
}

extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_attention_output_fp8(
    const uint8_t* input, const uint8_t* weight, const float* weight_scales,
    __half* output, int rows) {
    // Context is FP16-rounded before E4M3 conversion. Range 16 leaves margin
    // above the value path's range 8 for quantized attention probabilities.
    ffn_fp8_async<1024, 1024, 6>(
        input, weight, weight_scales, nullptr, output, rows, 1.0f / 28.0f, 0);
}
