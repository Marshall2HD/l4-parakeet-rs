#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <math.h>
#include <stdint.h>

namespace wmma = nvcuda::wmma;

struct __align__(2) Q4KBlock {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t quants[128];
};

static_assert(sizeof(Q4KBlock) == 144, "Q4_K block layout must match GGUF");

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

__device__ __forceinline__ void mma_m16n8k32_int8(
    int& d0,
    int& d1,
    int& d2,
    int& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* pointer) {
    return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ float apply_epilogue(float value, int activation) {
    if (activation == 1) {
        return value / (1.0f + expf(-value));
    }
    if (activation == 2) {
        return fmaxf(value, 0.0f);
    }
    return value;
}

// Static-shape baseline for the L4 engine. The host pads M to 16, N to 128,
// and K to 128. `weight_oi` is the native SafeTensors [output, input] layout,
// which is the column-major [K, N] view required by the B operand.
//
// Each warp computes two 16x16 output tiles at the same output columns so one
// B fragment serves 32 input rows. Bias, optional residual scaling, and
// activation are fused into the store so encoder blocks do not materialize
// epilogue intermediates.
extern "C" __global__ __launch_bounds__(64, 8)
void pk_sm89_fp16_linear_epilogue(
    const __half* __restrict__ input,
    const __half* __restrict__ weight_oi,
    const float* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int m,
    int n,
    int k,
    float output_scale,
    float residual_scale,
    int activation) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.y) * 32;
    const int col = (static_cast<int>(blockIdx.x) * 2 + warp) * 16;

    if (row >= m || col >= n) {
        return;
    }
    const bool has_second_row_tile = row + 16 < m;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
    wmma::fill_fragment(accumulator0, 0.0f);
    wmma::fill_fragment(accumulator1, 0.0f);

    for (int inner = 0; inner < k; inner += 16) {
        wmma::load_matrix_sync(a0, input + row * k + inner, k);
        wmma::load_matrix_sync(b, weight_oi + col * k + inner, k);
        wmma::mma_sync(accumulator0, a0, b, accumulator0);
        if (has_second_row_tile) {
            wmma::load_matrix_sync(a1, input + (row + 16) * k + inner, k);
            wmma::mma_sync(accumulator1, a1, b, accumulator1);
        }
    }

    __shared__ float tiles[2][2][16 * 16];
    wmma::store_matrix_sync(tiles[warp][0], accumulator0, 16, wmma::mem_row_major);
    if (has_second_row_tile) {
        wmma::store_matrix_sync(tiles[warp][1], accumulator1, 16, wmma::mem_row_major);
    }
    __syncwarp();

    for (int index = lane; index < 16 * 16; index += 32) {
        const int tile_row = index / 16;
        const int tile_col = index % 16;
        const int output_index = (row + tile_row) * n + col + tile_col;
        float value = tiles[warp][0][index];
        if (bias != nullptr) {
            value += bias[col + tile_col];
        }
        value *= output_scale;
        if (residual != nullptr) {
            value += residual_scale * __half2float(residual[output_index]);
        }
        if (activation == 1) {
            value = value / (1.0f + expf(-value));
        } else if (activation == 2) {
            value = fmaxf(value, 0.0f);
        }
        output[output_index] = __float2half_rn(value);
    }
    if (has_second_row_tile) {
        for (int index = lane; index < 16 * 16; index += 32) {
            const int tile_row = index / 16;
            const int tile_col = index % 16;
            const int output_index = (row + 16 + tile_row) * n + col + tile_col;
            float value = tiles[warp][1][index];
            if (bias != nullptr) {
                value += bias[col + tile_col];
            }
            value *= output_scale;
            if (residual != nullptr) {
                value += residual_scale * __half2float(residual[output_index]);
            }
            if (activation == 1) {
                value = value / (1.0f + expf(-value));
            } else if (activation == 2) {
                value = fmaxf(value, 0.0f);
            }
            output[output_index] = __float2half_rn(value);
        }
    }
}

// Short-sequence Q/K/V specialization. The three weights remain in their
// direct checkpoint-derived layouts; blockIdx.z selects one projection so a
// single launch feeds the contiguous query, key, and value workspace.
extern "C" __global__ __launch_bounds__(64, 8)
void pk_sm89_fp16_qkv(
    const __half* __restrict__ input,
    const __half* __restrict__ query_weight,
    const __half* __restrict__ key_weight,
    const __half* __restrict__ value_weight,
    __half* __restrict__ output,
    int m,
    int n,
    int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.y) * 32;
    const int col = (static_cast<int>(blockIdx.x) * 2 + warp) * 16;
    const int projection = static_cast<int>(blockIdx.z);

    if (row >= m || col >= n || projection >= 3) {
        return;
    }
    const bool has_second_row_tile = row + 16 < m;
    const __half* weights = projection == 0 ? query_weight :
        (projection == 1 ? key_weight : value_weight);
    __half* projected = output + projection * m * n;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
    wmma::fill_fragment(accumulator0, 0.0f);
    wmma::fill_fragment(accumulator1, 0.0f);

    for (int inner = 0; inner < k; inner += 16) {
        wmma::load_matrix_sync(a0, input + row * k + inner, k);
        wmma::load_matrix_sync(b, weights + col * k + inner, k);
        wmma::mma_sync(accumulator0, a0, b, accumulator0);
        if (has_second_row_tile) {
            wmma::load_matrix_sync(a1, input + (row + 16) * k + inner, k);
            wmma::mma_sync(accumulator1, a1, b, accumulator1);
        }
    }

    __shared__ float tiles[2][2][16 * 16];
    wmma::store_matrix_sync(tiles[warp][0], accumulator0, 16, wmma::mem_row_major);
    if (has_second_row_tile) {
        wmma::store_matrix_sync(tiles[warp][1], accumulator1, 16, wmma::mem_row_major);
    }
    __syncwarp();

    for (int index = lane; index < 16 * 16; index += 32) {
        const int tile_row = index / 16;
        const int tile_col = index % 16;
        projected[(row + tile_row) * n + col + tile_col] =
            __float2half_rn(tiles[warp][0][index]);
    }
    if (has_second_row_tile) {
        for (int index = lane; index < 16 * 16; index += 32) {
            const int tile_row = index / 16;
            const int tile_col = index % 16;
            projected[(row + 16 + tile_row) * n + col + tile_col] =
                __float2half_rn(tiles[warp][1][index]);
        }
    }
}

// Long-form specialization: four M tiles share each B fragment. Short inputs
// retain the two-tile kernel above to avoid its higher register footprint.
extern "C" __global__ __launch_bounds__(256, 2)
void pk_sm89_fp16_linear_epilogue_m64(
    const __half* __restrict__ input,
    const __half* __restrict__ weight_oi,
    const float* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ output,
    int m,
    int n,
    int k,
    float output_scale,
    float residual_scale,
    int activation) {
    constexpr int kRowTiles = 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.y) * (16 * kRowTiles);
    const int col = (static_cast<int>(blockIdx.x) * 8 + warp) * 16;

    if (row >= m || col >= n) {
        return;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[kRowTiles];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[kRowTiles];
    #pragma unroll
    for (int tile = 0; tile < kRowTiles; ++tile) {
        wmma::fill_fragment(accumulators[tile], 0.0f);
    }

    for (int inner = 0; inner < k; inner += 16) {
        wmma::load_matrix_sync(b, weight_oi + col * k + inner, k);
        #pragma unroll
        for (int tile = 0; tile < kRowTiles; ++tile) {
            if (row + tile * 16 < m) {
                wmma::load_matrix_sync(a[tile], input + (row + tile * 16) * k + inner, k);
                wmma::mma_sync(accumulators[tile], a[tile], b, accumulators[tile]);
            }
        }
    }

    __shared__ float tiles[8][kRowTiles][16 * 16];
    #pragma unroll
    for (int tile = 0; tile < kRowTiles; ++tile) {
        if (row + tile * 16 < m) {
            wmma::store_matrix_sync(
                tiles[warp][tile], accumulators[tile], 16, wmma::mem_row_major);
        }
    }
    __syncwarp();

    #pragma unroll
    for (int tile = 0; tile < kRowTiles; ++tile) {
        if (row + tile * 16 >= m) {
            continue;
        }
        for (int index = lane; index < 16 * 16; index += 32) {
            const int tile_row = index / 16;
            const int tile_col = index % 16;
            const int output_index = (row + tile * 16 + tile_row) * n + col + tile_col;
            float value = tiles[warp][tile][index];
            if (bias != nullptr) {
                value += bias[col + tile_col];
            }
            value *= output_scale;
            if (residual != nullptr) {
                value += residual_scale * __half2float(residual[output_index]);
            }
            if (activation == 1) {
                value = value / (1.0f + expf(-value));
            } else if (activation == 2) {
                value = fmaxf(value, 0.0f);
            }
            output[output_index] = __float2half_rn(value);
        }
    }
}

// Ada-native FP8 E4M3 path. One warp computes a 16x8 tile directly from
// row-major quantized activations and [output,input] AOT-packed weights.
// Activations use one static scale; weights use one scale per output row.
extern "C" __global__ __launch_bounds__(128, 4)
void pk_sm89_fp8_linear_epilogue(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ weight_oi,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    __half* __restrict__ output,
    int m,
    int n,
    int k,
    float input_scale,
    int activation) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int row = static_cast<int>(blockIdx.y) * 16;
    const int col = (static_cast<int>(blockIdx.x) * 4 + warp) * 8;

    if (row >= m || col >= n) {
        return;
    }

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    for (int inner = 0; inner < k; inner += 32) {
        const int lane_k = thread_in_group * 4;
        const uint32_t a0 = load_u32(input + (row + group) * k + inner + lane_k);
        const uint32_t a1 = load_u32(input + (row + group + 8) * k + inner + lane_k);
        const uint32_t a2 = load_u32(input + (row + group) * k + inner + 16 + lane_k);
        const uint32_t a3 = load_u32(input + (row + group + 8) * k + inner + 16 + lane_k);
        const uint32_t b0 = load_u32(weight_oi + (col + group) * k + inner + lane_k);
        const uint32_t b1 = load_u32(weight_oi + (col + group) * k + inner + 16 + lane_k);
        mma_m16n8k32_fp8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }

    const int output_col = col + thread_in_group * 2;
    const float scale0 = input_scale * weight_scales[output_col];
    const float scale1 = input_scale * weight_scales[output_col + 1];
    d0 = apply_epilogue(d0 * scale0 + bias[output_col], activation);
    d1 = apply_epilogue(d1 * scale1 + bias[output_col + 1], activation);
    d2 = apply_epilogue(d2 * scale0 + bias[output_col], activation);
    d3 = apply_epilogue(d3 * scale1 + bias[output_col + 1], activation);
    output[(row + group) * n + output_col] = __float2half_rn(d0);
    output[(row + group) * n + output_col + 1] = __float2half_rn(d1);
    output[(row + group + 8) * n + output_col] = __float2half_rn(d2);
    output[(row + group + 8) * n + output_col + 1] = __float2half_rn(d3);
}

// Ada-native signed INT8 path with exact INT32 Tensor Core accumulation and
// the same scaling and epilogue contract as the FP8 candidate.
extern "C" __global__ __launch_bounds__(128, 4)
void pk_sm89_int8_linear_epilogue(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight_oi,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    __half* __restrict__ output,
    int m,
    int n,
    int k,
    float input_scale,
    int activation) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int row = static_cast<int>(blockIdx.y) * 16;
    const int col = (static_cast<int>(blockIdx.x) * 4 + warp) * 8;

    if (row >= m || col >= n) {
        return;
    }

    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    for (int inner = 0; inner < k; inner += 32) {
        const int lane_k = thread_in_group * 4;
        const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(input);
        const uint8_t* weight_bytes = reinterpret_cast<const uint8_t*>(weight_oi);
        const uint32_t a0 = load_u32(input_bytes + (row + group) * k + inner + lane_k);
        const uint32_t a1 = load_u32(input_bytes + (row + group + 8) * k + inner + lane_k);
        const uint32_t a2 = load_u32(input_bytes + (row + group) * k + inner + 16 + lane_k);
        const uint32_t a3 = load_u32(input_bytes + (row + group + 8) * k + inner + 16 + lane_k);
        const uint32_t b0 = load_u32(weight_bytes + (col + group) * k + inner + lane_k);
        const uint32_t b1 = load_u32(weight_bytes + (col + group) * k + inner + 16 + lane_k);
        mma_m16n8k32_int8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }

    const int output_col = col + thread_in_group * 2;
    const float scale0 = input_scale * weight_scales[output_col];
    const float scale1 = input_scale * weight_scales[output_col + 1];
    const float f0 = apply_epilogue(static_cast<float>(d0) * scale0 + bias[output_col], activation);
    const float f1 = apply_epilogue(static_cast<float>(d1) * scale1 + bias[output_col + 1], activation);
    const float f2 = apply_epilogue(static_cast<float>(d2) * scale0 + bias[output_col], activation);
    const float f3 = apply_epilogue(static_cast<float>(d3) * scale1 + bias[output_col + 1], activation);
    output[(row + group) * n + output_col] = __float2half_rn(f0);
    output[(row + group) * n + output_col + 1] = __float2half_rn(f1);
    output[(row + group + 8) * n + output_col] = __float2half_rn(f2);
    output[(row + group + 8) * n + output_col + 1] = __float2half_rn(f3);
}

__device__ __forceinline__ void q4_k_scale_min(
    int group,
    const uint8_t* packed,
    uint8_t& scale,
    uint8_t& minimum) {
    if (group < 4) {
        scale = packed[group] & 63;
        minimum = packed[group + 4] & 63;
    } else {
        scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
        minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

__device__ __forceinline__ float q4_k_value(const Q4KBlock& block, int index) {
    const int group = index / 32;
    const int group_pair = group >> 1;
    const int value_in_group = index & 31;
    const uint8_t packed = block.quants[group_pair * 32 + value_in_group];
    const uint8_t quant = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
    uint8_t scale;
    uint8_t minimum;
    q4_k_scale_min(group, block.scales, scale, minimum);
    return __half2float(block.d) * static_cast<float>(scale) * static_cast<float>(quant)
        - __half2float(block.dmin) * static_cast<float>(minimum);
}

extern "C" __global__
void pk_sm89_q4_k_dequantize(
    const Q4KBlock* __restrict__ input,
    __half* __restrict__ output,
    int n,
    int k) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n * k) {
        const int row = index / k;
        const int column = index % k;
        const int blocks_per_row = k / 256;
        const Q4KBlock& block = input[row * blocks_per_row + column / 256];
        output[index] = __float2half_rn(q4_k_value(block, column % 256));
    }
}

// Direct GGUF Q4_K candidate. Each warp decodes one 16x256 weight tile into
// shared FP16 and immediately consumes it with Tensor Cores, so no full
// dequantized matrix is materialized. Q4_K requires K to be divisible by its
// 256-value superblock; unsupported Parakeet decoder matrices use FP16 instead.
extern "C" __global__ __launch_bounds__(128, 2)
void pk_sm89_q4_k_linear_epilogue(
    const __half* __restrict__ input,
    const Q4KBlock* __restrict__ weight_oi,
    const float* __restrict__ bias,
    __half* __restrict__ output,
    int m,
    int n,
    int k,
    int activation) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = static_cast<int>(blockIdx.y) * 16;
    const int col = (static_cast<int>(blockIdx.x) * 4 + warp) * 16;

    if (row >= m || col >= n) {
        return;
    }

    __shared__ __half decoded[4][16 * 256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    const int blocks_per_row = k / 256;
    for (int inner_block = 0; inner_block < blocks_per_row; ++inner_block) {
        for (int index = lane; index < 16 * 256; index += 32) {
            const int output_in_tile = index / 256;
            const int input_in_block = index % 256;
            const Q4KBlock& block = weight_oi[
                (col + output_in_tile) * blocks_per_row + inner_block];
            decoded[warp][index] = __float2half_rn(q4_k_value(block, input_in_block));
        }
        __syncwarp();

        const int inner = inner_block * 256;
        for (int offset = 0; offset < 256; offset += 16) {
            wmma::load_matrix_sync(a, input + row * k + inner + offset, k);
            wmma::load_matrix_sync(b, decoded[warp] + offset, 256);
            wmma::mma_sync(accumulator, a, b, accumulator);
        }
        __syncwarp();
    }

    __shared__ float q4_tiles[4][16 * 16];
    wmma::store_matrix_sync(q4_tiles[warp], accumulator, 16, wmma::mem_row_major);
    __syncwarp();

    for (int index = lane; index < 16 * 16; index += 32) {
        const int tile_row = index / 16;
        const int tile_col = index % 16;
        const int output_index = (row + tile_row) * n + col + tile_col;
        const float value = apply_epilogue(
            q4_tiles[warp][index] + bias[col + tile_col], activation);
        output[output_index] = __float2half_rn(value);
    }
}

extern "C" __global__
void pk_sm89_quantize_fp8(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    float inverse_scale,
    int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        output[index] = __nv_fp8_e4m3(__half2float(input[index]) * inverse_scale).__x;
    }
}

extern "C" __global__
void pk_sm89_quantize_int8(
    const __half* __restrict__ input,
    int8_t* __restrict__ output,
    float inverse_scale,
    int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        const float scaled = __half2float(input[index]) * inverse_scale;
        output[index] = static_cast<int8_t>(__float2int_rn(fminf(127.0f, fmaxf(-127.0f, scaled))));
    }
}

// Evicts candidate weights from the L4's 48 MiB L2 between cold-cache trials.
// The benchmark allocates a buffer larger than L2 and mutates every word so the
// compiler and memory system cannot elide the traffic.
extern "C" __global__
void pk_sm89_l2_scrub(uint32_t* data, int elements) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        data[index] = data[index] * 1664525u + 1013904223u;
    }
}
