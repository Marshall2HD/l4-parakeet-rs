#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <float.h>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int kHidden = 640;
constexpr int kGates = 2560;
constexpr int kTokenOutputs = 1025;
constexpr int kBlank = 1024;
constexpr int kDurations = 5;
constexpr int kMaxSymbols = 10;
constexpr int kWarpsPerBlock = 4;

constexpr int kCommittedH = 0;
constexpr int kCommittedC = kCommittedH + 2 * kHidden;
constexpr int kCandidateH = kCommittedC + 2 * kHidden;
constexpr int kCandidateC = kCandidateH + 2 * kHidden;
constexpr int kGateScratch = kCandidateC + 2 * kHidden;
constexpr int kPredictionProjection = kGateScratch + kGates;
constexpr int kLogits = kPredictionProjection + kHidden;

constexpr int kSelectedToken = 2;
constexpr int kSelectedDuration = 3;

__device__ __forceinline__ float warp_sum(float value) {
    for (int width = 16; width > 0; width /= 2) {
        value += __shfl_down_sync(0xffffffff, value, width);
    }
    return value;
}

__device__ __forceinline__ bool better(float score, int id, float other_score, int other_id) {
    return score > other_score || (score == other_score && id < other_id);
}

// NaNs are excluded by better() before encoding. Canonicalize signed zero so
// the integer maximum retains the original lower-token-ID tie break.
__device__ __forceinline__ unsigned long long decision_key(float score, int id) {
    if (score == 0.0f) score = 0.0f;
    const unsigned bits = __float_as_uint(score);
    const unsigned ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (static_cast<unsigned long long>(ordered) << 32) | (0xffffffffu - unsigned(id));
}

__device__ __forceinline__ float2 load_cached_half2(const __half2* pointer) {
    // Keep the shared address space explicit: merging this with the global
    // fallback creates long-lived 64-bit addresses and spills in the mainloop.
    unsigned int bits;
    asm volatile("ld.shared.u32 %0, [%1];"
        : "=r"(bits)
        : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(pointer))));
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __half22float2(__half2(raw));
}

// Long-decode workspace after the state prefix: three per-tensor scales, then
// per-warp maxima for the largest supported grid. The Rust workspace capacity
// matches kCacheMaxima + 3 * kCacheWarps.
constexpr int kCacheScales = 9360;
constexpr int kCacheMaxima = kCacheScales + 16;
constexpr int kCacheWarps = 696;

__device__ __forceinline__ float2 load_ih_weight(const __half* pointer) {
    return __half22float2(*reinterpret_cast<const __half2*>(pointer));
}

__device__ __forceinline__ float half_warp_sum(float value) {
    for (int width = 8; width > 0; width /= 2) {
        value += __shfl_down_sync(0xffffffff, value, width, 16);
    }
    return value;
}

__device__ __forceinline__ float2 unpack_half2(unsigned int bits) {
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __half22float2(__half2(raw));
}

// Packed per-warp weight cache for long decodes: both layers' four recurrent
// gates and the layer-1 input gates of one hidden unit as signed INT8 words. Word
// (step, lane) holds elements 4*lane + 128*step .. +3 of that row, matching
// four packed hidden activations, so consecutive lanes read consecutive words.
constexpr int kCacheMatrices = 12;
constexpr int kCacheSteps = 5;
constexpr int kCacheHh0 = 0;
constexpr int kCacheHh1 = 4;
constexpr int kCacheIh1 = 8;

__device__ __forceinline__ int quantize_i8(float value) {
    return max(-127, min(127, __float2int_rn(value)));
}

__device__ __forceinline__ void accumulate_packed_row(
    float (&sums)[4],
    const uint32_t (*cache)[kCacheSteps][32],
    int first_matrix,
    const int8_t* __restrict__ values,
    float scale,
    int lane) {
    int dots[4] = {0, 0, 0, 0};
#pragma unroll
    for (int step = 0; step < kCacheSteps; ++step) {
        const int word = *reinterpret_cast<const int*>(values + 4 * lane + 128 * step);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            dots[gate] = __dp4a(static_cast<int>(cache[first_matrix + gate][step][lane]), word, dots[gate]);
        }
    }
    // Hidden values are bounded by tanh and stored with a fixed 1/127 scale.
#pragma unroll
    for (int gate = 0; gate < 4; ++gate) {
        sums[gate] += float(dots[gate]) * (scale / 127.0f);
    }
}

// Per-lane recurrent (hidden-to-hidden) partial sums for one hidden unit of
// both layers, taken from the state the next update will consume. These do not
// depend on the pending decision, so the joint phase computes them before the
// decision barrier and both LSTM phases only add the token-dependent terms.
__device__ __forceinline__ void accumulate_recurrent_partials(
    float (&partial)[2][4],
    const int8_t* __restrict__ state_h,
    const uint32_t (*cache)[kCacheSteps][32],
    const float (&scales)[2],
    int lane) {
#pragma unroll
    for (int layer = 0; layer < 2; ++layer) {
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        accumulate_packed_row(
            sums, cache, layer == 0 ? kCacheHh0 : kCacheHh1, state_h + layer * kHidden,
            scales[layer], lane);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            partial[layer][gate] = sums[gate];
        }
    }
}

__device__ __forceinline__ void lstm_cell(
    float input_sum,
    float forget_sum,
    float cell_sum,
    float output_sum,
    const float* __restrict__ committed_c,
    float* __restrict__ candidate_c,
    float* __restrict__ candidate_h,
    int state) {
    const float input_gate = 1.0f / (1.0f + expf(-input_sum));
    const float forget_gate = 1.0f / (1.0f + expf(-forget_sum));
    const float cell_gate = tanhf(cell_sum);
    const float output_gate = 1.0f / (1.0f + expf(-output_sum));
    const float cell = forget_gate * committed_c[state] + input_gate * cell_gate;
    candidate_c[state] = cell;
    candidate_h[state] = output_gate * tanhf(cell);
}

// FP16 short-request kernel body. Unchanged arithmetic and phase structure.
__device__ __forceinline__ void tdt_persistent_fp16(
    const __half* __restrict__ encoder_projection,
    const __half* __restrict__ embedding,
    const __half* __restrict__ weight_ih_l0,
    const __half* __restrict__ weight_hh_l0,
    const float* __restrict__ bias_ih_l0,
    const float* __restrict__ bias_hh_l0,
    const __half* __restrict__ weight_ih_l1,
    const __half* __restrict__ weight_hh_l1,
    const float* __restrict__ bias_ih_l1,
    const float* __restrict__ bias_hh_l1,
    const __half* __restrict__ decoder_projection_weight,
    const float* __restrict__ decoder_projection_bias,
    const __half* __restrict__ joint_weight,
    const float* __restrict__ joint_bias,
    float* __restrict__ workspace,
    int32_t* __restrict__ control,
    int32_t* __restrict__ output_tokens,
    int32_t* __restrict__ output_count,
    int frames) {
    cg::grid_group grid = cg::this_grid();
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % 32;
    const int warp = thread / 32;
    const int global_thread = static_cast<int>(blockIdx.x) * blockDim.x + thread;
    const int global_threads = static_cast<int>(gridDim.x) * blockDim.x;
    const int global_warp = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int global_warps = static_cast<int>(gridDim.x) * kWarpsPerBlock;
    float* prediction_projection = workspace + kPredictionProjection;
    auto* decisions = reinterpret_cast<unsigned long long*>(workspace + kLogits);
    float* duration_ring = workspace + kLogits + 6;
    int decision_slot = 0;
    // Cache three recurrent gates across token updates. Each lane owns its
    // half2 words; the existing startup grid barrier precedes all consumption.
    // The 30 KiB cache plus warp reduction preserves three resident CTAs on L4.
    __shared__ __half2 cached_hh[2][3][4][10][32];
    if (global_warp < kHidden) {
#pragma unroll
        for (int layer = 0; layer < 2; ++layer) {
#pragma unroll
            for (int gate = 0; gate < 3; ++gate) {
#pragma unroll
                for (int pair = 0; pair < 10; ++pair) {
                    cached_hh[layer][gate][warp][pair][lane] = *reinterpret_cast<const __half2*>(
                        (layer == 0 ? weight_hh_l0 : weight_hh_l1) +
                        (gate * kHidden + global_warp) * kHidden + 2 * lane + 64 * pair);
                }
            }
        }
    }
    __shared__ float reduction_scores[kWarpsPerBlock];
    __shared__ int reduction_ids[kWarpsPerBlock];

    for (int index = global_thread; index < 8 * kHidden; index += global_threads) {
        workspace[index] = 0.0f;
    }
    if (blockIdx.x == 0 && thread == 0) {
        *output_count = 0;
        for (int slot = 0; slot < 3; ++slot) {
            decisions[slot] = decision_key(-FLT_MAX, kTokenOutputs);
        }
    }
    grid.sync();

    // Every thread reads the same decision after joint production's barrier.
    // Three generations prevent reuse while a slower CTA still reads the last
    // decision, including blank steps that skip all prediction barriers.
    int current_frame = 0;
    int symbols_added = 0;
    int last_token = kBlank;
    bool emitted_any = false;
    bool alternate = false;
    bool prediction_valid = false;
    while (current_frame < frames) {
        if (!prediction_valid) {
            float* committed_h = workspace + (alternate ? kCandidateH : kCommittedH);
            float* committed_c = workspace + (alternate ? kCandidateC : kCommittedC);
            float* candidate_h = workspace + (alternate ? kCommittedH : kCandidateH);
            float* candidate_c = workspace + (alternate ? kCommittedC : kCandidateC);
            const __half* recurrent_ih[2] = {weight_ih_l0, weight_ih_l1};
            const __half* recurrent_hh[2] = {weight_hh_l0, weight_hh_l1};
            const float* recurrent_bias_ih[2] = {bias_ih_l0, bias_ih_l1};
            const float* recurrent_bias_hh[2] = {bias_hh_l0, bias_hh_l1};
#pragma unroll
            for (int layer = 0; layer < 2; ++layer) {
                for (int hidden = global_warp; hidden < kHidden; hidden += global_warps) {
                    float input_sum = lane == 0
                        ? recurrent_bias_ih[layer][hidden] +
                            recurrent_bias_hh[layer][hidden]
                        : 0.0f;
                    float forget_sum = lane == 0
                        ? recurrent_bias_ih[layer][kHidden + hidden] +
                            recurrent_bias_hh[layer][kHidden + hidden]
                        : 0.0f;
                    float cell_sum = lane == 0
                        ? recurrent_bias_ih[layer][2 * kHidden + hidden] +
                            recurrent_bias_hh[layer][2 * kHidden + hidden]
                        : 0.0f;
                    float output_sum = lane == 0
                        ? recurrent_bias_ih[layer][3 * kHidden + hidden] +
                            recurrent_bias_hh[layer][3 * kHidden + hidden]
                        : 0.0f;
                    const __half* input_weight = recurrent_ih[layer] + hidden * kHidden;
                    const __half* hidden_weight =
                        recurrent_hh[layer] + hidden * kHidden;
#pragma unroll
                    for (int inner = 2 * lane; inner < kHidden; inner += 64) {
                        float2 input_value;
                        if (layer == 0) {
                            input_value = emitted_any
                                ? __half22float2(*reinterpret_cast<const __half2*>(
                                      embedding + last_token * kHidden + inner))
                                : make_float2(0.0f, 0.0f);
                        } else {
                            input_value = *reinterpret_cast<const float2*>(candidate_h + inner);
                        }
                        const float2 hidden_value = *reinterpret_cast<const float2*>(
                            committed_h + layer * kHidden + inner);
                        float2 input_weights = load_ih_weight(input_weight + inner);
                        float2 hidden_weights = hidden == global_warp
                            ? load_cached_half2(&cached_hh[layer][0][warp][(inner - 2 * lane) / 64][lane])
                            : __half22float2(*reinterpret_cast<const __half2*>(hidden_weight + inner));
                        input_sum = fmaf(input_weights.x, input_value.x, input_sum);
                        input_sum = fmaf(input_weights.y, input_value.y, input_sum);
                        input_sum = fmaf(hidden_weights.x, hidden_value.x, input_sum);
                        input_sum = fmaf(hidden_weights.y, hidden_value.y, input_sum);

                        input_weights = load_ih_weight(input_weight + kHidden * kHidden + inner);
                        hidden_weights = hidden == global_warp
                            ? load_cached_half2(&cached_hh[layer][1][warp][(inner - 2 * lane) / 64][lane])
                            : __half22float2(*reinterpret_cast<const __half2*>(hidden_weight + kHidden * kHidden + inner));
                        forget_sum = fmaf(input_weights.x, input_value.x, forget_sum);
                        forget_sum = fmaf(input_weights.y, input_value.y, forget_sum);
                        forget_sum = fmaf(hidden_weights.x, hidden_value.x, forget_sum);
                        forget_sum = fmaf(hidden_weights.y, hidden_value.y, forget_sum);

                        input_weights = load_ih_weight(input_weight + 2 * kHidden * kHidden + inner);
                        hidden_weights = hidden == global_warp
                            ? load_cached_half2(&cached_hh[layer][2][warp][(inner - 2 * lane) / 64][lane])
                            : __half22float2(*reinterpret_cast<const __half2*>(hidden_weight + 2 * kHidden * kHidden + inner));
                        cell_sum = fmaf(input_weights.x, input_value.x, cell_sum);
                        cell_sum = fmaf(input_weights.y, input_value.y, cell_sum);
                        cell_sum = fmaf(hidden_weights.x, hidden_value.x, cell_sum);
                        cell_sum = fmaf(hidden_weights.y, hidden_value.y, cell_sum);

                        input_weights = load_ih_weight(input_weight + 3 * kHidden * kHidden + inner);
                        hidden_weights = __half22float2(*reinterpret_cast<const __half2*>(
                            hidden_weight + 3 * kHidden * kHidden + inner));
                        output_sum = fmaf(input_weights.x, input_value.x, output_sum);
                        output_sum = fmaf(input_weights.y, input_value.y, output_sum);
                        output_sum = fmaf(hidden_weights.x, hidden_value.x, output_sum);
                        output_sum = fmaf(hidden_weights.y, hidden_value.y, output_sum);
                    }
                    input_sum = warp_sum(input_sum);
                    forget_sum = warp_sum(forget_sum);
                    cell_sum = warp_sum(cell_sum);
                    output_sum = warp_sum(output_sum);
                    if (lane == 0) {
                        const float input_gate = 1.0f / (1.0f + expf(-input_sum));
                        const float forget_gate = 1.0f / (1.0f + expf(-forget_sum));
                        const float cell_gate = tanhf(cell_sum);
                        const float output_gate = 1.0f / (1.0f + expf(-output_sum));
                        const int state = layer * kHidden + hidden;
                        const float cell = forget_gate * committed_c[state] +
                            input_gate * cell_gate;
                        candidate_c[state] = cell;
                        candidate_h[state] = output_gate * tanhf(cell);
                    }
                }
                grid.sync();
            }

            for (int output = global_warp; output < kHidden; output += global_warps) {
                float sum = lane == 0 ? decoder_projection_bias[output] : 0.0f;
                const __half* row = decoder_projection_weight + output * kHidden;
#pragma unroll
                for (int inner = 2 * lane; inner < kHidden; inner += 64) {
                    const float2 weights = __half22float2(
                        *reinterpret_cast<const __half2*>(row + inner));
                    const float2 values = *reinterpret_cast<const float2*>(
                        candidate_h + kHidden + inner);
                    sum = fmaf(weights.x, values.x, sum);
                    sum = fmaf(weights.y, values.y, sum);
                }
                sum = warp_sum(sum);
                if (lane == 0) {
                    prediction_projection[output] = sum;
                }
            }
            grid.sync();
            prediction_valid = true;
        }

        // This slot was last used two decisions ago. Every old reader must
        // have finished before entering the preceding joint grid barrier.
        const int next_slot = decision_slot == 2 ? 0 : decision_slot + 1;
        if (global_thread == 0) {
            decisions[next_slot] = decision_key(-FLT_MAX, kTokenOutputs);
        }
        float local_score = -FLT_MAX;
        int local_id = kTokenOutputs;
        for (int output = global_warp;
             output < kTokenOutputs + kDurations;
             output += global_warps) {
            float sum = lane == 0 ? joint_bias[output] : 0.0f;
            const __half* row = joint_weight + output * kHidden;
#pragma unroll
            for (int inner = 2 * lane; inner < kHidden; inner += 64) {
                const float2 weights = __half22float2(
                    *reinterpret_cast<const __half2*>(row + inner));
                const float2 encoder_values = __half22float2(
                    *reinterpret_cast<const __half2*>(
                        encoder_projection + current_frame * kHidden + inner));
                const float2 prediction_values = *reinterpret_cast<const float2*>(
                    prediction_projection + inner);
                const float joint_input_0 =
                    fmaxf(encoder_values.x + prediction_values.x, 0.0f);
                const float joint_input_1 =
                    fmaxf(encoder_values.y + prediction_values.y, 0.0f);
                sum = fmaf(weights.x, joint_input_0, sum);
                sum = fmaf(weights.y, joint_input_1, sum);
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                if (output >= kTokenOutputs) {
                    duration_ring[decision_slot * kDurations + output - kTokenOutputs] = sum;
                } else if (better(sum, output, local_score, local_id)) {
                    local_score = sum;
                    local_id = output;
                }
            }
        }
        if (lane == 0) {
            reduction_scores[warp] = local_score;
            reduction_ids[warp] = local_id;
        }
        __syncthreads();
        if (thread == 0) {
            float best_score = -FLT_MAX;
            int best_id = kTokenOutputs;
#pragma unroll
            for (int i = 0; i < kWarpsPerBlock; ++i) {
                if (better(reduction_scores[i], reduction_ids[i], best_score, best_id)) {
                    best_score = reduction_scores[i];
                    best_id = reduction_ids[i];
                }
            }
            atomicMax(decisions + decision_slot, decision_key(best_score, best_id));
        }
        grid.sync();
        const int token = int(0xffffffffu - unsigned(decisions[decision_slot]));
        int duration = 0;
        for (int i = 1; i < kDurations; ++i) {
            if (duration_ring[decision_slot * kDurations + i] >
                duration_ring[decision_slot * kDurations + duration]) {
                duration = i;
            }
        }
        if (global_thread == 0) {
            control[kSelectedToken] = token;
            control[kSelectedDuration] = duration;
            if (token != kBlank) {
                output_tokens[*output_count] = token;
                *output_count += 1;
            }
        }
        decision_slot = next_slot;
        if (token != kBlank) {
            last_token = token;
            emitted_any = true;
            prediction_valid = false;
            alternate = !alternate;
        }
        ++symbols_added;
        current_frame += duration;
        if (duration != 0 || symbols_added == kMaxSymbols) {
            if (duration == 0) {
                ++current_frame;
            }
            symbols_added = 0;
        }
    }
}

// Long-request kernel body: E4M3 per-warp weight cache for both LSTM layers'
// recurrent gates and the layer-1 input gates, a per-request layer-0 input
// table, and recurrent terms accumulated before each decision.
__device__ __forceinline__ void tdt_persistent_fp8(
    const __half* __restrict__ encoder_projection,
    const __half* __restrict__ weight_hh_l0,
    const float* __restrict__ bias_ih_l0,
    const float* __restrict__ bias_hh_l0,
    const __half* __restrict__ weight_ih_l1,
    const __half* __restrict__ weight_hh_l1,
    const float* __restrict__ bias_ih_l1,
    const float* __restrict__ bias_hh_l1,
    const __half* __restrict__ decoder_projection_weight,
    const float* __restrict__ decoder_projection_bias,
    const __half* __restrict__ joint_weight,
    const float* __restrict__ joint_bias,
    const __half* __restrict__ input_table,
    float* __restrict__ workspace,
    int32_t* __restrict__ control,
    int32_t* __restrict__ output_tokens,
    int32_t* __restrict__ output_count,
    int frames) {
    cg::grid_group grid = cg::this_grid();
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % 32;
    const int warp = thread / 32;
    const int global_thread = static_cast<int>(blockIdx.x) * blockDim.x + thread;
    const int global_threads = static_cast<int>(gridDim.x) * blockDim.x;
    const int global_warp = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int global_warps = static_cast<int>(gridDim.x) * kWarpsPerBlock;
    // Each hidden unit is owned by one warp for the whole decode, and joint
    // outputs are owned by half-warps. Fail uniformly before any grid barrier
    // so the host observes an invalid token count on an unsupported geometry.
    if (global_warps < kHidden || global_warps > kCacheWarps ||
        2 * global_warps < kTokenOutputs + kDurations) {
        if (blockIdx.x == 0 && thread == 0) {
            *output_count = -1;
        }
        return;
    }
    float* prediction_projection = workspace + kPredictionProjection;
    auto* decisions = reinterpret_cast<unsigned long long*>(workspace + kLogits);
    float* duration_ring = workspace + kLogits + 6;
    int decision_slot = 0;
    // Warps beyond the hidden width mirror the last unit's operands so every
    // phase is straight-line code; they never publish state or scores.
    const bool owns_unit = global_warp < kHidden;
    const int unit = owns_unit ? global_warp : kHidden - 1;
    // 30 KiB per CTA keeps three resident CTAs on L4; each warp owns its rows.
    __shared__ uint32_t weight_cache[kWarpsPerBlock][kCacheMatrices][kCacheSteps][32];
    __shared__ float reduction_scores[kWarpsPerBlock];
    __shared__ int reduction_ids[kWarpsPerBlock];

    // The long path never uses gate scratch. Keep two quantized H banks here;
    // FP32 H/C remain authoritative for cell updates and the FP16 projection.
    int8_t* packed_h = reinterpret_cast<int8_t*>(workspace + kGateScratch);
    for (int index = global_thread; index < 8 * kHidden; index += global_threads) {
        workspace[index] = 0.0f;
    }
    for (int index = global_thread; index < 4 * kHidden; index += global_threads) {
        packed_h[index] = 0;
    }
    if (blockIdx.x == 0 && thread == 0) {
        *output_count = 0;
        for (int slot = 0; slot < 3; ++slot) {
            decisions[slot] = decision_key(-FLT_MAX, kTokenOutputs);
        }
    }
    // Derive per-tensor symmetric INT8 scales and pack the cache inside every
    // measured request, not at load. Each warp scans a strided subset of rows
    // of the three matrices, then block zero reduces the per-warp maxima.
    float* maxima = workspace + kCacheMaxima;
    float* scales = workspace + kCacheScales;
    {
        float matrix_maximum[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int matrix = 0; matrix < 3; ++matrix) {
            const __half* source = matrix == 0 ? weight_hh_l0 : matrix == 1 ? weight_hh_l1 : weight_ih_l1;
            for (int row = global_warp; row < kGates; row += global_warps) {
                for (int inner = lane; inner < kHidden; inner += 32) {
                    matrix_maximum[matrix] = fmaxf(
                        matrix_maximum[matrix], fabsf(__half2float(source[row * kHidden + inner])));
                }
            }
            for (int width = 16; width; width /= 2) {
                matrix_maximum[matrix] = fmaxf(
                    matrix_maximum[matrix],
                    __shfl_down_sync(0xffffffff, matrix_maximum[matrix], width));
            }
            if (lane == 0) maxima[matrix * global_warps + global_warp] = matrix_maximum[matrix];
        }
    }
    grid.sync();
    if (blockIdx.x == 0 && warp < 3) {
        float maximum = 0.0f;
        for (int index = lane; index < global_warps; index += 32) {
            maximum = fmaxf(maximum, maxima[warp * global_warps + index]);
        }
        for (int width = 16; width; width /= 2) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, width));
        }
        if (lane == 0) {
            scales[warp] = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        }
    }
    grid.sync();
    const float hh_scale[2] = {scales[0], scales[1]};
    const float ih_scale = scales[2];
#pragma unroll
    for (int matrix = 0; matrix < kCacheMatrices; ++matrix) {
        const int gate = matrix % 4;
        const __half* source = (matrix < kCacheHh1 ? weight_hh_l0 : matrix < kCacheIh1 ? weight_hh_l1 : weight_ih_l1) +
            (gate * kHidden + unit) * kHidden;
        const float scale = matrix < kCacheHh1 ? hh_scale[0] : matrix < kCacheIh1 ? hh_scale[1] : ih_scale;
#pragma unroll
        for (int step = 0; step < kCacheSteps; ++step) {
            const uint2 halves = *reinterpret_cast<const uint2*>(source + 4 * lane + 128 * step);
            const float2 low = unpack_half2(halves.x);
            const float2 high = unpack_half2(halves.y);
            const uint32_t word =
                static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(low.x / scale))) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(low.y / scale))) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(high.x / scale))) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(high.y / scale))) << 24);
            weight_cache[warp][matrix][step][lane] = word;
        }
    }
    grid.sync();

    // Per-unit and per-output operands that never change during the decode.
    float unit_bias[2][4];
#pragma unroll
    for (int gate = 0; gate < 4; ++gate) {
        unit_bias[0][gate] = bias_ih_l0[gate * kHidden + unit] + bias_hh_l0[gate * kHidden + unit];
        unit_bias[1][gate] = bias_ih_l1[gate * kHidden + unit] + bias_hh_l1[gate * kHidden + unit];
    }
    const __half* projection_row = decoder_projection_weight + unit * kHidden;
    const float projection_bias = decoder_projection_bias[unit];
    // The protected band includes 1024..1039 valid frames on this kernel.
    // Pack one immutable INT8 row per warp inside each longer request.
    const bool use_dp4a_projection = frames >= 1040;
    float projection_scale = 1.0f;
    uint32_t packed_projection_words[5];
    if (use_dp4a_projection) {
        float projection_maximum = 0.0f;
#pragma unroll
        for (int word = 0; word < 10; ++word) {
            const float2 values = __half22float2(*reinterpret_cast<const __half2*>(
                projection_row + 2 * lane + 64 * word));
            projection_maximum = fmaxf(projection_maximum, fmaxf(fabsf(values.x), fabsf(values.y)));
        }
        for (int width = 16; width; width /= 2) {
            projection_maximum = fmaxf(
                projection_maximum, __shfl_down_sync(0xffffffff, projection_maximum, width));
        }
        projection_maximum = __shfl_sync(0xffffffff, projection_maximum, 0);
        projection_scale = projection_maximum > 0.0f ? projection_maximum / 127.0f : 1.0f;
#pragma unroll
        for (int step = 0; step < 5; ++step) {
            const uint2 halves = *reinterpret_cast<const uint2*>(projection_row + 4 * lane + 128 * step);
            const float2 low = unpack_half2(halves.x);
            const float2 high = unpack_half2(halves.y);
            packed_projection_words[step] =
                static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(low.x / projection_scale))) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(low.y / projection_scale))) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(high.x / projection_scale))) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(quantize_i8(high.y / projection_scale))) << 24);
        }
    }
    const int half = lane >> 4;
    const int half_lane = lane & 15;
    const int joint_output = 2 * global_warp + half;
    const bool owns_output = joint_output < kTokenOutputs + kDurations;
    const int joint_row_index = owns_output ? joint_output : kTokenOutputs + kDurations - 1;
    const __half* joint_row = joint_weight + joint_row_index * kHidden;
    const float joint_row_bias = joint_bias[joint_row_index];
    // Each half-warp owns one immutable joint row throughout the request.
    // Keep its FP16 pairs packed in registers instead of streaming them per token.
    uint32_t joint_words[20];
#pragma unroll
    for (int step = 0; step < 20; ++step) {
        joint_words[step] = *reinterpret_cast<const uint32_t*>(
            joint_row + 2 * half_lane + 32 * step);
    }
    const uint32_t (*cache)[kCacheSteps][32] = weight_cache[warp];

    // Every thread reads the same decision after joint production's barrier.
    // Three generations prevent reuse while a slower CTA still reads the last
    // decision, including blank steps that skip all prediction barriers.
    int current_frame = 0;
    int symbols_added = 0;
    int last_token = kBlank;
    bool emitted_any = false;
    bool alternate = false;
    bool prediction_valid = false;
    // Recurrent partial sums for the state the next update consumes. The
    // first update reads the zero state; every joint phase refreshes them.
    float recurrent_partial[2][4];
    accumulate_recurrent_partials(recurrent_partial, packed_h, cache, hh_scale, lane);
    while (current_frame < frames) {
        if (!prediction_valid) {
            float* committed_c = workspace + (alternate ? kCandidateC : kCommittedC);
            float* candidate_h = workspace + (alternate ? kCommittedH : kCandidateH);
            float* candidate_c = workspace + (alternate ? kCommittedC : kCandidateC);
            int8_t* packed_candidate_h = packed_h + (alternate ? 0 : 2 * kHidden);

            // Layer 0: the recurrent terms were accumulated before the
            // decision; the input terms are one table row per gate. Issue the
            // table loads first so their latency overlaps the reductions.
            float table_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (emitted_any) {
                const __half* row = input_table + last_token * kGates + unit;
#pragma unroll
                for (int gate = 0; gate < 4; ++gate) {
                    table_values[gate] = __half2float(row[gate * kHidden]);
                }
            }
            float gate_sums[4];
#pragma unroll
            for (int gate = 0; gate < 4; ++gate) {
                gate_sums[gate] = warp_sum(recurrent_partial[0][gate]) + unit_bias[0][gate] +
                    table_values[gate];
            }
            if (owns_unit && lane == 0) {
                lstm_cell(gate_sums[0], gate_sums[1], gate_sums[2], gate_sums[3],
                    committed_c, candidate_c, candidate_h, unit);
                packed_candidate_h[unit] = static_cast<int8_t>(quantize_i8(candidate_h[unit] * 127.0f));
            }
            grid.sync();

            // Layer 1: only the cached input projection of the new layer-0
            // output remains on the critical path; prefetch the FP16 row only
            // in the protected band, otherwise its INT8 words are cached.
            uint32_t projection_words[10];
            if (!use_dp4a_projection) {
#pragma unroll
                for (int word = 0; word < 10; ++word) {
                    projection_words[word] =
                        *reinterpret_cast<const uint32_t*>(projection_row + 2 * lane + 64 * word);
                }
            }
#pragma unroll
            for (int gate = 0; gate < 4; ++gate) {
                gate_sums[gate] = recurrent_partial[1][gate];
            }
            accumulate_packed_row(gate_sums, cache, kCacheIh1, packed_candidate_h, ih_scale, lane);
#pragma unroll
            for (int gate = 0; gate < 4; ++gate) {
                gate_sums[gate] = warp_sum(gate_sums[gate]) + unit_bias[1][gate];
            }
            if (owns_unit && lane == 0) {
                lstm_cell(gate_sums[0], gate_sums[1], gate_sums[2], gate_sums[3],
                    committed_c, candidate_c, candidate_h, kHidden + unit);
                packed_candidate_h[kHidden + unit] = static_cast<int8_t>(quantize_i8(candidate_h[kHidden + unit] * 127.0f));
            }
            grid.sync();

            // Prediction projection of the new layer-1 output.
            float sum = lane == 0 ? projection_bias : 0.0f;
            if (use_dp4a_projection) {
                int dot = 0;
#pragma unroll
                for (int word = 0; word < 5; ++word) {
                    const int values = *reinterpret_cast<const int*>(
                        packed_candidate_h + kHidden + 4 * lane + 128 * word);
                    dot = __dp4a(static_cast<int>(packed_projection_words[word]), values, dot);
                }
                sum = float(dot) * (projection_scale / 127.0f) + sum;
            } else {
#pragma unroll
                for (int word = 0; word < 10; ++word) {
                    const float2 weights = unpack_half2(projection_words[word]);
                    const float2 values = *reinterpret_cast<const float2*>(
                        candidate_h + kHidden + 2 * lane + 64 * word);
                    sum = fmaf(weights.x, values.x, sum);
                    sum = fmaf(weights.y, values.y, sum);
                }
            }
            sum = warp_sum(sum);
            if (owns_unit && lane == 0) {
                prediction_projection[unit] = sum;
            }
            grid.sync();
            prediction_valid = true;
        }

        // This slot was last used two decisions ago. Every old reader must
        // have finished before entering the preceding joint grid barrier.
        const int next_slot = decision_slot == 2 ? 0 : decision_slot + 1;
        if (global_thread == 0) {
            decisions[next_slot] = decision_key(-FLT_MAX, kTokenOutputs);
        }
        // One joint output per half-warp, with the recurrent terms of the
        // state a token would update accumulated in the same straight-line
        // block so all loads overlap while the decision is being published.
        float score = half_lane == 0 ? joint_row_bias : 0.0f;
        const __half* frame_projection = encoder_projection + current_frame * kHidden;
#pragma unroll
        for (int step = 0; step < 20; ++step) {
            const int inner = 2 * half_lane + 32 * step;
            const float2 weights = unpack_half2(joint_words[step]);
            const float2 encoder_values = __half22float2(
                *reinterpret_cast<const __half2*>(frame_projection + inner));
            const float2 prediction_values = *reinterpret_cast<const float2*>(
                prediction_projection + inner);
            const float joint_input_0 = fmaxf(encoder_values.x + prediction_values.x, 0.0f);
            const float joint_input_1 = fmaxf(encoder_values.y + prediction_values.y, 0.0f);
            score = fmaf(weights.x, joint_input_0, score);
            score = fmaf(weights.y, joint_input_1, score);
        }
        accumulate_recurrent_partials(
            recurrent_partial, packed_h + (alternate ? 0 : 2 * kHidden), cache,
            hh_scale, lane);
        score = half_warp_sum(score);
        float local_score = -FLT_MAX;
        int local_id = kTokenOutputs;
        if (owns_output && half_lane == 0) {
            if (joint_output >= kTokenOutputs) {
                duration_ring[decision_slot * kDurations + joint_output - kTokenOutputs] = score;
            } else {
                local_score = score;
                local_id = joint_output;
            }
        }
        const float upper_score = __shfl_sync(0xffffffff, local_score, 16);
        const int upper_id = __shfl_sync(0xffffffff, local_id, 16);
        if (lane == 0) {
            if (better(upper_score, upper_id, local_score, local_id)) {
                local_score = upper_score;
                local_id = upper_id;
            }
            reduction_scores[warp] = local_score;
            reduction_ids[warp] = local_id;
        }
        __syncthreads();
        if (thread == 0) {
            float best_score = -FLT_MAX;
            int best_id = kTokenOutputs;
#pragma unroll
            for (int i = 0; i < kWarpsPerBlock; ++i) {
                if (better(reduction_scores[i], reduction_ids[i], best_score, best_id)) {
                    best_score = reduction_scores[i];
                    best_id = reduction_ids[i];
                }
            }
            atomicMax(decisions + decision_slot, decision_key(best_score, best_id));
        }
        grid.sync();
        const int token = int(0xffffffffu - unsigned(decisions[decision_slot]));
        int duration = 0;
        for (int i = 1; i < kDurations; ++i) {
            if (duration_ring[decision_slot * kDurations + i] >
                duration_ring[decision_slot * kDurations + duration]) {
                duration = i;
            }
        }
        if (global_thread == 0) {
            control[kSelectedToken] = token;
            control[kSelectedDuration] = duration;
            if (token != kBlank) {
                output_tokens[*output_count] = token;
                *output_count += 1;
            }
        }
        decision_slot = next_slot;
        if (token != kBlank) {
            last_token = token;
            emitted_any = true;
            prediction_valid = false;
            alternate = !alternate;
        }
        ++symbols_added;
        current_frame += duration;
        if (duration != 0 || symbols_added == kMaxSymbols) {
            if (duration == 0) {
                ++current_frame;
            }
            symbols_added = 0;
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 3)
void pk_sm89_tdt_persistent_v2(
    const __half* encoder_projection, const __half* embedding,
    const __half* weight_ih_l0, const __half* weight_hh_l0,
    const float* bias_ih_l0, const float* bias_hh_l0,
    const __half* weight_ih_l1, const __half* weight_hh_l1,
    const float* bias_ih_l1, const float* bias_hh_l1,
    const __half* decoder_projection_weight, const float* decoder_projection_bias,
    const __half* joint_weight, const float* joint_bias, const __half* input_table,
    float* workspace, int32_t* control, int32_t* output_tokens, int32_t* output_count,
    int frames) {
    (void)input_table;
    tdt_persistent_fp16(
        encoder_projection, embedding, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1, decoder_projection_weight,
        decoder_projection_bias, joint_weight, joint_bias, workspace,
        control, output_tokens, output_count, frames);
}

extern "C" __global__ __launch_bounds__(128, 3)
void pk_sm89_tdt_persistent_fp8_ih(
    const __half* encoder_projection, const __half* embedding,
    const __half* weight_ih_l0, const __half* weight_hh_l0,
    const float* bias_ih_l0, const float* bias_hh_l0,
    const __half* weight_ih_l1, const __half* weight_hh_l1,
    const float* bias_ih_l1, const float* bias_hh_l1,
    const __half* decoder_projection_weight, const float* decoder_projection_bias,
    const __half* joint_weight, const float* joint_bias, const __half* input_table,
    float* workspace, int32_t* control, int32_t* output_tokens, int32_t* output_count,
    int frames) {
    // The input table replaces the embedding and layer-0 input weights.
    (void)embedding;
    (void)weight_ih_l0;
    tdt_persistent_fp8(
        encoder_projection, weight_hh_l0, bias_ih_l0, bias_hh_l0,
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1, decoder_projection_weight,
        decoder_projection_bias, joint_weight, joint_bias, input_table, workspace,
        control, output_tokens, output_count, frames);
}
