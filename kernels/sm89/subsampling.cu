#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

constexpr int kChannels = 256;
constexpr int kInputFeatures = 128;
constexpr int kKernel = 3;
constexpr int kStride = 2;
constexpr int kFeatureGroup = 8;

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_subsample_first(
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int input_frames,
    int output_frame_start,
    int active_output_frames,
    int output_features) {
    const int channel = static_cast<int>(threadIdx.x);
    const int first_feature = static_cast<int>(blockIdx.x) * kFeatureGroup;
    const int local_frame = static_cast<int>(blockIdx.y);
    if (channel >= kChannels || first_feature >= output_features ||
        local_frame >= active_output_frames) {
        return;
    }

    const int output_frame = output_frame_start + local_frame;
    const float channel_bias = __half2float(bias[channel]);
    float sums[kFeatureGroup];
    #pragma unroll
    for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
        sums[grouped] = channel_bias;
    }
    #pragma unroll
    for (int kernel_frame = 0; kernel_frame < kKernel; ++kernel_frame) {
        const int input_frame = output_frame * kStride + kernel_frame - 1;
        if (input_frame < 0 || input_frame >= input_frames) {
            continue;
        }
        #pragma unroll
        for (int kernel_feature = 0; kernel_feature < kKernel; ++kernel_feature) {
            const int weight_index =
                (channel * kKernel + kernel_frame) * kKernel + kernel_feature;
            const float channel_weight = __half2float(weight[weight_index]);
            #pragma unroll
            for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
                const int feature = first_feature + grouped;
                const int input_feature = feature * kStride + kernel_feature - 1;
                if (feature < output_features && input_feature >= 0 &&
                    input_feature < kInputFeatures) {
                    const float value =
                        input[input_frame * kInputFeatures + input_feature];
                    sums[grouped] = fmaf(value, channel_weight, sums[grouped]);
                }
            }
        }
    }
    #pragma unroll
    for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
        const int feature = first_feature + grouped;
        if (feature < output_features) {
            const int row = local_frame * output_features + feature;
            output[row * kChannels + channel] =
                __float2half_rn(fmaxf(sums[grouped], 0.0f));
        }
    }
}

template<bool Quantize>
__device__ __forceinline__ void subsample_depthwise(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int input_frame_start,
    int input_frames,
    int input_features,
    int output_frame_start,
    int active_output_frames,
    int output_features,
    uint8_t* quantized,
    float* scales) {
    const int channel = static_cast<int>(threadIdx.x);
    const int first_feature = static_cast<int>(blockIdx.x) * kFeatureGroup;
    const int local_output_frame = static_cast<int>(blockIdx.y);
    if (channel >= kChannels || first_feature >= output_features ||
        local_output_frame >= active_output_frames) {
        return;
    }

    const int output_frame = output_frame_start + local_output_frame;
    const float channel_bias = __half2float(bias[channel]);
    float sums[kFeatureGroup];
    #pragma unroll
    for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
        sums[grouped] = channel_bias;
    }
    #pragma unroll
    for (int kernel_frame = 0; kernel_frame < kKernel; ++kernel_frame) {
        const int input_frame = output_frame * kStride + kernel_frame - 1;
        if (input_frame < 0 || input_frame >= input_frames) {
            continue;
        }
        #pragma unroll
        for (int kernel_feature = 0; kernel_feature < kKernel; ++kernel_feature) {
            const int weight_index =
                (channel * kKernel + kernel_frame) * kKernel + kernel_feature;
            const float channel_weight = __half2float(weight[weight_index]);
            #pragma unroll
            for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
                const int feature = first_feature + grouped;
                const int input_feature = feature * kStride + kernel_feature - 1;
                if (feature < output_features && input_feature >= 0 &&
                    input_feature < input_features) {
                    const int local_input_frame = input_frame - input_frame_start;
                    const int row = local_input_frame * input_features + input_feature;
                    sums[grouped] = fmaf(
                        __half2float(input[row * kChannels + channel]),
                        channel_weight,
                        sums[grouped]);
                }
            }
        }
    }
    if constexpr (Quantize) {
        __shared__ float partial[kFeatureGroup][8];
        __shared__ float row_scales[kFeatureGroup];
        const int lane = channel & 31;
        const int warp = channel >> 5;
        #pragma unroll
        for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
            // Preserve the materialized FP16 boundary before dynamic quantization.
            sums[grouped] = __half2float(__float2half_rn(sums[grouped]));
            float maximum = fabsf(sums[grouped]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffff, maximum, offset));
            }
            if (lane == 0) partial[grouped][warp] = maximum;
        }
        __syncthreads();
        // Each warp finishes one feature's reduction across the eight warps.
        float maximum = lane < 8 ? partial[warp][lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum,
                __shfl_down_sync(0xffffffff, maximum, offset));
        }
        if (lane == 0) {
            const float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
            row_scales[warp] = scale;
            if (first_feature + warp < output_features) {
                scales[local_output_frame * output_features + first_feature + warp] = scale;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
            const int feature = first_feature + grouped;
            if (feature < output_features) {
                const int row = local_output_frame * output_features + feature;
                quantized[row * kChannels + channel] =
                    __nv_fp8_e4m3(sums[grouped] / row_scales[grouped]).__x;
            }
        }
    } else {
        #pragma unroll
        for (int grouped = 0; grouped < kFeatureGroup; ++grouped) {
            const int feature = first_feature + grouped;
            if (feature < output_features) {
                const int row = local_output_frame * output_features + feature;
                output[row * kChannels + channel] = __float2half_rn(sums[grouped]);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_subsample_depthwise(
    const __half* input, const __half* weight, const __half* bias, __half* output,
    int input_frame_start, int input_frames, int input_features,
    int output_frame_start, int active_output_frames, int output_features) {
    subsample_depthwise<false>(input, weight, bias, output, input_frame_start,
        input_frames, input_features, output_frame_start, active_output_frames,
        output_features, nullptr, nullptr);
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_subsample_depthwise_fp8(
    const __half* input, const __half* weight, const __half* bias, uint8_t* output,
    int input_frame_start, int input_frames, int input_features,
    int output_frame_start, int active_output_frames, int output_features,
    float* scales) {
    subsample_depthwise<true>(input, weight, bias, nullptr, input_frame_start,
        input_frames, input_features, output_frame_start, active_output_frames,
        output_features, output, scales);
}

extern "C" __global__
void pk_sm89_subsample_flatten(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int active_frames,
    int padded_frames,
    int features,
    int output_frame_start,
    int valid_frames) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row_width = kChannels * features;
    const int total = padded_frames * row_width;
    if (index >= total) {
        return;
    }
    const int local_frame = index / row_width;
    const int flattened = index - local_frame * row_width;
    const int channel = flattened / features;
    const int feature = flattened - channel * features;
    const bool valid = local_frame < active_frames &&
        output_frame_start + local_frame < valid_frames;
    output[index] = valid
        ? input[(local_frame * features + feature) * kChannels + channel]
        : __float2half_rn(0.0f);
}
