#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

constexpr int kFft = 512;
constexpr int kFftBins = 257;
constexpr int kHop = 160;
constexpr int kMels = 128;

extern "C" __global__
void pk_sm89_pcm16_to_f32(
    const int16_t* __restrict__ pcm,
    float* __restrict__ samples,
    int sample_count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < sample_count) {
        samples[index] = static_cast<float>(pcm[index]) / 32768.0f;
    }
}

extern "C" __global__
void pk_sm89_frame_window(
    const float* __restrict__ samples,
    const float* __restrict__ window,
    float* __restrict__ frames,
    int sample_count,
    int active_frames,
    int frame_offset,
    float preemphasis) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = active_frames * kFft;
    if (index >= total) {
        return;
    }
    const int local_frame = index / kFft;
    const int fft_index = index - local_frame * kFft;
    const int frame = frame_offset + local_frame;
    const int sample = frame * kHop + fft_index - kFft / 2;
    float value = 0.0f;
    if (sample >= 0 && sample < sample_count) {
        value = samples[sample];
        if (sample > 0) {
            value -= preemphasis * samples[sample - 1];
        }
    }
    frames[index] = value * window[fft_index];
}

extern "C" __global__ __launch_bounds__(128)
void pk_sm89_mel_log_sparse(
    const float2* __restrict__ spectrum,
    const int32_t* __restrict__ filter_offsets,
    const int32_t* __restrict__ filter_bins,
    const float* __restrict__ filter_values,
    float* __restrict__ output,
    int active_frames,
    int frame_offset,
    float log_zero_guard) {
    const int local_frame = static_cast<int>(blockIdx.x);
    const int mel = static_cast<int>(threadIdx.x);
    if (local_frame >= active_frames || mel >= kMels) {
        return;
    }

    float sum = 0.0f;
    const float2* bins = spectrum + local_frame * kFftBins;
    for (int index = filter_offsets[mel]; index < filter_offsets[mel + 1]; ++index) {
        const float2 value = bins[filter_bins[index]];
        sum += filter_values[index] * (value.x * value.x + value.y * value.y);
    }
    output[(frame_offset + local_frame) * kMels + mel] = logf(sum + log_zero_guard);
}

extern "C" __global__ __launch_bounds__(256)
void pk_sm89_normalize_mel(
    float* __restrict__ features,
    int frame_count,
    int valid_frames,
    float epsilon) {
    const int mel = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ double reductions[256];

    double sum = 0.0;
    for (int frame = lane; frame < valid_frames; frame += blockDim.x) {
        sum += static_cast<double>(features[frame * kMels + mel]);
    }
    reductions[lane] = sum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (lane < width) {
            reductions[lane] += reductions[lane + width];
        }
        __syncthreads();
    }
    const double mean = valid_frames > 0 ? reductions[0] / valid_frames : 0.0;
    // Consume the mean in every warp before variance overwrites shared scratch.
    __syncthreads();

    double square_sum = 0.0;
    for (int frame = lane; frame < valid_frames; frame += blockDim.x) {
        const double difference =
            static_cast<double>(features[frame * kMels + mel]) - mean;
        square_sum += difference * difference;
    }
    reductions[lane] = square_sum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (lane < width) {
            reductions[lane] += reductions[lane + width];
        }
        __syncthreads();
    }
    const double variance = valid_frames > 1
        ? reductions[0] / static_cast<double>(valid_frames - 1)
        : 0.0;
    const double denominator = sqrt(variance) + static_cast<double>(epsilon);

    for (int frame = lane; frame < frame_count; frame += blockDim.x) {
        const int index = frame * kMels + mel;
        features[index] = frame < valid_frames
            ? static_cast<float>((static_cast<double>(features[index]) - mean) / denominator)
            : 0.0f;
    }
}
