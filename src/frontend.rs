use crate::config::FeatureExtractorConfig;
use thiserror::Error;

#[derive(Debug)]
pub struct MelFeatures {
    /// Feature-major `[mel_bin, frame]`; index with `mel_bin * frames + frame`.
    pub values: Vec<f32>,
    pub mel_bins: usize,
    pub frames: usize,
    pub valid_frames: usize,
}

#[derive(Debug, Error)]
pub enum FrontendError {
    #[error("invalid frontend contract: {0}")]
    Invalid(String),
}

pub struct CpuMelFrontend {
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    preemphasis: f32,
    log_zero_guard: f32,
    normalize_epsilon: f32,
    window: Vec<f32>,
    filter_offsets: Vec<usize>,
    filter_bins: Vec<usize>,
    filter_values: Vec<f32>,
}

impl CpuMelFrontend {
    pub fn new(
        config: &FeatureExtractorConfig,
        window: Vec<f32>,
        mel_filters: Vec<f32>,
    ) -> Result<Self, FrontendError> {
        if config.n_fft == 0 || !config.n_fft.is_power_of_two() {
            return Err(FrontendError::Invalid(
                "FFT size must be a nonzero power of two".into(),
            ));
        }
        if config.hop_length == 0 || config.win_length > config.n_fft {
            return Err(FrontendError::Invalid(
                "hop must be nonzero and window must fit the FFT".into(),
            ));
        }
        if !config.center
            || config.dither != 0.0
            || config.mag_power != 2.0
            || config.normalize != "per_feature"
            || config.pad_mode != "constant"
        {
            return Err(FrontendError::Invalid(
                "only the pinned offline Parakeet v2 frontend is supported".into(),
            ));
        }
        if window.len() != config.win_length {
            return Err(FrontendError::Invalid(format!(
                "window has {} values, expected {}",
                window.len(),
                config.win_length
            )));
        }
        let bins = config.n_fft / 2 + 1;
        let expected_filters = config
            .feature_size
            .checked_mul(bins)
            .ok_or_else(|| FrontendError::Invalid("mel filter size overflow".into()))?;
        if mel_filters.len() != expected_filters {
            return Err(FrontendError::Invalid(format!(
                "mel filter bank has {} values, expected {expected_filters}",
                mel_filters.len()
            )));
        }
        if window
            .iter()
            .chain(&mel_filters)
            .any(|value| !value.is_finite())
        {
            return Err(FrontendError::Invalid(
                "window and mel filters must be finite".into(),
            ));
        }

        let mut filter_offsets = Vec::with_capacity(config.feature_size + 1);
        let mut filter_bins = Vec::new();
        let mut filter_values = Vec::new();
        filter_offsets.push(0);
        for filter in mel_filters.chunks_exact(bins) {
            for (bin, &value) in filter.iter().enumerate() {
                if value != 0.0 {
                    filter_bins.push(bin);
                    filter_values.push(value);
                }
            }
            filter_offsets.push(filter_bins.len());
        }

        Ok(Self {
            n_fft: config.n_fft,
            hop: config.hop_length,
            n_mels: config.feature_size,
            preemphasis: config.preemphasis,
            log_zero_guard: config.log_zero_guard,
            normalize_epsilon: config.normalize_epsilon,
            window: center_window(config.n_fft, &window),
            filter_offsets,
            filter_bins,
            filter_values,
        })
    }

    pub fn compute(&self, samples: &[f32]) -> Result<MelFeatures, FrontendError> {
        if samples.iter().any(|value| !value.is_finite()) {
            return Err(FrontendError::Invalid("PCM samples must be finite".into()));
        }
        if samples.is_empty() {
            return Ok(MelFeatures {
                values: Vec::new(),
                mel_bins: self.n_mels,
                frames: 0,
                valid_frames: 0,
            });
        }

        let frames = samples.len() / self.hop + 1;
        let valid_frames = samples.len() / self.hop;
        let mut values = vec![0.0_f32; self.n_mels * frames];
        let mut real = vec![0.0_f64; self.n_fft];
        let mut imaginary = vec![0.0_f64; self.n_fft];
        let mut power = vec![0.0_f64; self.n_fft / 2 + 1];
        let center = self.n_fft / 2;

        for frame in 0..frames {
            real.fill(0.0);
            imaginary.fill(0.0);
            let frame_origin = frame * self.hop;
            for (fft_index, value) in real.iter_mut().enumerate() {
                let centered = frame_origin + fft_index;
                if centered < center {
                    continue;
                }
                let sample = centered - center;
                if sample >= samples.len() {
                    continue;
                }
                let emphasized = if self.preemphasis > 0.0 && sample > 0 {
                    f64::from(samples[sample])
                        - f64::from(self.preemphasis) * f64::from(samples[sample - 1])
                } else {
                    f64::from(samples[sample])
                };
                *value = f64::from((emphasized * f64::from(self.window[fft_index])) as f32);
            }
            fft_in_place(&mut real, &mut imaginary);

            for bin in 0..power.len() {
                let re = f64::from(real[bin] as f32);
                let im = f64::from(imaginary[bin] as f32);
                let magnitude = (re * re + im * im).sqrt();
                power[bin] = magnitude.powf(2.0);
            }
            for mel in 0..self.n_mels {
                let start = self.filter_offsets[mel];
                let end = self.filter_offsets[mel + 1];
                let mut sum = 0.0_f64;
                for index in start..end {
                    sum += f64::from(self.filter_values[index]) * power[self.filter_bins[index]];
                }
                values[mel * frames + frame] = (sum + f64::from(self.log_zero_guard)).ln() as f32;
            }
        }

        for mel in 0..self.n_mels {
            let row = &mut values[mel * frames..(mel + 1) * frames];
            let mean = if valid_frames == 0 {
                0.0
            } else {
                row[..valid_frames]
                    .iter()
                    .map(|value| f64::from(*value))
                    .sum::<f64>()
                    / valid_frames as f64
            };
            let variance = if valid_frames <= 1 {
                0.0
            } else {
                row[..valid_frames]
                    .iter()
                    .map(|value| {
                        let difference = f64::from(*value) - mean;
                        difference * difference
                    })
                    .sum::<f64>()
                    / (valid_frames - 1) as f64
            };
            let denominator = variance.sqrt() + f64::from(self.normalize_epsilon);
            for (frame, value) in row.iter_mut().enumerate() {
                *value = if frame < valid_frames {
                    ((f64::from(*value) - mean) / denominator) as f32
                } else {
                    0.0
                };
            }
        }

        Ok(MelFeatures {
            values,
            mel_bins: self.n_mels,
            frames,
            valid_frames,
        })
    }
}

fn center_window(n_fft: usize, window: &[f32]) -> Vec<f32> {
    let mut centered = vec![0.0; n_fft];
    let left = (n_fft - window.len()) / 2;
    centered[left..left + window.len()].copy_from_slice(window);
    centered
}

fn fft_in_place(real: &mut [f64], imaginary: &mut [f64]) {
    let n = real.len();
    debug_assert!(n.is_power_of_two() && imaginary.len() == n);
    let mut reverse = 0;
    for index in 1..n {
        let mut bit = n >> 1;
        while reverse & bit != 0 {
            reverse ^= bit;
            bit >>= 1;
        }
        reverse ^= bit;
        if index < reverse {
            real.swap(index, reverse);
            imaginary.swap(index, reverse);
        }
    }

    let mut length = 2;
    while length <= n {
        let angle = -2.0 * std::f64::consts::PI / length as f64;
        let root_real = angle.cos();
        let root_imaginary = angle.sin();
        for start in (0..n).step_by(length) {
            let mut current_real = 1.0;
            let mut current_imaginary = 0.0;
            for index in 0..length / 2 {
                let left = start + index;
                let right = left + length / 2;
                let product_real =
                    current_real * real[right] - current_imaginary * imaginary[right];
                let product_imaginary =
                    current_real * imaginary[right] + current_imaginary * real[right];
                real[right] = real[left] - product_real;
                imaginary[right] = imaginary[left] - product_imaginary;
                real[left] += product_real;
                imaginary[left] += product_imaginary;
                let next_real = current_real * root_real - current_imaginary * root_imaginary;
                let next_imaginary = current_real * root_imaginary + current_imaginary * root_real;
                current_real = next_real;
                current_imaginary = next_imaginary;
            }
        }
        length <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_the_extra_centered_frame_at_boundary_lengths() {
        let frontend = fixture_frontend();
        for (samples, frames, valid) in [
            (1, 1, 0),
            (159, 1, 0),
            (160, 2, 1),
            (161, 2, 1),
            (320, 3, 2),
        ] {
            let input = (0..samples)
                .map(|index| (index as f32 * 0.01).sin())
                .collect::<Vec<_>>();
            let output = frontend.compute(&input).unwrap();
            assert_eq!((output.frames, output.valid_frames), (frames, valid));
            assert!(output.values.iter().all(|value| value.is_finite()));
            for mel in 0..output.mel_bins {
                for frame in valid..frames {
                    assert_eq!(output.values[mel * frames + frame], 0.0);
                }
            }
        }
    }

    #[test]
    fn empty_audio_has_no_frames() {
        let output = fixture_frontend().compute(&[]).unwrap();
        assert_eq!(output.mel_bins, 128);
        assert_eq!(output.frames, 0);
        assert_eq!(output.valid_frames, 0);
        assert!(output.values.is_empty());
    }

    fn fixture_frontend() -> CpuMelFrontend {
        let config = FeatureExtractorConfig::v2_english();
        let window = vec![1.0; 400];
        let mut filters = vec![0.0; 128 * 257];
        for mel in 0..128 {
            filters[mel * 257 + (mel % 257)] = 1.0;
        }
        CpuMelFrontend::new(&config, window, filters).unwrap()
    }
}
