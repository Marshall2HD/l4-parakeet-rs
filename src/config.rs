use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use thiserror::Error;

pub const V2_MODEL_REPOSITORY: &str = "nvidia/parakeet-tdt-0.6b-v2";
pub const V2_MODEL_REVISION: &str = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0";
pub const V3_MODEL_REPOSITORY: &str = "nvidia/parakeet-tdt-0.6b-v3";
pub const V3_MODEL_REVISION: &str = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    V2English,
    V3Multilingual,
}

impl ModelProfile {
    pub fn repository(self) -> &'static str {
        match self {
            Self::V2English => V2_MODEL_REPOSITORY,
            Self::V3Multilingual => V3_MODEL_REPOSITORY,
        }
    }

    pub fn revision(self) -> &'static str {
        match self {
            Self::V2English => V2_MODEL_REVISION,
            Self::V3Multilingual => V3_MODEL_REVISION,
        }
    }

    pub fn source_format(self) -> &'static str {
        match self {
            Self::V2English => "nemo",
            Self::V3Multilingual => "safetensors",
        }
    }

    pub fn blank_token_id(self) -> usize {
        match self {
            Self::V2English => 1024,
            Self::V3Multilingual => 8192,
        }
    }

    pub fn vocabulary_with_blank(self) -> usize {
        self.blank_token_id() + 1
    }

    pub fn joint_outputs(self) -> usize {
        self.vocabulary_with_blank() + 5
    }
}

impl std::fmt::Display for ModelProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V2English => formatter.write_str("v2-english"),
            Self::V3Multilingual => formatter.write_str("v3-multilingual"),
        }
    }
}

impl FromStr for ModelProfile {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "v2" | "v2-english" => Ok(Self::V2English),
            "v3" | "v3-multilingual" => Ok(Self::V3Multilingual),
            _ => Err(format!(
                "unknown model profile {value:?}; expected v2-english or v3-multilingual"
            )),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("model contract mismatch:\n  - {}", .0.join("\n  - "))]
    Contract(Vec<String>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub blank_token_id: usize,
    pub decoder_hidden_size: usize,
    pub dtype: String,
    pub durations: Vec<usize>,
    pub encoder_config: EncoderConfig,
    pub hidden_act: String,
    pub max_symbols_per_step: usize,
    pub model_type: String,
    pub num_decoder_layers: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EncoderConfig {
    pub attention_bias: bool,
    pub conv_kernel_size: usize,
    pub convolution_bias: bool,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_mel_bins: usize,
    pub scale_input: bool,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProcessorConfig {
    pub feature_extractor: FeatureExtractorConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureExtractorConfig {
    pub center: bool,
    pub dither: f32,
    pub feature_size: usize,
    pub hop_length: usize,
    pub log_zero_guard: f32,
    pub mag_power: f32,
    pub n_fft: usize,
    pub normalize: String,
    pub normalize_epsilon: f32,
    pub pad_mode: String,
    pub preemphasis: f32,
    pub return_attention_mask: bool,
    pub sampling_rate: usize,
    pub win_length: usize,
    pub window: String,
}

impl FeatureExtractorConfig {
    pub fn v2_english() -> Self {
        Self {
            center: true,
            dither: 0.0,
            feature_size: 128,
            hop_length: 160,
            log_zero_guard: 2.0_f32.powi(-24),
            mag_power: 2.0,
            n_fft: 512,
            normalize: "per_feature".into(),
            normalize_epsilon: 1e-5,
            pad_mode: "constant".into(),
            preemphasis: 0.97,
            return_attention_mask: true,
            sampling_rate: 16_000,
            win_length: 400,
            window: "hann".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationConfig {
    pub decoder_start_token_id: usize,
    pub suppress_tokens: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBundleConfig {
    pub profile: ModelProfile,
    pub source_repository: &'static str,
    pub source_format: &'static str,
    pub target_revision: &'static str,
    pub model: ModelConfig,
    pub processor: ProcessorConfig,
    pub generation: GenerationConfig,
}

impl ModelBundleConfig {
    pub fn load(model_dir: &Path, profile: ModelProfile) -> Result<Self, ConfigError> {
        let bundle = Self {
            profile,
            source_repository: profile.repository(),
            source_format: profile.source_format(),
            target_revision: profile.revision(),
            model: read_json(&model_dir.join("config.json"))?,
            processor: read_json(&model_dir.join("processor_config.json"))?,
            generation: read_json(&model_dir.join("generation_config.json"))?,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut mismatches = Vec::new();
        let model = &self.model;
        let encoder = &model.encoder_config;
        let frontend = &self.processor.feature_extractor;
        let blank_token_id = self.profile.blank_token_id();

        check_eq(
            &mut mismatches,
            "architectures",
            &model.architectures,
            &vec!["ParakeetForTDT".to_owned()],
        );
        check_eq(
            &mut mismatches,
            "model_type",
            &model.model_type,
            &"parakeet_tdt".to_owned(),
        );
        check_eq(
            &mut mismatches,
            "dtype",
            &model.dtype,
            &"float32".to_owned(),
        );
        check_eq(
            &mut mismatches,
            "blank_token_id",
            &model.blank_token_id,
            &blank_token_id,
        );
        check_eq(
            &mut mismatches,
            "vocab_size",
            &model.vocab_size,
            &self.profile.vocabulary_with_blank(),
        );
        check_eq(
            &mut mismatches,
            "durations",
            &model.durations,
            &vec![0, 1, 2, 3, 4],
        );
        check_eq(
            &mut mismatches,
            "decoder_hidden_size",
            &model.decoder_hidden_size,
            &640,
        );
        check_eq(
            &mut mismatches,
            "num_decoder_layers",
            &model.num_decoder_layers,
            &2,
        );
        check_eq(
            &mut mismatches,
            "max_symbols_per_step",
            &model.max_symbols_per_step,
            &10,
        );
        check_eq(
            &mut mismatches,
            "joint activation",
            &model.hidden_act,
            &"relu".to_owned(),
        );

        check_eq(
            &mut mismatches,
            "encoder.hidden_size",
            &encoder.hidden_size,
            &1024,
        );
        check_eq(
            &mut mismatches,
            "encoder.intermediate_size",
            &encoder.intermediate_size,
            &4096,
        );
        check_eq(
            &mut mismatches,
            "encoder.num_hidden_layers",
            &encoder.num_hidden_layers,
            &24,
        );
        check_eq(
            &mut mismatches,
            "encoder.num_attention_heads",
            &encoder.num_attention_heads,
            &8,
        );
        check_eq(
            &mut mismatches,
            "encoder.num_key_value_heads",
            &encoder.num_key_value_heads,
            &8,
        );
        check_eq(
            &mut mismatches,
            "encoder.num_mel_bins",
            &encoder.num_mel_bins,
            &128,
        );
        check_eq(
            &mut mismatches,
            "encoder.conv_kernel_size",
            &encoder.conv_kernel_size,
            &9,
        );
        check_eq(
            &mut mismatches,
            "encoder.hidden_act",
            &encoder.hidden_act,
            &"silu".to_owned(),
        );
        check_eq(
            &mut mismatches,
            "encoder.attention_bias",
            &encoder.attention_bias,
            &false,
        );
        check_eq(
            &mut mismatches,
            "encoder.convolution_bias",
            &encoder.convolution_bias,
            &false,
        );
        check_eq(
            &mut mismatches,
            "encoder.scale_input",
            &encoder.scale_input,
            &false,
        );
        check_eq(
            &mut mismatches,
            "encoder.subsampling_conv_channels",
            &encoder.subsampling_conv_channels,
            &256,
        );
        check_eq(
            &mut mismatches,
            "encoder.subsampling_conv_kernel_size",
            &encoder.subsampling_conv_kernel_size,
            &3,
        );
        check_eq(
            &mut mismatches,
            "encoder.subsampling_conv_stride",
            &encoder.subsampling_conv_stride,
            &2,
        );
        check_eq(
            &mut mismatches,
            "encoder.subsampling_factor",
            &encoder.subsampling_factor,
            &8,
        );

        check_eq(
            &mut mismatches,
            "frontend.feature_size",
            &frontend.feature_size,
            &128,
        );
        check_eq(&mut mismatches, "frontend.center", &frontend.center, &true);
        check_eq(&mut mismatches, "frontend.dither", &frontend.dither, &0.0);
        check_eq(
            &mut mismatches,
            "frontend.hop_length",
            &frontend.hop_length,
            &160,
        );
        check_eq(&mut mismatches, "frontend.n_fft", &frontend.n_fft, &512);
        check_eq(
            &mut mismatches,
            "frontend.log_zero_guard",
            &frontend.log_zero_guard,
            &2.0_f32.powi(-24),
        );
        check_eq(
            &mut mismatches,
            "frontend.mag_power",
            &frontend.mag_power,
            &2.0,
        );
        check_eq(
            &mut mismatches,
            "frontend.normalize",
            &frontend.normalize,
            &"per_feature".to_owned(),
        );
        check_eq(
            &mut mismatches,
            "frontend.normalize_epsilon",
            &frontend.normalize_epsilon,
            &1e-5,
        );
        check_eq(
            &mut mismatches,
            "frontend.pad_mode",
            &frontend.pad_mode,
            &"constant".to_owned(),
        );
        check_eq(
            &mut mismatches,
            "frontend.win_length",
            &frontend.win_length,
            &400,
        );
        check_eq(
            &mut mismatches,
            "frontend.sampling_rate",
            &frontend.sampling_rate,
            &16000,
        );
        check_eq(
            &mut mismatches,
            "frontend.preemphasis",
            &frontend.preemphasis,
            &0.97,
        );
        check_eq(
            &mut mismatches,
            "frontend.return_attention_mask",
            &frontend.return_attention_mask,
            &true,
        );
        check_eq(
            &mut mismatches,
            "frontend.window",
            &frontend.window,
            &"hann".to_owned(),
        );

        check_eq(
            &mut mismatches,
            "generation.decoder_start_token_id",
            &self.generation.decoder_start_token_id,
            &blank_token_id,
        );
        check_eq(
            &mut mismatches,
            "generation.suppress_tokens",
            &self.generation.suppress_tokens,
            &(self.profile.vocabulary_with_blank()..self.profile.joint_outputs())
                .collect::<Vec<_>>(),
        );

        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Contract(mismatches))
        }
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ConfigError> {
    let bytes = fs::read(path).map_err(|source| ConfigError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        source,
    })
}

fn check_eq<T: std::fmt::Debug + PartialEq>(
    mismatches: &mut Vec<String>,
    field: &str,
    actual: &T,
    expected: &T,
) {
    if actual != expected {
        mismatches.push(format!("{field}: expected {expected:?}, got {actual:?}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_nearly_correct_model() {
        let bundle = fixture_bundle(ModelProfile::V2English, 23);
        let error = bundle.validate().unwrap_err().to_string();
        assert!(error.contains("encoder.num_hidden_layers"));
        assert!(error.contains("expected 24, got 23"));
    }

    #[test]
    fn accepts_both_pinned_model_contracts() {
        fixture_bundle(ModelProfile::V2English, 24)
            .validate()
            .unwrap();
        fixture_bundle(ModelProfile::V3Multilingual, 24)
            .validate()
            .unwrap();
    }

    #[test]
    fn parses_short_and_explicit_profile_names() {
        assert_eq!(
            "v2".parse::<ModelProfile>().unwrap(),
            ModelProfile::V2English
        );
        assert_eq!(
            "v3-multilingual".parse::<ModelProfile>().unwrap(),
            ModelProfile::V3Multilingual
        );
    }

    fn fixture_bundle(profile: ModelProfile, layers: usize) -> ModelBundleConfig {
        ModelBundleConfig {
            profile,
            source_repository: profile.repository(),
            source_format: profile.source_format(),
            target_revision: profile.revision(),
            model: ModelConfig {
                architectures: vec!["ParakeetForTDT".into()],
                blank_token_id: profile.blank_token_id(),
                decoder_hidden_size: 640,
                dtype: "float32".into(),
                durations: vec![0, 1, 2, 3, 4],
                encoder_config: EncoderConfig {
                    attention_bias: false,
                    conv_kernel_size: 9,
                    convolution_bias: false,
                    hidden_act: "silu".into(),
                    hidden_size: 1024,
                    intermediate_size: 4096,
                    num_attention_heads: 8,
                    num_hidden_layers: layers,
                    num_key_value_heads: 8,
                    num_mel_bins: 128,
                    scale_input: false,
                    subsampling_conv_channels: 256,
                    subsampling_conv_kernel_size: 3,
                    subsampling_conv_stride: 2,
                    subsampling_factor: 8,
                },
                hidden_act: "relu".into(),
                max_symbols_per_step: 10,
                model_type: "parakeet_tdt".into(),
                num_decoder_layers: 2,
                vocab_size: profile.vocabulary_with_blank(),
            },
            processor: ProcessorConfig {
                feature_extractor: FeatureExtractorConfig::v2_english(),
            },
            generation: GenerationConfig {
                decoder_start_token_id: profile.blank_token_id(),
                suppress_tokens: (profile.vocabulary_with_blank()..profile.joint_outputs())
                    .collect(),
            },
        }
    }
}
