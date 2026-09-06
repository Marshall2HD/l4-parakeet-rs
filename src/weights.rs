use crate::config::ModelProfile;
use crate::packing::{LinearFormat, estimate_linear};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use thiserror::Error;

const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U16,
    U8,
}

impl DType {
    fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::U16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    pub fn elements(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |total, size| total.checked_mul(*size))
    }

    pub fn bytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

#[derive(Debug, Serialize)]
pub struct SafeTensorIndex {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub data_start: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid SafeTensors file: {0}")]
    Invalid(String),
    #[error("invalid SafeTensors JSON header: {0}")]
    Json(#[from] serde_json::Error),
    #[error("weight manifest mismatch:\n  - {}", .0.join("\n  - "))]
    Manifest(Vec<String>),
}

impl SafeTensorIndex {
    pub fn open(path: &Path) -> Result<Self, WeightError> {
        let mut file = File::open(path).map_err(|source| WeightError::Io {
            path: path.to_owned(),
            source,
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|source| WeightError::Io {
                path: path.to_owned(),
                source,
            })?
            .len();

        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|source| WeightError::Io {
                path: path.to_owned(),
                source,
            })?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(WeightError::Invalid(format!(
                "header length {header_bytes} is outside 1..={MAX_HEADER_BYTES}"
            )));
        }
        let data_start = 8u64
            .checked_add(header_bytes)
            .ok_or_else(|| WeightError::Invalid("header offset overflow".into()))?;
        if data_start > file_bytes {
            return Err(WeightError::Invalid(format!(
                "header ends at byte {data_start}, beyond file length {file_bytes}"
            )));
        }

        file.seek(SeekFrom::Start(8))
            .map_err(|source| WeightError::Io {
                path: path.to_owned(),
                source,
            })?;
        let mut header = vec![0u8; header_bytes as usize];
        file.read_exact(&mut header)
            .map_err(|source| WeightError::Io {
                path: path.to_owned(),
                source,
            })?;

        let mut root = serde_json::from_slice::<BTreeMap<String, Value>>(&header)?;
        let metadata = root
            .remove("__metadata__")
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or_default();
        let tensors = root
            .into_iter()
            .map(|(name, value)| Ok((name, serde_json::from_value(value)?)))
            .collect::<Result<BTreeMap<_, _>, serde_json::Error>>()?;

        validate_layout(&tensors, file_bytes - data_start)?;
        Ok(Self {
            path: path.to_owned(),
            file_bytes,
            header_bytes,
            data_start,
            metadata,
            tensors,
        })
    }

    pub fn validate_model(&self, profile: ModelProfile) -> Result<ManifestReport, WeightError> {
        let expected = expected_model_manifest(profile);
        let mut mismatches = Vec::new();

        for (name, expected_info) in &expected {
            match self.tensors.get(name) {
                None => mismatches.push(format!("missing tensor {name}")),
                Some(actual) if actual.dtype != expected_info.dtype => mismatches.push(format!(
                    "{name}: expected dtype {:?}, got {:?}",
                    expected_info.dtype, actual.dtype
                )),
                Some(actual) if actual.shape != expected_info.shape => mismatches.push(format!(
                    "{name}: expected shape {:?}, got {:?}",
                    expected_info.shape, actual.shape
                )),
                Some(_) => {}
            }
        }

        for name in self.tensors.keys() {
            if !expected.contains_key(name) {
                mismatches.push(format!("unexpected tensor {name}"));
            }
        }

        if !mismatches.is_empty() {
            return Err(WeightError::Manifest(mismatches));
        }

        Ok(ManifestReport {
            tensors: self.tensors.len(),
            parameters: self
                .tensors
                .values()
                .filter(|tensor| tensor.dtype != DType::I64)
                .filter_map(TensorInfo::elements)
                .sum(),
            payload_bytes: self.file_bytes - self.data_start,
        })
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, WeightError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightError::Invalid(format!("missing tensor {name}")))?;
        if tensor.dtype != DType::F32 {
            return Err(WeightError::Invalid(format!(
                "{name}: expected F32, got {:?}",
                tensor.dtype
            )));
        }
        let mut file = File::open(&self.path).map_err(|source| WeightError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(self.data_start + tensor.data_offsets[0]))
            .map_err(|source| WeightError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut encoded = vec![
            0u8;
            usize::try_from(tensor.bytes()).map_err(|_| {
                WeightError::Invalid(format!("{name}: tensor byte count does not fit usize"))
            })?
        ];
        file.read_exact(&mut encoded)
            .map_err(|source| WeightError::Io {
                path: self.path.clone(),
                source,
            })?;
        encoded
            .chunks_exact(4)
            .enumerate()
            .map(|(index, bytes)| {
                let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
                value.is_finite().then_some(value).ok_or_else(|| {
                    WeightError::Invalid(format!("{name}: non-finite F32 value at element {index}"))
                })
            })
            .collect()
    }

    pub fn plan(&self, exl3_bits: u8) -> Result<WeightPlanSummary, WeightError> {
        let mut families = BTreeMap::<TensorFamily, FamilySummary>::new();
        for (name, tensor) in &self.tensors {
            let family = classify(name);
            let entry = families.entry(family).or_default();
            entry.tensors += 1;
            entry.parameters += tensor
                .elements()
                .ok_or_else(|| WeightError::Invalid(format!("{name}: element count overflow")))?;
            entry.source_bytes += tensor.bytes();
            entry.candidate_linear = family.is_candidate_linear();
        }

        let fixed_deployment_bytes = self
            .tensors
            .iter()
            .map(|(name, tensor)| fixed_deployment_bytes(classify(name), tensor))
            .try_fold(0u64, |total, bytes| {
                total
                    .checked_add(bytes? as u64)
                    .ok_or_else(|| WeightError::Invalid("deployment byte count overflow".into()))
            })?;
        let candidates = LinearFormat::ALL
            .into_iter()
            .map(|format| self.plan_format(format, exl3_bits, fixed_deployment_bytes))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(WeightPlanSummary {
            exl3_bits,
            source_bytes: self.tensors.values().map(TensorInfo::bytes).sum(),
            fixed_deployment_bytes,
            families,
            candidates,
        })
    }

    fn plan_format(
        &self,
        format: LinearFormat,
        exl3_bits: u8,
        fixed_bytes: u64,
    ) -> Result<FormatSummary, WeightError> {
        let mut linear_tensors = 0usize;
        let mut fallback_tensors = 0usize;
        let mut linear_bytes = 0u64;

        for (name, tensor) in &self.tensors {
            if !classify(name).is_candidate_linear() {
                continue;
            }
            linear_tensors += 1;
            let (logical_out, logical_in) = linear_shape(&tensor.shape).ok_or_else(|| {
                WeightError::Invalid(format!(
                    "candidate linear {name} has unsupported shape {:?}",
                    tensor.shape
                ))
            })?;
            let estimate = estimate_linear(format, logical_in, logical_out, exl3_bits)
                .map_err(|error| WeightError::Invalid(error.to_string()))?;
            let layout = match estimate {
                Some(layout) => layout,
                None => {
                    fallback_tensors += 1;
                    estimate_linear(LinearFormat::PackedFp16, logical_in, logical_out, exl3_bits)
                        .map_err(|error| WeightError::Invalid(error.to_string()))?
                        .expect("FP16 supports every nonempty linear")
                }
            };
            linear_bytes = linear_bytes
                .checked_add(
                    layout
                        .total_bytes()
                        .ok_or_else(|| WeightError::Invalid("packed byte count overflow".into()))?
                        as u64,
                )
                .ok_or_else(|| WeightError::Invalid("deployment byte count overflow".into()))?;
        }

        let deployment_bytes = fixed_bytes
            .checked_add(linear_bytes)
            .ok_or_else(|| WeightError::Invalid("deployment byte count overflow".into()))?;
        Ok(FormatSummary {
            format: format.label(exl3_bits),
            linear_tensors,
            fallback_tensors,
            linear_bytes,
            deployment_bytes,
            fraction_of_source: deployment_bytes as f64
                / self.tensors.values().map(TensorInfo::bytes).sum::<u64>() as f64,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct ManifestReport {
    pub tensors: usize,
    pub parameters: usize,
    pub payload_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorFamily {
    EncoderFfnLinear,
    EncoderAttentionLinear,
    EncoderPointwiseLinear,
    SubsamplingLinear,
    EncoderProjectionLinear,
    DecoderProjectionLinear,
    DecoderLstmLinear,
    JointHeadLinear,
    Embedding,
    Convolution,
    SensitiveFp32,
    OtherFp16,
    Counter,
}

impl TensorFamily {
    pub(crate) fn is_candidate_linear(self) -> bool {
        matches!(
            self,
            Self::EncoderFfnLinear
                | Self::EncoderAttentionLinear
                | Self::EncoderPointwiseLinear
                | Self::SubsamplingLinear
                | Self::EncoderProjectionLinear
                | Self::DecoderProjectionLinear
                | Self::DecoderLstmLinear
                | Self::JointHeadLinear
        )
    }
}

#[derive(Debug, Default, Serialize)]
pub struct FamilySummary {
    pub tensors: usize,
    pub parameters: usize,
    pub source_bytes: u64,
    pub candidate_linear: bool,
}

#[derive(Debug, Serialize)]
pub struct FormatSummary {
    pub format: String,
    pub linear_tensors: usize,
    pub fallback_tensors: usize,
    pub linear_bytes: u64,
    pub deployment_bytes: u64,
    pub fraction_of_source: f64,
}

#[derive(Debug, Serialize)]
pub struct WeightPlanSummary {
    pub exl3_bits: u8,
    pub source_bytes: u64,
    pub fixed_deployment_bytes: u64,
    pub families: BTreeMap<TensorFamily, FamilySummary>,
    pub candidates: Vec<FormatSummary>,
}

fn validate_layout(
    tensors: &BTreeMap<String, TensorInfo>,
    payload_bytes: u64,
) -> Result<(), WeightError> {
    let mut ranges = tensors
        .iter()
        .map(|(name, tensor)| (tensor.data_offsets[0], tensor.data_offsets[1], name, tensor))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.0);

    let mut cursor = 0u64;
    for (start, end, name, tensor) in ranges {
        if start != cursor {
            return Err(WeightError::Invalid(format!(
                "{name}: expected contiguous offset {cursor}, got {start}"
            )));
        }
        if end < start || end > payload_bytes {
            return Err(WeightError::Invalid(format!(
                "{name}: invalid range [{start}, {end}) for {payload_bytes}-byte payload"
            )));
        }
        let elements = tensor
            .elements()
            .ok_or_else(|| WeightError::Invalid(format!("{name}: element count overflow")))?;
        let expected_bytes = elements
            .checked_mul(tensor.dtype.bytes())
            .ok_or_else(|| WeightError::Invalid(format!("{name}: byte count overflow")))?
            as u64;
        if end - start != expected_bytes {
            return Err(WeightError::Invalid(format!(
                "{name}: shape/dtype require {expected_bytes} bytes, range contains {}",
                end - start
            )));
        }
        cursor = end;
    }

    if cursor != payload_bytes {
        return Err(WeightError::Invalid(format!(
            "tensor ranges end at {cursor}, payload ends at {payload_bytes}"
        )));
    }
    Ok(())
}

pub(crate) fn classify(name: &str) -> TensorFamily {
    if name.ends_with("num_batches_tracked") {
        TensorFamily::Counter
    } else if name.ends_with(".weight") && name.contains(".feed_forward") {
        TensorFamily::EncoderFfnLinear
    } else if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
        || name.ends_with(".self_attn.o_proj.weight")
        || name.ends_with(".self_attn.relative_k_proj.weight")
    {
        TensorFamily::EncoderAttentionLinear
    } else if name.ends_with(".conv.pointwise_conv1.weight")
        || name.ends_with(".conv.pointwise_conv2.weight")
    {
        TensorFamily::EncoderPointwiseLinear
    } else if name == "encoder.subsampling.linear.weight"
        || name == "encoder.subsampling.layers.3.weight"
        || name == "encoder.subsampling.layers.6.weight"
    {
        TensorFamily::SubsamplingLinear
    } else if name == "encoder_projector.weight" {
        TensorFamily::EncoderProjectionLinear
    } else if name == "decoder.decoder_projector.weight" {
        TensorFamily::DecoderProjectionLinear
    } else if name.starts_with("decoder.lstm.weight_") {
        TensorFamily::DecoderLstmLinear
    } else if name == "joint.head.weight" {
        TensorFamily::JointHeadLinear
    } else if name == "decoder.embedding.weight" {
        TensorFamily::Embedding
    } else if name == "joint.head.bias"
        || name == "encoder.subsampling.linear.bias"
        || name == "encoder_projector.bias"
        || name == "decoder.decoder_projector.bias"
        || name.contains(".conv.norm.")
        || name.starts_with("decoder.lstm.bias_")
    {
        TensorFamily::SensitiveFp32
    } else if name.contains(".conv.") || name.starts_with("encoder.subsampling.layers.") {
        TensorFamily::Convolution
    } else {
        TensorFamily::OtherFp16
    }
}

fn fixed_deployment_bytes(family: TensorFamily, tensor: &TensorInfo) -> Result<usize, WeightError> {
    let elements = tensor
        .elements()
        .ok_or_else(|| WeightError::Invalid("element count overflow".into()))?;
    match family {
        TensorFamily::Counter => Ok(0),
        TensorFamily::SensitiveFp32 => elements
            .checked_mul(4)
            .ok_or_else(|| WeightError::Invalid("FP32 deployment size overflow".into())),
        family if family.is_candidate_linear() => Ok(0),
        _ => elements
            .checked_mul(2)
            .ok_or_else(|| WeightError::Invalid("FP16 deployment size overflow".into())),
    }
}

pub(crate) fn linear_shape(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [output, input] => Some((*output, *input)),
        [output, input, 1] => Some((*output, *input)),
        [output, input, 1, 1] => Some((*output, *input)),
        _ => None,
    }
}

fn expected_model_manifest(profile: ModelProfile) -> BTreeMap<String, TensorInfo> {
    let mut tensors = BTreeMap::new();
    let mut offset = 0;
    let mut add = |name: String, dtype: DType, shape: &[usize]| {
        let bytes = shape.iter().product::<usize>() * dtype.bytes();
        tensors.insert(
            name,
            TensorInfo {
                dtype,
                shape: shape.to_vec(),
                data_offsets: [offset, offset + bytes as u64],
            },
        );
        offset += bytes as u64;
    };

    add("decoder.decoder_projector.bias".into(), DType::F32, &[640]);
    add(
        "decoder.decoder_projector.weight".into(),
        DType::F32,
        &[640, 640],
    );
    add(
        "decoder.embedding.weight".into(),
        DType::F32,
        &[profile.vocabulary_with_blank(), 640],
    );
    for kind in ["bias_hh", "bias_ih"] {
        for layer in 0..2 {
            add(format!("decoder.lstm.{kind}_l{layer}"), DType::F32, &[2560]);
        }
    }
    for kind in ["weight_hh", "weight_ih"] {
        for layer in 0..2 {
            add(
                format!("decoder.lstm.{kind}_l{layer}"),
                DType::F32,
                &[2560, 640],
            );
        }
    }

    let subsampling = [
        ("encoder.subsampling.layers.0.bias", vec![256]),
        ("encoder.subsampling.layers.0.weight", vec![256, 1, 3, 3]),
        ("encoder.subsampling.layers.2.bias", vec![256]),
        ("encoder.subsampling.layers.2.weight", vec![256, 1, 3, 3]),
        ("encoder.subsampling.layers.3.bias", vec![256]),
        ("encoder.subsampling.layers.3.weight", vec![256, 256, 1, 1]),
        ("encoder.subsampling.layers.5.bias", vec![256]),
        ("encoder.subsampling.layers.5.weight", vec![256, 1, 3, 3]),
        ("encoder.subsampling.layers.6.bias", vec![256]),
        ("encoder.subsampling.layers.6.weight", vec![256, 256, 1, 1]),
        ("encoder.subsampling.linear.bias", vec![1024]),
        ("encoder.subsampling.linear.weight", vec![1024, 4096]),
    ];
    for (name, shape) in subsampling {
        add(name.into(), DType::F32, &shape);
    }

    if profile == ModelProfile::V2English {
        add("frontend.mel_filters".into(), DType::F32, &[128, 257]);
        add("frontend.window".into(), DType::F32, &[400]);
    }

    for layer in 0..24 {
        let prefix = format!("encoder.layers.{layer}");
        add(
            format!("{prefix}.conv.norm.num_batches_tracked"),
            DType::I64,
            &[],
        );
        add(
            format!("{prefix}.conv.depthwise_conv.weight"),
            DType::F32,
            &[1024, 1, 9],
        );
        for suffix in ["bias", "running_mean", "running_var", "weight"] {
            add(format!("{prefix}.conv.norm.{suffix}"), DType::F32, &[1024]);
        }
        add(
            format!("{prefix}.conv.pointwise_conv1.weight"),
            DType::F32,
            &[2048, 1024, 1],
        );
        add(
            format!("{prefix}.conv.pointwise_conv2.weight"),
            DType::F32,
            &[1024, 1024, 1],
        );
        for feed_forward in ["feed_forward1", "feed_forward2"] {
            add(
                format!("{prefix}.{feed_forward}.linear1.weight"),
                DType::F32,
                &[4096, 1024],
            );
            add(
                format!("{prefix}.{feed_forward}.linear2.weight"),
                DType::F32,
                &[1024, 4096],
            );
        }
        for norm in [
            "norm_conv",
            "norm_feed_forward1",
            "norm_feed_forward2",
            "norm_out",
            "norm_self_att",
        ] {
            add(format!("{prefix}.{norm}.bias"), DType::F32, &[1024]);
            add(format!("{prefix}.{norm}.weight"), DType::F32, &[1024]);
        }
        add(format!("{prefix}.self_attn.bias_u"), DType::F32, &[8, 128]);
        add(format!("{prefix}.self_attn.bias_v"), DType::F32, &[8, 128]);
        for projection in ["k_proj", "o_proj", "q_proj", "relative_k_proj", "v_proj"] {
            add(
                format!("{prefix}.self_attn.{projection}.weight"),
                DType::F32,
                &[1024, 1024],
            );
        }
    }

    add("encoder_projector.bias".into(), DType::F32, &[640]);
    add("encoder_projector.weight".into(), DType::F32, &[640, 1024]);
    add(
        "joint.head.bias".into(),
        DType::F32,
        &[profile.joint_outputs()],
    );
    add(
        "joint.head.weight".into(),
        DType::F32,
        &[profile.joint_outputs(), 640],
    );
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn pinned_manifests_have_expected_inventories() {
        for (profile, tensor_count, parameter_count) in [
            (ModelProfile::V2English, 725, 617_908_374),
            (ModelProfile::V3Multilingual, 723, 627_057_286),
        ] {
            let manifest = expected_model_manifest(profile);
            assert_eq!(manifest.len(), tensor_count);
            let parameters = manifest
                .values()
                .filter(|tensor| tensor.dtype != DType::I64)
                .filter_map(TensorInfo::elements)
                .sum::<usize>();
            assert_eq!(parameters, parameter_count);
        }
    }

    #[test]
    fn reads_a_header_without_loading_tensor_data() {
        let path = temporary_path("valid.safetensors");
        let header = br#"{"value":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"format":"pt"}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0u8; 8]).unwrap();
        drop(file);

        let index = SafeTensorIndex::open(&path).unwrap();
        assert_eq!(index.tensors["value"].shape, [2]);
        assert_eq!(index.metadata["format"], "pt");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_gap_in_tensor_data() {
        let path = temporary_path("gap.safetensors");
        let header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0u8; 8]).unwrap();
        drop(file);

        let error = SafeTensorIndex::open(&path).unwrap_err().to_string();
        assert!(error.contains("expected contiguous offset 0, got 4"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn plan_keeps_numeric_choices_open_per_tensor_family() {
        assert_eq!(
            classify("decoder.lstm.weight_hh_l0"),
            TensorFamily::DecoderLstmLinear
        );
        assert_eq!(
            classify("decoder.lstm.bias_hh_l0"),
            TensorFamily::SensitiveFp32
        );
        assert_eq!(
            classify("encoder.layers.12.feed_forward1.linear2.weight"),
            TensorFamily::EncoderFfnLinear
        );
        assert_eq!(classify("joint.head.weight"), TensorFamily::JointHeadLinear);
        assert_eq!(classify("joint.head.bias"), TensorFamily::SensitiveFp32);
        assert_eq!(
            classify("encoder_projector.bias"),
            TensorFamily::SensitiveFp32
        );
    }

    #[test]
    fn compares_every_candidate_on_the_pinned_manifest() {
        let tensors = expected_model_manifest(ModelProfile::V2English);
        let index = SafeTensorIndex {
            path: PathBuf::from("model.safetensors"),
            file_bytes: tensors.values().map(TensorInfo::bytes).sum(),
            header_bytes: 0,
            data_start: 0,
            metadata: BTreeMap::new(),
            tensors,
        };
        let plan = index.plan(4).unwrap();
        assert_eq!(plan.candidates.len(), LinearFormat::ALL.len());
        assert!(
            plan.candidates
                .iter()
                .all(|candidate| candidate.deployment_bytes > 0)
        );
        assert!(
            plan.candidates
                .iter()
                .find(|candidate| candidate.format.starts_with("packed_q4_k"))
                .unwrap()
                .fallback_tensors
                > 0
        );
    }

    fn temporary_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("parakeet-l4-{nonce}-{name}"))
    }
}
