use crate::config::ModelProfile;
use crate::weights::{DType, SafeTensorIndex, TensorFamily, WeightError, classify, linear_shape};
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const MAGIC: &[u8; 8] = b"PKL4AOT\0";
const SCHEMA_VERSION: u32 = 2;
const PREFIX_BYTES: u64 = 24;
const HEADER_ALIGNMENT: u64 = 4096;
const TENSOR_ALIGNMENT: u64 = 256;
const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AotStorage {
    Fp16,
    Fp32,
    Sm89Fp16Linear,
    Sm89Fp32Bias,
}

impl AotStorage {
    fn element_bytes(self) -> u64 {
        match self {
            Self::Fp16 | Self::Sm89Fp16Linear => 2,
            Self::Fp32 | Self::Sm89Fp32Bias => 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AotTensor {
    pub name: String,
    pub storage: AotStorage,
    pub logical_shape: Vec<usize>,
    pub physical_shape: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AotHeader {
    pub schema_version: u32,
    pub target: String,
    pub profile: ModelProfile,
    pub source_repository: String,
    pub source_revision: String,
    pub source_payload_bytes: u64,
    pub precision_plan: String,
    pub tensor_alignment: u64,
    pub payload_bytes: u64,
    pub vocabulary: Vec<String>,
    pub tensors: Vec<AotTensor>,
}

#[derive(Debug, Serialize)]
pub struct PackReport {
    pub output: PathBuf,
    pub profile: ModelProfile,
    pub precision_plan: &'static str,
    pub tensors: usize,
    pub payload_bytes: u64,
    pub file_bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Serialize)]
pub struct AotArtifactIndex {
    pub path: PathBuf,
    pub file_bytes: u64,
    pub data_start: u64,
    pub header: AotHeader,
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid L4 artifact: {0}")]
    Invalid(String),
    #[error("invalid L4 artifact header: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Weights(#[from] WeightError),
}

impl AotArtifactIndex {
    pub fn open(path: &Path) -> Result<Self, ArtifactError> {
        let mut file = open_file(path)?;
        let file_bytes = file
            .metadata()
            .map_err(|source| io_error(path, source))?
            .len();
        if file_bytes < PREFIX_BYTES {
            return Err(ArtifactError::Invalid(format!(
                "file has {file_bytes} bytes; prefix requires {PREFIX_BYTES}"
            )));
        }

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|source| io_error(path, source))?;
        if &magic != MAGIC {
            return Err(ArtifactError::Invalid("bad magic".into()));
        }
        let version = read_u32(&mut file, path)?;
        let flags = read_u32(&mut file, path)?;
        let header_bytes = read_u64(&mut file, path)?;
        if version != SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(format!(
                "expected schema version {SCHEMA_VERSION}, got {version}"
            )));
        }
        if flags != 0 {
            return Err(ArtifactError::Invalid(format!(
                "unsupported prefix flags {flags:#x}"
            )));
        }
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(ArtifactError::Invalid(format!(
                "header length {header_bytes} is outside 1..={MAX_HEADER_BYTES}"
            )));
        }
        let header_end = PREFIX_BYTES
            .checked_add(header_bytes)
            .ok_or_else(|| ArtifactError::Invalid("header offset overflow".into()))?;
        let data_start = align_up(header_end, HEADER_ALIGNMENT)?;
        if data_start > file_bytes {
            return Err(ArtifactError::Invalid(format!(
                "aligned header ends at {data_start}, beyond file length {file_bytes}"
            )));
        }

        let mut encoded = vec![0u8; usize_from_u64(header_bytes, "header length")?];
        file.read_exact(&mut encoded)
            .map_err(|source| io_error(path, source))?;
        let header: AotHeader = serde_json::from_slice(&encoded)?;
        validate_header(&header, file_bytes - data_start)?;

        Ok(Self {
            path: path.to_owned(),
            file_bytes,
            data_start,
            header,
        })
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, ArtifactError> {
        let tensor = self
            .header
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| ArtifactError::Invalid(format!("missing tensor {name}")))?;
        if tensor.storage != AotStorage::Fp32 || tensor.logical_shape != tensor.physical_shape {
            return Err(ArtifactError::Invalid(format!(
                "{name}: tensor is not plain FP32 storage"
            )));
        }
        let mut file = open_file(&self.path)?;
        file.seek(SeekFrom::Start(self.data_start + tensor.offset))
            .map_err(|source| io_error(&self.path, source))?;
        let mut encoded = vec![0u8; usize_from_u64(tensor.bytes, name)?];
        file.read_exact(&mut encoded)
            .map_err(|source| io_error(&self.path, source))?;
        encoded
            .chunks_exact(4)
            .enumerate()
            .map(|(index, bytes)| {
                let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
                value.is_finite().then_some(value).ok_or_else(|| {
                    ArtifactError::Invalid(format!(
                        "{name}: non-finite FP32 value at element {index}"
                    ))
                })
            })
            .collect()
    }
}

pub fn pack_fp16(
    source: &SafeTensorIndex,
    profile: ModelProfile,
    vocabulary: &[String],
    output: &Path,
) -> Result<PackReport, ArtifactError> {
    source.validate_model(profile)?;
    if output.exists() {
        return Err(ArtifactError::OutputExists(output.to_owned()));
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
    }

    let header = plan_fp16(source, profile, vocabulary)?;
    let encoded_header = serde_json::to_vec(&header)?;
    let header_bytes = u64::try_from(encoded_header.len())
        .map_err(|_| ArtifactError::Invalid("header length overflow".into()))?;
    let data_start = align_up(
        PREFIX_BYTES
            .checked_add(header_bytes)
            .ok_or_else(|| ArtifactError::Invalid("header offset overflow".into()))?,
        HEADER_ALIGNMENT,
    )?;
    let temporary = temporary_path(output)?;

    let result = write_fp16_artifact(source, &header, &encoded_header, data_start, &temporary)
        .and_then(|()| {
            AotArtifactIndex::open(&temporary)?;
            fs::rename(&temporary, output).map_err(|source| io_error(output, source))?;
            Ok(())
        });
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;

    let file_bytes = fs::metadata(output)
        .map_err(|source| io_error(output, source))?
        .len();
    Ok(PackReport {
        output: output.to_owned(),
        profile,
        precision_plan: "fp16",
        tensors: header.tensors.len(),
        payload_bytes: header.payload_bytes,
        file_bytes,
        blake3: digest_file(output)?,
    })
}

fn plan_fp16(
    source: &SafeTensorIndex,
    profile: ModelProfile,
    vocabulary: &[String],
) -> Result<AotHeader, ArtifactError> {
    validate_vocabulary(vocabulary, profile)?;
    let mut tensors = Vec::with_capacity(source.tensors.len());
    let mut cursor = 0u64;

    for (name, tensor) in &source.tensors {
        let family = classify(name);
        if family == TensorFamily::Counter {
            continue;
        }
        if tensor.dtype != DType::F32 {
            return Err(ArtifactError::Invalid(format!(
                "{name}: FP16 plan requires F32 source, got {:?}",
                tensor.dtype
            )));
        }

        let matching_linear = matching_linear_weight(name).and_then(|weight| {
            classify(&weight)
                .is_candidate_linear()
                .then(|| source.tensors.get(&weight).map(|tensor| (weight, tensor)))
                .flatten()
        });
        let (storage, logical_shape, physical_shape) = if family.is_candidate_linear() {
            let (logical_out, logical_in) = linear_shape(&tensor.shape).ok_or_else(|| {
                ArtifactError::Invalid(format!(
                    "{name}: candidate linear has unsupported shape {:?}",
                    tensor.shape
                ))
            })?;
            (
                AotStorage::Sm89Fp16Linear,
                vec![logical_out, logical_in],
                vec![
                    logical_out.next_multiple_of(128),
                    logical_in.next_multiple_of(128),
                ],
            )
        } else if let Some((weight_name, weight)) = matching_linear {
            let (logical_out, _) = linear_shape(&weight.shape).ok_or_else(|| {
                ArtifactError::Invalid(format!(
                    "{weight_name}: matching linear has unsupported shape {:?}",
                    weight.shape
                ))
            })?;
            if tensor.shape != [logical_out] {
                return Err(ArtifactError::Invalid(format!(
                    "{name}: bias shape {:?} does not match {weight_name} output {logical_out}",
                    tensor.shape
                )));
            }
            (
                AotStorage::Sm89Fp32Bias,
                tensor.shape.clone(),
                vec![logical_out.next_multiple_of(128)],
            )
        } else if family == TensorFamily::SensitiveFp32 || name.starts_with("frontend.") {
            (AotStorage::Fp32, tensor.shape.clone(), tensor.shape.clone())
        } else {
            (AotStorage::Fp16, tensor.shape.clone(), tensor.shape.clone())
        };

        cursor = align_up(cursor, TENSOR_ALIGNMENT)?;
        let elements = checked_elements(&physical_shape, name)?;
        let bytes = elements
            .checked_mul(storage.element_bytes())
            .ok_or_else(|| ArtifactError::Invalid(format!("{name}: byte count overflow")))?;
        tensors.push(AotTensor {
            name: name.clone(),
            storage,
            logical_shape,
            physical_shape,
            offset: cursor,
            bytes,
        });
        cursor = cursor
            .checked_add(bytes)
            .ok_or_else(|| ArtifactError::Invalid("payload size overflow".into()))?;
    }

    Ok(AotHeader {
        schema_version: SCHEMA_VERSION,
        target: "sm_89".into(),
        profile,
        source_repository: profile.repository().into(),
        source_revision: profile.revision().into(),
        source_payload_bytes: source.file_bytes - source.data_start,
        precision_plan: "fp16".into(),
        tensor_alignment: TENSOR_ALIGNMENT,
        payload_bytes: cursor,
        vocabulary: vocabulary.to_vec(),
        tensors,
    })
}

fn write_fp16_artifact(
    source: &SafeTensorIndex,
    header: &AotHeader,
    encoded_header: &[u8],
    data_start: u64,
    output: &Path,
) -> Result<(), ArtifactError> {
    let mut input = open_file(&source.path)?;
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|source| io_error(output, source))?;

    destination
        .write_all(MAGIC)
        .and_then(|()| destination.write_all(&SCHEMA_VERSION.to_le_bytes()))
        .and_then(|()| destination.write_all(&0u32.to_le_bytes()))
        .and_then(|()| destination.write_all(&(encoded_header.len() as u64).to_le_bytes()))
        .and_then(|()| destination.write_all(encoded_header))
        .map_err(|source| io_error(output, source))?;

    for record in &header.tensors {
        let source_tensor = &source.tensors[&record.name];
        input
            .seek(SeekFrom::Start(
                source.data_start + source_tensor.data_offsets[0],
            ))
            .map_err(|error| io_error(&source.path, error))?;
        let mut source_bytes = vec![0u8; usize_from_u64(source_tensor.bytes(), &record.name)?];
        input
            .read_exact(&mut source_bytes)
            .map_err(|error| io_error(&source.path, error))?;
        let packed = match record.storage {
            AotStorage::Fp32 => {
                validate_f32(&source_bytes, &record.name)?;
                source_bytes
            }
            AotStorage::Fp16 => encode_fp16(&source_bytes, &record.name)?,
            AotStorage::Sm89Fp16Linear => encode_fp16_linear(
                &source_bytes,
                record.logical_shape[0],
                record.logical_shape[1],
                record.physical_shape[0],
                record.physical_shape[1],
                &record.name,
            )?,
            AotStorage::Sm89Fp32Bias => encode_fp32_bias(
                &source_bytes,
                record.logical_shape[0],
                record.physical_shape[0],
                &record.name,
            )?,
        };
        if u64::try_from(packed.len()).ok() != Some(record.bytes) {
            return Err(ArtifactError::Invalid(format!(
                "{}: planned {} bytes, encoded {}",
                record.name,
                record.bytes,
                packed.len()
            )));
        }
        destination
            .seek(SeekFrom::Start(data_start + record.offset))
            .and_then(|_| destination.write_all(&packed))
            .map_err(|source| io_error(output, source))?;
    }

    destination
        .set_len(data_start + header.payload_bytes)
        .and_then(|()| destination.sync_all())
        .map_err(|source| io_error(output, source))
}

fn encode_fp16(source: &[u8], name: &str) -> Result<Vec<u8>, ArtifactError> {
    if !source.len().is_multiple_of(4) {
        return Err(ArtifactError::Invalid(format!(
            "{name}: F32 source byte count {} is not divisible by four",
            source.len()
        )));
    }
    let mut packed = Vec::with_capacity(source.len() / 2);
    for (index, bytes) in source.chunks_exact(4).enumerate() {
        let value = decode_f32(bytes, name, index)?;
        packed.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(packed)
}

fn encode_fp16_linear(
    source: &[u8],
    logical_out: usize,
    logical_in: usize,
    padded_out: usize,
    padded_in: usize,
    name: &str,
) -> Result<Vec<u8>, ArtifactError> {
    let logical_elements = logical_out
        .checked_mul(logical_in)
        .ok_or_else(|| ArtifactError::Invalid(format!("{name}: logical shape overflow")))?;
    if source.len() != logical_elements * 4 {
        return Err(ArtifactError::Invalid(format!(
            "{name}: shape requires {} F32 bytes, source has {}",
            logical_elements * 4,
            source.len()
        )));
    }
    let padded_elements = padded_out
        .checked_mul(padded_in)
        .ok_or_else(|| ArtifactError::Invalid(format!("{name}: padded shape overflow")))?;
    let mut packed = vec![0u8; padded_elements * 2];
    for row in 0..logical_out {
        for column in 0..logical_in {
            let source_index = row * logical_in + column;
            let source_offset = source_index * 4;
            let value = decode_f32(
                &source[source_offset..source_offset + 4],
                name,
                source_index,
            )?;
            let target_offset = (row * padded_in + column) * 2;
            packed[target_offset..target_offset + 2]
                .copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    Ok(packed)
}

fn encode_fp32_bias(
    source: &[u8],
    logical: usize,
    padded: usize,
    name: &str,
) -> Result<Vec<u8>, ArtifactError> {
    if source.len() != logical * 4 || padded < logical {
        return Err(ArtifactError::Invalid(format!(
            "{name}: invalid FP32 bias source/padded lengths"
        )));
    }
    validate_f32(source, name)?;
    let mut packed = vec![0u8; padded * 4];
    packed[..source.len()].copy_from_slice(source);
    Ok(packed)
}

fn matching_linear_weight(name: &str) -> Option<String> {
    name.strip_suffix(".bias")
        .map(|prefix| format!("{prefix}.weight"))
}

fn validate_f32(source: &[u8], name: &str) -> Result<(), ArtifactError> {
    if !source.len().is_multiple_of(4) {
        return Err(ArtifactError::Invalid(format!(
            "{name}: F32 source byte count {} is not divisible by four",
            source.len()
        )));
    }
    for (index, bytes) in source.chunks_exact(4).enumerate() {
        decode_f32(bytes, name, index)?;
    }
    Ok(())
}

fn decode_f32(bytes: &[u8], name: &str, index: usize) -> Result<f32, ArtifactError> {
    let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    if !value.is_finite() {
        return Err(ArtifactError::Invalid(format!(
            "{name}: non-finite F32 value at element {index}"
        )));
    }
    Ok(value)
}

fn validate_header(header: &AotHeader, payload_bytes: u64) -> Result<(), ArtifactError> {
    if header.schema_version != SCHEMA_VERSION {
        return Err(ArtifactError::Invalid(format!(
            "header schema is {}, expected {SCHEMA_VERSION}",
            header.schema_version
        )));
    }
    if header.target != "sm_89" {
        return Err(ArtifactError::Invalid(format!(
            "target is {:?}, expected sm_89",
            header.target
        )));
    }
    if header.source_repository != header.profile.repository()
        || header.source_revision != header.profile.revision()
    {
        return Err(ArtifactError::Invalid(
            "profile does not match source repository and revision".into(),
        ));
    }
    if header.tensor_alignment != TENSOR_ALIGNMENT {
        return Err(ArtifactError::Invalid(format!(
            "tensor alignment is {}, expected {TENSOR_ALIGNMENT}",
            header.tensor_alignment
        )));
    }
    if header.payload_bytes != payload_bytes {
        return Err(ArtifactError::Invalid(format!(
            "header declares {} payload bytes, file contains {payload_bytes}",
            header.payload_bytes
        )));
    }
    validate_vocabulary(&header.vocabulary, header.profile)?;

    let mut names = BTreeSet::new();
    let mut cursor = 0u64;
    for tensor in &header.tensors {
        if tensor.name.is_empty() || !names.insert(&tensor.name) {
            return Err(ArtifactError::Invalid(format!(
                "empty or duplicate tensor name {:?}",
                tensor.name
            )));
        }
        if tensor.offset % TENSOR_ALIGNMENT != 0 || tensor.offset < cursor {
            return Err(ArtifactError::Invalid(format!(
                "{}: invalid aligned offset {} after {cursor}",
                tensor.name, tensor.offset
            )));
        }
        validate_tensor_layout(tensor)?;
        let elements = checked_elements(&tensor.physical_shape, &tensor.name)?;
        let expected_bytes = elements
            .checked_mul(tensor.storage.element_bytes())
            .ok_or_else(|| {
                ArtifactError::Invalid(format!("{}: byte count overflow", tensor.name))
            })?;
        if tensor.bytes != expected_bytes {
            return Err(ArtifactError::Invalid(format!(
                "{}: physical shape and storage require {expected_bytes} bytes, header declares {}",
                tensor.name, tensor.bytes
            )));
        }
        cursor = tensor
            .offset
            .checked_add(tensor.bytes)
            .ok_or_else(|| ArtifactError::Invalid(format!("{}: range overflow", tensor.name)))?;
        if cursor > payload_bytes {
            return Err(ArtifactError::Invalid(format!(
                "{}: range ends at {cursor}, beyond {payload_bytes}-byte payload",
                tensor.name
            )));
        }
    }
    if cursor != payload_bytes {
        return Err(ArtifactError::Invalid(format!(
            "last tensor ends at {cursor}, payload ends at {payload_bytes}"
        )));
    }
    Ok(())
}

fn validate_vocabulary(vocabulary: &[String], profile: ModelProfile) -> Result<(), ArtifactError> {
    let expected = profile.blank_token_id();
    if vocabulary.len() != expected {
        return Err(ArtifactError::Invalid(format!(
            "{profile} requires {expected} vocabulary entries before the blank token, got {}",
            vocabulary.len()
        )));
    }
    if let Some(index) = vocabulary
        .iter()
        .position(|piece| piece.is_empty() || piece.contains(['\n', '\r', '\t']))
    {
        return Err(ArtifactError::Invalid(format!(
            "vocabulary entry {index} is empty or contains control separators"
        )));
    }
    Ok(())
}

fn validate_tensor_layout(tensor: &AotTensor) -> Result<(), ArtifactError> {
    let aligned = |dimension: usize| -> Result<usize, ArtifactError> {
        usize_from_u64(align_up(dimension as u64, 128)?, &tensor.name)
    };
    match tensor.storage {
        AotStorage::Fp16 | AotStorage::Fp32 => {
            if tensor.physical_shape != tensor.logical_shape {
                return Err(ArtifactError::Invalid(format!(
                    "{}: plain storage cannot change logical shape {:?} to {:?}",
                    tensor.name, tensor.logical_shape, tensor.physical_shape
                )));
            }
        }
        AotStorage::Sm89Fp16Linear => match tensor.logical_shape.as_slice() {
            [output, input]
                if *output > 0
                    && *input > 0
                    && tensor.physical_shape == [aligned(*output)?, aligned(*input)?] => {}
            _ => {
                return Err(ArtifactError::Invalid(format!(
                    "{}: invalid sm_89 FP16 linear shapes {:?} -> {:?}",
                    tensor.name, tensor.logical_shape, tensor.physical_shape
                )));
            }
        },
        AotStorage::Sm89Fp32Bias => match tensor.logical_shape.as_slice() {
            [output] if *output > 0 && tensor.physical_shape == [aligned(*output)?] => {}
            _ => {
                return Err(ArtifactError::Invalid(format!(
                    "{}: invalid sm_89 FP32 bias shapes {:?} -> {:?}",
                    tensor.name, tensor.logical_shape, tensor.physical_shape
                )));
            }
        },
    }
    Ok(())
}

fn checked_elements(shape: &[usize], name: &str) -> Result<u64, ArtifactError> {
    shape.iter().try_fold(1u64, |elements, dimension| {
        elements
            .checked_mul(*dimension as u64)
            .ok_or_else(|| ArtifactError::Invalid(format!("{name}: element count overflow")))
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ArtifactError> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| ArtifactError::Invalid("alignment overflow".into()))
}

fn usize_from_u64(value: u64, description: &str) -> Result<usize, ArtifactError> {
    usize::try_from(value)
        .map_err(|_| ArtifactError::Invalid(format!("{description}: size does not fit usize")))
}

fn temporary_path(output: &Path) -> Result<PathBuf, ArtifactError> {
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ArtifactError::Invalid("output has no UTF-8 file name".into()))?;
    Ok(output.with_file_name(format!(".{file_name}.{}.tmp", std::process::id())))
}

fn open_file(path: &Path) -> Result<File, ArtifactError> {
    File::open(path).map_err(|source| io_error(path, source))
}

fn io_error(path: &Path, source: std::io::Error) -> ArtifactError {
    ArtifactError::Io {
        path: path.to_owned(),
        source,
    }
}

fn read_u32(file: &mut File, path: &Path) -> Result<u32, ArtifactError> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File, path: &Path) -> Result<u64, ArtifactError> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    Ok(u64::from_le_bytes(bytes))
}

fn digest_file(path: &Path) -> Result<String, ArtifactError> {
    let mut file = open_file(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io_error(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_fp16_linear_rows_for_sm89_tiles() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let source = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let packed = encode_fp16_linear(&source, 2, 3, 4, 4, "test").unwrap();
        let values = packed
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            [
                1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn pads_fp32_linear_bias_for_the_same_output_tiles() {
        let source = [1.25f32, -2.5]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let packed = encode_fp32_bias(&source, 2, 4, "test.bias").unwrap();
        let values = packed
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [1.25, -2.5, 0.0, 0.0]);
    }

    #[test]
    fn rejects_non_finite_source_values() {
        let error = encode_fp16(&f32::NAN.to_le_bytes(), "bad").unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn alignment_is_checked_for_overflow() {
        assert_eq!(align_up(257, 256).unwrap(), 512);
        assert!(align_up(u64::MAX, 256).is_err());
    }

    #[test]
    fn validates_profile_vocabulary_contract() {
        let vocabulary = vec!["piece".to_owned(); ModelProfile::V2English.blank_token_id()];
        validate_vocabulary(&vocabulary, ModelProfile::V2English).unwrap();

        let error = validate_vocabulary(&vocabulary[..1_023], ModelProfile::V2English)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires 1024"));
    }
}
