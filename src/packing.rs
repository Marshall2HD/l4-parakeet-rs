use crate::exl3::Exl3Layout;
use serde::Serialize;
use thiserror::Error;

pub const TILE_ALIGNMENT: usize = 128;
const Q4_K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearFormat {
    PackedFp16,
    PackedFp8E4m3,
    PackedInt8,
    GgmlQ4K,
    Exl3Derived,
}

impl LinearFormat {
    pub const ALL: [Self; 5] = [
        Self::PackedFp16,
        Self::PackedFp8E4m3,
        Self::PackedInt8,
        Self::GgmlQ4K,
        Self::Exl3Derived,
    ];

    pub fn label(self, exl3_bits: u8) -> String {
        match self {
            Self::PackedFp16 => "packed_fp16".into(),
            Self::PackedFp8E4m3 => "packed_fp8_e4m3".into(),
            Self::PackedInt8 => "packed_int8".into(),
            Self::GgmlQ4K => "packed_q4_k_with_fp16_fallback".into(),
            Self::Exl3Derived => format!("exl3_derived_{exl3_bits}bpw"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearLayout {
    pub logical_in: usize,
    pub logical_out: usize,
    pub padded_in: usize,
    pub padded_out: usize,
    pub payload_bytes: usize,
    pub metadata_bytes: usize,
}

impl LinearLayout {
    pub fn total_bytes(self) -> Option<usize> {
        self.payload_bytes.checked_add(self.metadata_bytes)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PackingError {
    #[error("linear dimensions must be nonzero")]
    Empty,
    #[error("packed byte count overflow")]
    Overflow,
    #[error(transparent)]
    Exl3(#[from] crate::exl3::Exl3LayoutError),
}

/// Estimate the exact weight bytes required by one candidate runtime layout.
///
/// `None` means that an existing GGUF Q4_K row cannot be consumed as-is because
/// its contraction dimension is not divisible by the 256-element superblock.
/// The caller must retain a non-Q4 fallback rather than silently repacking a
/// different tensor and calling it the same comparison.
pub fn estimate_linear(
    format: LinearFormat,
    logical_in: usize,
    logical_out: usize,
    exl3_bits: u8,
) -> Result<Option<LinearLayout>, PackingError> {
    if logical_in == 0 || logical_out == 0 {
        return Err(PackingError::Empty);
    }

    match format {
        LinearFormat::PackedFp16 => {
            let (padded_in, padded_out) = tiled_shape(logical_in, logical_out);
            Ok(Some(layout(
                logical_in,
                logical_out,
                padded_in,
                padded_out,
                checked_product(padded_in, padded_out, 2)?,
                0,
            )))
        }
        LinearFormat::PackedFp8E4m3 | LinearFormat::PackedInt8 => {
            let (padded_in, padded_out) = tiled_shape(logical_in, logical_out);
            Ok(Some(layout(
                logical_in,
                logical_out,
                padded_in,
                padded_out,
                checked_product(padded_in, padded_out, 1)?,
                padded_out.checked_mul(4).ok_or(PackingError::Overflow)?,
            )))
        }
        LinearFormat::GgmlQ4K => {
            if !logical_in.is_multiple_of(Q4_K_BLOCK_ELEMENTS) {
                return Ok(None);
            }
            let blocks_per_row = logical_in / Q4_K_BLOCK_ELEMENTS;
            Ok(Some(layout(
                logical_in,
                logical_out,
                logical_in,
                logical_out,
                checked_product(logical_out, blocks_per_row, Q4_K_BLOCK_BYTES)?,
                0,
            )))
        }
        LinearFormat::Exl3Derived => {
            let estimate = Exl3Layout::new(logical_in, logical_out, exl3_bits)?;
            Ok(Some(layout(
                logical_in,
                logical_out,
                estimate.padded_in,
                estimate.padded_out,
                estimate.trellis_bytes(),
                estimate.scale_bytes(),
            )))
        }
    }
}

fn tiled_shape(logical_in: usize, logical_out: usize) -> (usize, usize) {
    (
        logical_in.next_multiple_of(TILE_ALIGNMENT),
        logical_out.next_multiple_of(TILE_ALIGNMENT),
    )
}

fn checked_product(a: usize, b: usize, c: usize) -> Result<usize, PackingError> {
    a.checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(PackingError::Overflow)
}

fn layout(
    logical_in: usize,
    logical_out: usize,
    padded_in: usize,
    padded_out: usize,
    payload_bytes: usize,
    metadata_bytes: usize,
) -> LinearLayout {
    LinearLayout {
        logical_in,
        logical_out,
        padded_in,
        padded_out,
        payload_bytes,
        metadata_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_native_l4_layouts_for_an_ffn_matrix() {
        let fp16 = estimate_linear(LinearFormat::PackedFp16, 1024, 4096, 4)
            .unwrap()
            .unwrap();
        let fp8 = estimate_linear(LinearFormat::PackedFp8E4m3, 1024, 4096, 4)
            .unwrap()
            .unwrap();
        assert_eq!(fp16.total_bytes(), Some(8 * 1024 * 1024));
        assert_eq!(fp8.payload_bytes, 4 * 1024 * 1024);
        assert_eq!(fp8.metadata_bytes, 16 * 1024);
    }

    #[test]
    fn q4_k_reports_unsupported_rows_instead_of_changing_the_format() {
        assert!(
            estimate_linear(LinearFormat::GgmlQ4K, 640, 8198, 4)
                .unwrap()
                .is_none()
        );
        let q4 = estimate_linear(LinearFormat::GgmlQ4K, 1024, 4096, 4)
            .unwrap()
            .unwrap();
        assert_eq!(q4.total_bytes(), Some(4096 * 4 * 144));
    }

    #[test]
    fn pads_the_joint_head_for_native_tensor_core_tiles() {
        let fp8 = estimate_linear(LinearFormat::PackedFp8E4m3, 640, 8198, 4)
            .unwrap()
            .unwrap();
        assert_eq!((fp8.padded_in, fp8.padded_out), (640, 8320));
    }
}
