use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const HADAMARD_BLOCK: usize = 128;
pub const TRELLIS_TILE: usize = 16;
pub const OUTPUT_ALIGNMENT: usize = 128;

/// Byte and padding estimate for an EXL3-derived candidate.
///
/// This deliberately is not a serialized deployment format. EXL3's public
/// format is still evolving, and this project will not freeze a Parakeet
/// variant until L4 benchmarks and ASR calibration show that it should ship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exl3Layout {
    pub logical_in: usize,
    pub logical_out: usize,
    pub padded_in: usize,
    pub padded_out: usize,
    pub qbits: u8,
    pub codebook: Codebook,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Codebook {
    Mul1,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Exl3LayoutError {
    #[error("EXL3 supports integer rates from 1 through 8 bpw, got {0}")]
    InvalidBitRate(u8),
    #[error("linear dimensions must be nonzero")]
    Empty,
}

impl Exl3Layout {
    pub fn new(logical_in: usize, logical_out: usize, qbits: u8) -> Result<Self, Exl3LayoutError> {
        if logical_in == 0 || logical_out == 0 {
            return Err(Exl3LayoutError::Empty);
        }
        if !(1..=8).contains(&qbits) {
            return Err(Exl3LayoutError::InvalidBitRate(qbits));
        }

        Ok(Self {
            logical_in,
            logical_out,
            padded_in: logical_in.next_multiple_of(HADAMARD_BLOCK),
            padded_out: logical_out.next_multiple_of(OUTPUT_ALIGNMENT),
            qbits,
            codebook: Codebook::Mul1,
        })
    }

    pub fn trellis_shape(&self) -> [usize; 3] {
        [
            self.padded_in / TRELLIS_TILE,
            self.padded_out / TRELLIS_TILE,
            TRELLIS_TILE * usize::from(self.qbits),
        ]
    }

    pub fn trellis_bytes(&self) -> usize {
        self.padded_in * self.padded_out * usize::from(self.qbits) / 8
    }

    pub fn scale_bytes(&self) -> usize {
        2 * (self.padded_in + self.padded_out)
    }

    pub fn packed_bytes(&self) -> usize {
        self.trellis_bytes() + self.scale_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_ffn_shape_has_exact_four_bit_payload() {
        let linear = Exl3Layout::new(4096, 1024, 4).unwrap();
        assert_eq!(linear.trellis_shape(), [256, 64, 64]);
        assert_eq!(linear.trellis_bytes(), 2 * 1024 * 1024);
        assert_eq!(linear.scale_bytes(), 10 * 1024);
    }

    #[test]
    fn joint_head_is_padded_without_changing_logical_shape() {
        let linear = Exl3Layout::new(640, 8198, 4).unwrap();
        assert_eq!(linear.padded_in, 640);
        assert_eq!(linear.padded_out, 8320);
        assert_eq!(linear.trellis_shape(), [40, 520, 64]);
    }
}
