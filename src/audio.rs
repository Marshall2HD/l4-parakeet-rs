use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid WAV: {0}")]
    Invalid(String),
}

pub fn read_pcm16_mono(path: &Path) -> Result<Audio, AudioError> {
    let bytes = fs::read(path).map_err(|source| AudioError::Read {
        path: path.to_owned(),
        source,
    })?;
    decode_pcm16_mono(&bytes)
}

pub fn decode_pcm16_mono(bytes: &[u8]) -> Result<Audio, AudioError> {
    let mut samples = Vec::new();
    let sample_rate = decode_pcm16_mono_into(bytes, &mut samples)?;
    Ok(Audio {
        samples,
        sample_rate,
    })
}

pub fn decode_pcm16_mono_into(bytes: &[u8], samples: &mut Vec<f32>) -> Result<u32, AudioError> {
    let (data, sample_rate) = pcm16_mono_data(bytes)?;
    samples.clear();
    samples.extend(
        data.chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0),
    );
    Ok(sample_rate)
}

/// Validate the WAV contract and borrow its little-endian PCM16 payload.
pub fn pcm16_mono_data(bytes: &[u8]) -> Result<(&[u8], u32), AudioError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::Invalid("expected RIFF/WAVE header".into()));
    }

    let declared = usize::try_from(read_u32(bytes, 4)?)
        .map_err(|_| AudioError::Invalid("RIFF length does not fit usize".into()))?
        .checked_add(8)
        .ok_or_else(|| AudioError::Invalid("RIFF length overflow".into()))?;
    if declared > bytes.len() {
        return Err(AudioError::Invalid(format!(
            "RIFF declares {declared} bytes, file has {}",
            bytes.len()
        )));
    }

    let mut format = None;
    let mut data = None;
    let mut cursor = 12usize;
    while cursor + 8 <= declared {
        let chunk = &bytes[cursor..cursor + 4];
        let chunk_bytes = usize::try_from(read_u32(bytes, cursor + 4)?)
            .map_err(|_| AudioError::Invalid("chunk length does not fit usize".into()))?;
        let start = cursor + 8;
        let end = start
            .checked_add(chunk_bytes)
            .ok_or_else(|| AudioError::Invalid("chunk range overflow".into()))?;
        if end > declared {
            return Err(AudioError::Invalid(format!(
                "chunk {:?} ends beyond RIFF payload",
                String::from_utf8_lossy(chunk)
            )));
        }
        match chunk {
            b"fmt " => format = Some(&bytes[start..end]),
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end
            .checked_add(chunk_bytes & 1)
            .ok_or_else(|| AudioError::Invalid("chunk padding overflow".into()))?;
    }

    let format = format.ok_or_else(|| AudioError::Invalid("missing fmt chunk".into()))?;
    if format.len() < 16 {
        return Err(AudioError::Invalid(
            "fmt chunk is shorter than 16 bytes".into(),
        ));
    }
    let encoding = read_u16(format, 0)?;
    let channels = read_u16(format, 2)?;
    let sample_rate = read_u32(format, 4)?;
    let block_alignment = read_u16(format, 12)?;
    let bits_per_sample = read_u16(format, 14)?;
    if encoding != 1 || channels != 1 || bits_per_sample != 16 || block_alignment != 2 {
        return Err(AudioError::Invalid(format!(
            "expected PCM s16le mono (format=1, channels=1, bits=16, block_align=2), got format={encoding}, channels={channels}, bits={bits_per_sample}, block_align={block_alignment}"
        )));
    }

    let data = data.ok_or_else(|| AudioError::Invalid("missing data chunk".into()))?;
    if !data.len().is_multiple_of(2) {
        return Err(AudioError::Invalid(
            "PCM payload has an odd byte count".into(),
        ));
    }
    Ok((data, sample_rate))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, AudioError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| AudioError::Invalid("truncated 16-bit field".into()))?;
    Ok(u16::from_le_bytes(
        value.try_into().expect("two-byte field"),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, AudioError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| AudioError::Invalid("truncated 32-bit field".into()))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("four-byte field"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_fixed_pcm_contract() {
        let samples = [-32768i16, 0, 16384, 32767];
        let data = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);

        let audio = decode_pcm16_mono(&wav).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.samples, [-1.0, 0.0, 0.5, 32767.0 / 32768.0]);
    }

    #[test]
    fn reuses_samples_with_exact_pcm_values_and_no_stale_tail() {
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&(36 + 65_536 * 2u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(65_536 * 2u32).to_le_bytes());
        for sample in i16::MIN..=i16::MAX {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        let mut samples = Vec::new();
        assert_eq!(decode_pcm16_mono_into(&wav, &mut samples).unwrap(), 16_000);
        assert_eq!(samples.len(), 65_536);
        for (actual, expected) in samples.iter().zip(i16::MIN..=i16::MAX) {
            assert_eq!(actual.to_bits(), (expected as f32 / 32768.0).to_bits());
        }
        let allocation = samples.as_ptr();
        wav.truncate(46);
        wav[4..8].copy_from_slice(&38u32.to_le_bytes());
        wav[40..44].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(decode_pcm16_mono_into(&wav, &mut samples).unwrap(), 16_000);
        assert_eq!(samples, [-1.0]);
        assert_eq!(samples.as_ptr(), allocation);
    }

    #[test]
    fn rejects_stereo_instead_of_silently_downmixing() {
        let mut wav = b"RIFF$\0\0\0WAVEfmt \x10\0\0\0\x01\0\x02\0".to_vec();
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&64_000u32.to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data\0\0\0\0");
        let error = decode_pcm16_mono(&wav).unwrap_err().to_string();
        assert!(error.contains("expected PCM s16le mono"));
    }
}
