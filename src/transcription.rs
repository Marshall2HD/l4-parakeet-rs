//! Offline V2 result metadata. TDT offsets are model predictions, not forced alignment.
use serde::Serialize;

/// Stay below the long-form GEMM/FFN policies; batch attention remains FP16.
pub const MAX_BATCH_ROWS: usize = 1008;
pub const MAX_BATCH_ITEMS: usize = 16;

#[derive(Debug)]
pub struct BatchLayout {
    pub offsets: Vec<usize>,
    pub valid_frames: Vec<usize>,
    /// Per packed row: inclusive start and exclusive end of its utterance.
    /// Padding uses an empty interval. Sixteen guard rows isolate convolution.
    pub bounds: Vec<i32>,
}

impl BatchLayout {
    pub fn new(sample_counts: &[usize]) -> Result<Self, &'static str> {
        if sample_counts.is_empty() || sample_counts.len() > MAX_BATCH_ITEMS {
            return Err("batch must contain between 1 and 16 inputs");
        }
        let mut result = Self {
            offsets: Vec::new(),
            valid_frames: Vec::new(),
            bounds: Vec::new(),
        };
        for &samples in sample_counts {
            if samples == 0 {
                return Err("batch audio must not be empty");
            }
            let frames = (samples / 160 + 1).div_ceil(8);
            let valid = (samples / 160).div_ceil(8);
            if frames.next_multiple_of(16) >= 512 {
                return Err(
                    "multi-input batches require each clip to fit the short (<512 padded encoder rows) precision policy; transcribe long inputs individually",
                );
            }
            let offset = result.rows();
            let end = offset + frames.next_multiple_of(16) + 16;
            if end > MAX_BATCH_ROWS {
                return Err(
                    "packed batch exceeds 1008 encoder rows including padding and guards; split the batch",
                );
            }
            result.offsets.push(offset);
            result.valid_frames.push(valid);
            result.bounds.resize(end * 2, 0);
            for row in offset..offset + valid {
                result.bounds[row * 2] = offset as i32;
                result.bounds[row * 2 + 1] = (offset + valid) as i32;
            }
        }
        Ok(result)
    }

    pub fn rows(&self) -> usize {
        self.bounds.len() / 2
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenTimestamp {
    pub id: i32,
    pub piece: String,
    pub start_frame: i32,
    pub duration_frames: i32,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// SentencePiece boundaries start words; closing punctuation stays with its word.
/// Zero-duration predictions remain zero-duration, as in NeMo's TDT offsets.
pub fn word_timestamps(tokens: &[TokenTimestamp]) -> Vec<WordTimestamp> {
    let mut words: Vec<WordTimestamp> = Vec::new();
    let mut boundary = false;
    for token in tokens {
        let piece = token.piece.replace('▁', " ");
        if piece.trim().is_empty() {
            // A standalone separator has no acoustic content. NeMo starts the
            // next word at its first nonempty subword, not the separator frame.
            boundary = true;
            continue;
        }
        let punctuation = piece
            .trim()
            .chars()
            .all(|c| matches!(c, ',' | '.' | '!' | '?'));
        if words.is_empty() || ((boundary || piece.starts_with(' ')) && !punctuation) {
            words.push(WordTimestamp {
                word: piece.trim().to_owned(),
                start: token.start,
                end: token.end,
            });
        } else if let Some(word) = words.last_mut() {
            word.word.push_str(piece.trim());
            word.end = token.end;
        }
        boundary = false;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_layout_masks_padding_and_isolates_convolution() {
        let layout = BatchLayout::new(&[16_000, 320, 1]).unwrap();
        assert_eq!(layout.offsets, [0, 32, 64]);
        assert_eq!(layout.valid_frames, [13, 1, 0]);
        assert_eq!(&layout.bounds[..2], &[0, 13]);
        assert!(layout.bounds[26..64].iter().all(|v| *v == 0));
        assert_eq!(&layout.bounds[64..66], &[32, 33]);
        assert!(layout.bounds[66..].iter().all(|v| *v == 0));
    }

    #[test]
    fn rejects_empty_oversized_and_overflowing_batches() {
        for counts in [
            vec![],
            vec![0],
            vec![1; 17],
            vec![usize::MAX],
            vec![64000; 16],
        ] {
            assert!(BatchLayout::new(&counts).is_err());
        }
        assert!(BatchLayout::new(&[16_000; 16]).is_ok());
    }

    #[test]
    fn standalone_separator_is_not_the_word_start_and_quotes_can_open_words() {
        let tokens = ["▁Hi", "▁", "\"", "there", "!", "\""]
            .into_iter()
            .enumerate()
            .map(|(i, piece)| TokenTimestamp {
                id: i as i32,
                piece: piece.into(),
                start_frame: i as i32,
                duration_frames: 1,
                start: i as f64 * 0.08,
                end: (i + 1) as f64 * 0.08,
            })
            .collect::<Vec<_>>();
        let words = word_timestamps(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[1].word, "\"there!\"");
        assert_eq!(words[1].start, 0.16);
        assert_eq!(words[1].end, 0.48);
    }

    #[test]
    fn words_follow_pieces_not_uniform_clip_spans() {
        let tokens = [
            ("▁Hel", 1.0, 1.08),
            ("lo", 1.08, 1.16),
            (",", 1.16, 1.16),
            ("▁world", 3.0, 3.24),
            ("!", 3.24, 3.24),
        ]
        .map(|(piece, start, end)| TokenTimestamp {
            id: 0,
            piece: piece.into(),
            start_frame: 0,
            duration_frames: 0,
            start,
            end,
        });
        let words = word_timestamps(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello,");
        assert_eq!((words[0].start, words[0].end), (1.0, 1.16));
        assert_eq!(words[1].word, "world!");
        assert_eq!((words[1].start, words[1].end), (3.0, 3.24));
        assert!(word_timestamps(&[]).is_empty());
    }
}
