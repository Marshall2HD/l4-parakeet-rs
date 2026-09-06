use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct BenchmarkManifest {
    pub reference: ReferenceContract,
    pub scoring: ScoringContract,
    pub performance: PerformanceContract,
    pub optimization_baseline: OptimizationBaseline,
}

#[derive(Debug, Deserialize)]
pub struct ReferenceContract {
    pub alignment_anchor: String,
}

#[derive(Debug, Deserialize)]
pub struct ScoringContract {
    pub minimum_reference_coverage: f64,
    pub maximum_absolute_wer_regression: f64,
}

#[derive(Debug, Deserialize)]
pub struct PerformanceContract {
    pub minimum_relative_improvement: f64,
}

#[derive(Debug, Deserialize)]
pub struct OptimizationBaseline {
    pub wer: f64,
    pub scored_words: usize,
    pub median_latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub candidate: PathBuf,
    pub normalization: &'static str,
    pub alignment_anchor: String,
    pub scored_words: usize,
    pub word_edits: usize,
    pub wer: f64,
    pub reference_coverage: f64,
    pub median_latency_ms: f64,
    pub relative_speed_improvement: f64,
    pub gates: EvaluationGates,
    pub accepted: bool,
}

#[derive(Debug, Serialize)]
pub struct EvaluationGates {
    pub minimum_reference_coverage: f64,
    pub maximum_wer: f64,
    pub maximum_latency_ms: f64,
    pub coverage_pass: bool,
    pub wer_pass: bool,
    pub speed_pass: bool,
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("invalid benchmark input: {0}")]
    Invalid(String),
}

pub fn evaluate_night_circus(
    manifest_path: &Path,
    reference_path: &Path,
    candidate_path: &Path,
) -> Result<EvaluationReport, EvaluationError> {
    let manifest: BenchmarkManifest = read_json(manifest_path)?;
    validate_contract(&manifest)?;
    let candidate: serde_json::Value = read_json(candidate_path)?;
    let transcript = candidate
        .get("transcript")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| EvaluationError::Invalid("candidate has no string transcript".into()))?;
    let median_latency_ms = candidate
        .get("median_latency_ms")
        .and_then(serde_json::Value::as_f64)
        .filter(|latency| latency.is_finite() && *latency > 0.0)
        .ok_or_else(|| {
            EvaluationError::Invalid("candidate has no positive median_latency_ms".into())
        })?;
    let reference = fs::read_to_string(reference_path).map_err(|source| EvaluationError::Read {
        path: reference_path.to_owned(),
        source,
    })?;

    let reference_words = normalize_words(&reference);
    let hypothesis_words = normalize_words(transcript);
    let anchor_words = normalize_words(&manifest.reference.alignment_anchor);
    if anchor_words.is_empty() {
        return Err(EvaluationError::Invalid(
            "alignment anchor is empty after normalization".into(),
        ));
    }
    let reference_start = find_subsequence(&reference_words, &anchor_words)
        .ok_or_else(|| EvaluationError::Invalid("reference does not contain the anchor".into()))?;
    let hypothesis_start = find_subsequence(&hypothesis_words, &anchor_words)
        .ok_or_else(|| EvaluationError::Invalid("candidate does not contain the anchor".into()))?;
    let hypothesis = &hypothesis_words[hypothesis_start..];
    if hypothesis.is_empty() || reference_start + hypothesis.len() > reference_words.len() {
        return Err(EvaluationError::Invalid(
            "candidate has no scoreable anchored span".into(),
        ));
    }
    let reference = &reference_words[reference_start..reference_start + hypothesis.len()];
    let word_edits = levenshtein(reference, hypothesis);
    let scored_words = hypothesis.len();
    let wer = word_edits as f64 / scored_words as f64;
    let reference_coverage =
        (scored_words as f64 / manifest.optimization_baseline.scored_words as f64).min(1.0);
    let relative_speed_improvement =
        1.0 - median_latency_ms / manifest.optimization_baseline.median_latency_ms;
    let maximum_wer =
        manifest.optimization_baseline.wer + manifest.scoring.maximum_absolute_wer_regression;
    let maximum_latency_ms = manifest.optimization_baseline.median_latency_ms
        * (1.0 - manifest.performance.minimum_relative_improvement);
    let gates = EvaluationGates {
        minimum_reference_coverage: manifest.scoring.minimum_reference_coverage,
        maximum_wer,
        maximum_latency_ms,
        coverage_pass: reference_coverage >= manifest.scoring.minimum_reference_coverage,
        wer_pass: wer <= maximum_wer,
        speed_pass: median_latency_ms <= maximum_latency_ms,
    };

    Ok(EvaluationReport {
        schema_version: 1,
        candidate: candidate_path.to_owned(),
        normalization: "lowercase ASCII; preserve apostrophes; replace other punctuation with spaces",
        alignment_anchor: manifest.reference.alignment_anchor,
        scored_words,
        word_edits,
        wer,
        reference_coverage,
        median_latency_ms,
        relative_speed_improvement,
        accepted: gates.coverage_pass && gates.wer_pass && gates.speed_pass,
        gates,
    })
}

fn validate_contract(manifest: &BenchmarkManifest) -> Result<(), EvaluationError> {
    let baseline = &manifest.optimization_baseline;
    if baseline.scored_words == 0
        || !baseline.wer.is_finite()
        || baseline.wer < 0.0
        || !baseline.median_latency_ms.is_finite()
        || baseline.median_latency_ms <= 0.0
        || !(0.0..=1.0).contains(&manifest.scoring.minimum_reference_coverage)
        || manifest.scoring.maximum_absolute_wer_regression < 0.0
        || !(0.0..1.0).contains(&manifest.performance.minimum_relative_improvement)
    {
        return Err(EvaluationError::Invalid(
            "manifest has invalid optimization gates or baseline".into(),
        ));
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, EvaluationError> {
    let contents = fs::read(path).map_err(|source| EvaluationError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&contents).map_err(|source| EvaluationError::Json {
        path: path.to_owned(),
        source,
    })
}

fn normalize_words(text: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len());
    for character in text.chars() {
        let character = match character {
            '’' | '‘' => '\'',
            other => other,
        };
        let character = character.to_ascii_lowercase();
        if character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || character == '\''
            || character.is_whitespace()
        {
            normalized.push(character);
        } else {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().map(str::to_owned).collect()
}

fn find_subsequence(haystack: &[String], needle: &[String]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn levenshtein(reference: &[String], hypothesis: &[String]) -> usize {
    let mut row = (0..=hypothesis.len()).collect::<Vec<_>>();
    for (reference_index, reference_word) in reference.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = reference_index + 1;
        for (hypothesis_index, hypothesis_word) in hypothesis.iter().enumerate() {
            let column = hypothesis_index + 1;
            let previous_row = row[column];
            row[column] = (previous_row + 1)
                .min(row[column - 1] + 1)
                .min(diagonal + usize::from(reference_word != hypothesis_word));
            diagonal = previous_row;
        }
    }
    row[hypothesis.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_matches_the_established_evaluator() {
        assert_eq!(
            normalize_words("It’s déjà-vu, #42!"),
            ["it's", "d", "j", "vu", "42"]
        );
    }

    #[test]
    fn computes_exact_word_edit_distance() {
        let reference = normalize_words("the circus arrives without warning");
        let hypothesis = normalize_words("the circus came without a warning");
        assert_eq!(levenshtein(&reference, &hypothesis), 2);
    }

    #[test]
    fn finds_only_whole_word_anchors() {
        let text = normalize_words("before the circus arrives after");
        let anchor = normalize_words("the circus arrives");
        assert_eq!(find_subsequence(&text, &anchor), Some(1));
    }
}
