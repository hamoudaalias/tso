use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct SNLISample {
    pub sentence1: String,
    pub sentence2: String,
    #[serde(alias = "gold_label")]
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct NLIDataset {
    pub samples: Vec<SNLISample>,
}

impl NLIDataset {
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(sample) = serde_json::from_str::<SNLISample>(line) {
                if sample.label != "-" {
                    samples.push(sample);
                }
            }
        }
        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn label_to_idx(label: &str) -> Option<usize> {
        match label {
            "entailment" => Some(0),
            "neutral" => Some(1),
            "contradiction" => Some(2),
            _ => None,
        }
    }

    pub fn idx_to_label(idx: usize) -> &'static str {
        match idx {
            0 => "entailment",
            1 => "neutral",
            2 => "contradiction",
            _ => "unknown",
        }
    }

    pub fn count_by_label(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for s in &self.samples {
            if let Some(idx) = Self::label_to_idx(&s.label) {
                counts[idx] += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_jsonl() {
        let jsonl = r#"{"sentence1":"A cat sat.","sentence2":"An animal sat.","gold_label":"entailment"}"#;
        let sample: SNLISample = serde_json::from_str(jsonl).unwrap();
        assert_eq!(sample.label, "entailment");
        assert_eq!(sample.sentence1, "A cat sat.");
    }

    #[test]
    fn test_parse_jsonl_skip_dash() {
        let jsonl = r#"{"sentence1":"a","sentence2":"b","gold_label":"-"}"#;
        let sample: SNLISample = serde_json::from_str(jsonl).unwrap();
        assert!(NLIDataset::label_to_idx(&sample.label).is_none());
    }

    #[test]
    fn test_label_conversion() {
        assert_eq!(NLIDataset::label_to_idx("entailment"), Some(0));
        assert_eq!(NLIDataset::label_to_idx("neutral"), Some(1));
        assert_eq!(NLIDataset::label_to_idx("contradiction"), Some(2));
        assert_eq!(NLIDataset::label_to_idx("-"), None);
        assert_eq!(NLIDataset::idx_to_label(0), "entailment");
        assert_eq!(NLIDataset::idx_to_label(1), "neutral");
        assert_eq!(NLIDataset::idx_to_label(2), "contradiction");
    }
}
