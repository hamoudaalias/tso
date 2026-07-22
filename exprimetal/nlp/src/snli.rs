use ndarray::Array1;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::embeddings::LinearOperator;

pub const LABEL_ENT: usize = 0;
pub const LABEL_NEU: usize = 1;
pub const LABEL_CON: usize = 2;

pub struct SNLIData {
    pub premise: Vec<String>,
    pub hypothesis: Vec<String>,
    pub label: Vec<usize>,
}

/// Parse a single JSONL line: extract "sentence1", "sentence2", "gold_label" values.
fn parse_line(line: &str) -> Option<(String, String, usize)> {
    // Cheap JSON parsing: find quoted values after known keys
    let sent1 = extract_field(line, "\"sentence1\":")?;
    let sent2 = extract_field(line, "\"sentence2\":")?;
    let label_str = extract_field(line, "\"gold_label\":")?;
    let label = match label_str.as_str() {
        "entailment" => LABEL_ENT,
        "neutral" => LABEL_NEU,
        "contradiction" => LABEL_CON,
        _ => return None,
    };
    Some((sent1, sent2, label))
}

fn extract_field(line: &str, key: &str) -> Option<String> {
    let start = line.find(key)? + key.len();
    // skip whitespace and opening quote
    let after_key = &line[start..];
    let quote_start = after_key.find('"')?;
    let after_quote = &after_key[quote_start + 1..];
    let quote_end = after_quote.find('"')?;
    Some(after_quote[..quote_end].to_string())
}

/// Load SNLI from a JSONL file path (streaming via BufReader).
pub fn load_snli(path: &str, max_lines: usize) -> SNLIData {
    let file = File::open(path).expect("SNLI file not found");
    let reader = BufReader::new(file);
    let mut data = SNLIData {
        premise: Vec::new(),
        hypothesis: Vec::new(),
        label: Vec::new(),
    };
    for line in reader.lines().take(max_lines) {
        let line = line.expect("read error");
        if line.trim().is_empty() {
            continue;
        }
        if let Some((p, h, l)) = parse_line(&line) {
            data.premise.push(p);
            data.hypothesis.push(h);
            data.label.push(l);
        }
    }
    data
}

/// Sparse PPMI matrix: co-occurrence counts from a corpus of sentences.
/// Supports matrix-free multiplication for randomized SVD.
pub struct SparsePPMI {
    cooc: HashMap<(usize, usize), usize>,
    freq: Vec<usize>,
    total_pairs: f64,
    pub vocab_size: usize,
}

impl SparsePPMI {
    pub fn new(sentences: &[Vec<usize>], window: usize, vocab_size: usize, freq: &[usize]) -> Self {
        let mut cooc: HashMap<(usize, usize), usize> = HashMap::new();
        for ids in sentences {
            for i in 0..ids.len() {
                let j_start = if i > window { i - window } else { 0 };
                let j_end = (i + window + 1).min(ids.len());
                for j in j_start..j_end {
                    if i == j {
                        continue;
                    }
                    let a = ids[i].min(ids[j]);
                    let b = ids[i].max(ids[j]);
                    *cooc.entry((a, b)).or_insert(0) += 1;
                }
            }
        }
        let total_pairs = cooc.values().sum::<usize>() as f64;
        SparsePPMI {
            cooc,
            freq: freq.to_vec(),
            total_pairs,
            vocab_size,
        }
    }

    #[allow(dead_code)]
    fn ppmi(&self, i: usize, j: usize) -> f64 {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let c_ij = *self.cooc.get(&(a, b)).unwrap_or(&0) as f64;
        if c_ij == 0.0 {
            return 0.0;
        }
        let f_i = self.freq[i] as f64;
        let f_j = self.freq[j] as f64;
        if f_i == 0.0 || f_j == 0.0 {
            return 0.0;
        }
        let pmi = (c_ij * self.total_pairs / (f_i * f_j)).ln();
        pmi.max(0.0)
    }

    /// y = A · x where A is the PPMI matrix.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.vocab_size];
        for (&(a, b), &cnt) in &self.cooc {
            let ppmi_val = {
                let c = cnt as f64;
                if c == 0.0 {
                    continue;
                }
                let f_a = self.freq[a] as f64;
                let f_b = self.freq[b] as f64;
                if f_a == 0.0 || f_b == 0.0 {
                    continue;
                }
                (c * self.total_pairs / (f_a * f_b)).ln().max(0.0)
            };
            if ppmi_val > 0.0 {
                y[a] += ppmi_val * x[b];
                y[b] += ppmi_val * x[a];
            }
        }
        y
    }
}

impl SparsePPMI {
    pub fn cooc_count(&self) -> usize {
        self.cooc.len()
    }

    pub fn cooc_iter(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.cooc.keys()
    }

    pub fn ppmi_from_cooc(&self, a: usize, b: usize) -> f64 {
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        let c_ij = *self.cooc.get(&(i, j)).unwrap_or(&0) as f64;
        if c_ij == 0.0 {
            return 0.0;
        }
        let f_a = self.freq[i] as f64;
        let f_b = self.freq[j] as f64;
        if f_a == 0.0 || f_b == 0.0 {
            return 0.0;
        }
        (c_ij * self.total_pairs / (f_a * f_b)).ln().max(0.0)
    }
}

impl LinearOperator for SparsePPMI {
    fn nrows(&self) -> usize { self.vocab_size }
    fn ncols(&self) -> usize { self.vocab_size }
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        let v = self.matvec(x.as_slice().unwrap());
        Array1::from_vec(v)
    }
}
