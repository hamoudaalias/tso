use ndarray::Array2;
use std::collections::{HashMap, HashSet};

pub struct Vocabulary {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Vocabulary {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        }
    }

    pub fn insert(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            let id = self.id_to_word.len();
            self.id_to_word.push(word.to_string());
            self.word_to_id.insert(word.to_string(), id);
            id
        }
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }
}

pub struct Corpus {
    pub vocab: Vocabulary,
    pub sentences: Vec<Vec<usize>>,
    cooc: HashMap<(usize, usize), usize>,
    pub freq: Vec<usize>,
    window: usize,
}

impl Corpus {
    pub fn new(window: usize) -> Self {
        Corpus {
            vocab: Vocabulary::new(),
            sentences: Vec::new(),
            cooc: HashMap::new(),
            freq: Vec::new(),
            window,
        }
    }

    pub fn add_sentence(&mut self, sentence: &str) {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        let ids: Vec<usize> = tokens.iter().map(|t| self.vocab.insert(t)).collect();

        if self.freq.len() < self.vocab.size() {
            self.freq.resize(self.vocab.size(), 0);
        }
        for &id in &ids {
            self.freq[id] += 1;
        }

        self.sentences.push(ids.clone());

        for i in 0..ids.len() {
            let j_start = if i > self.window { i - self.window } else { 0 };
            let j_end = (i + self.window + 1).min(ids.len());
            for j in j_start..j_end {
                if i == j { continue; }
                let a = ids[i];
                let b = ids[j];
                if a < b {
                    *self.cooc.entry((a, b)).or_insert(0) += 1;
                } else {
                    *self.cooc.entry((b, a)).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn total_pairs(&self) -> f64 {
        self.cooc.values().sum::<usize>() as f64
    }

    pub fn ppmi(&self, i: usize, j: usize) -> f64 {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let c_ij = *self.cooc.get(&(a, b)).unwrap_or(&0) as f64;
        if c_ij == 0.0 {
            return 0.0;
        }
        let n = self.total_pairs();
        let f_i = self.freq[i] as f64;
        let f_j = self.freq[j] as f64;
        if f_i == 0.0 || f_j == 0.0 {
            return 0.0;
        }
        let pmi = (c_ij * n / (f_i * f_j)).ln();
        pmi.max(0.0)
    }

    pub fn neighbors(&self, word_id: usize) -> HashSet<usize> {
        let mut nbrs = HashSet::new();
        for (&(a, b), _) in self.cooc.iter() {
            if a == word_id { nbrs.insert(b); }
            if b == word_id { nbrs.insert(a); }
        }
        nbrs
    }

    pub fn jaccard_context(&self, i: usize, j: usize) -> f64 {
        let ni = self.neighbors(i);
        let nj = self.neighbors(j);
        let union: HashSet<_> = ni.union(&nj).cloned().collect();
        if union.is_empty() {
            return 0.0;
        }
        let intersection = ni.intersection(&nj).count();
        intersection as f64 / union.len() as f64
    }

    pub fn ppmi_matrix(&self) -> Array2<f64> {
        let n = self.vocab.size();
        let mut m = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = self.ppmi(i, j);
            }
        }
        m
    }

    pub fn infer_edge_weight(&self, i: usize, j: usize, ppmi_threshold: f64, jaccard_threshold: f64) -> Option<i8> {
        let p = self.ppmi(i, j);
        if p > ppmi_threshold {
            return Some(1);
        }
        if p == 0.0 {
            let jac = self.jaccard_context(i, j);
            if jac > jaccard_threshold {
                return Some(-1);
            }
        }
        None
    }
}
