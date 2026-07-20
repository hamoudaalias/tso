use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// TSO V7 Anchored Dual-LIF Generative Decoder
///
/// Adds **episodic memory** (anchor) to the V6 Dual-LIF architecture.
/// After ingesting the prompt, the slow state is frozen as an `anchor_state`.
/// At each generation step, if the slow state drifts too far from the anchor
/// (cosine < 1 - drift_threshold), the anchor is partially reinjected:
///     S_slow ← 0.8 · S_slow + 0.2 · S_anchor
///
/// This prevents the semantic collapse that pure LIF reservoirs suffer from
/// over long sequences — without BPTT, without gradients.
pub struct AnchoredTSODecoder {
    pub idx_to_word: HashMap<usize, String>,
    pub word_to_idx: HashMap<String, usize>,
    pub embeddings: Array2<f64>,

    /// Slow memory — semantic topic (α=0.9)
    pub slow_state: Array1<f64>,
    /// Fast memory — local syntax (α=0.5)
    pub fast_state: Array1<f64>,
    /// Episodic anchor — frozen slow state after prompt ingest
    pub anchor_state: Array1<f64>,

    pub alpha_slow: f64,
    pub alpha_fast: f64,
    /// S_pred = S_slow + η · S_fast
    pub syntax_weight: f64,
    /// If drift = 1 - cos(S_slow, S_anchor) > drift_threshold, reinject anchor
    pub drift_threshold: f64,
    /// Fraction of anchor reinjected during episodic recall (default 0.2)
    pub recall_strength: f64,

    pub stability_threshold: f64,
    pub negation_set: Vec<String>,
    pub friction_graph: Option<HashMap<String, Vec<String>>>,
    pub friction_lambda: f64,
}

impl AnchoredTSODecoder {
    pub fn new(
        idx_to_word: HashMap<usize, String>,
        word_to_idx: HashMap<String, usize>,
        embeddings: Array2<f64>,
        alpha_slow: f64,
        alpha_fast: f64,
    ) -> Self {
        let dim = embeddings.ncols();
        Self {
            idx_to_word,
            word_to_idx,
            embeddings,
            slow_state: Array1::zeros(dim),
            fast_state: Array1::zeros(dim),
            anchor_state: Array1::zeros(dim),
            alpha_slow,
            alpha_fast,
            syntax_weight: 0.4,
            drift_threshold: 0.3,
            recall_strength: 0.2,
            stability_threshold: 0.001,
            negation_set: Vec::new(),
            friction_graph: None,
            friction_lambda: 0.5,
        }
    }

    pub fn with_friction_graph(mut self, graph: HashMap<String, Vec<String>>) -> Self {
        self.friction_graph = Some(graph);
        self
    }

    pub fn reset(&mut self) {
        self.slow_state.fill(0.0);
        self.fast_state.fill(0.0);
        self.anchor_state.fill(0.0);
    }

    /// Build the combined predictive state: S_slow + η · S_fast
    fn predictive_state(&self) -> Array1<f64> {
        let p = self.slow_state.clone()
            + self.syntax_weight * &self.fast_state;
        let norm = p.dot(&p).sqrt();
        if norm > 1e-10 { p / norm } else { p }
    }

    /// Ingest prompt words and freeze the anchor.
    pub fn ingest(&mut self, word_vecs: &[(String, Array1<f64>)]) {
        for (word, v) in word_vecs {
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * v;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * v;
            if self.negation_set.contains(word) {
                self.slow_state.mapv_inplace(|x| -x);
                self.fast_state.mapv_inplace(|x| -x);
            }
        }
        self.normalize_states();
        // Episodic memory: freeze the slow state as the topic anchor
        self.anchor_state = self.slow_state.clone();
    }

    fn normalize_states(&mut self) {
        let norm_s = self.slow_state.dot(&self.slow_state).sqrt();
        if norm_s > 1e-10 { self.slow_state /= norm_s; }
        let norm_f = self.fast_state.dot(&self.fast_state).sqrt();
        if norm_f > 1e-10 { self.fast_state /= norm_f; }
    }

    /// Inverse Motor with friction biasing.
    pub fn predict_next(&self, last_word: Option<&str>, emitted: &HashSet<usize>) -> (usize, f64) {
        let s_pred = self.predictive_state();
        let friction_set: Option<&Vec<String>> =
            last_word.and_then(|w| self.friction_graph.as_ref()?.get(w));
        let lambda = self.friction_lambda;

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.embeddings.nrows() {
            if emitted.contains(&i) {
                continue;
            }
            let cos = s_pred.dot(&self.embeddings.row(i));
            let score = if let Some(ref neighbors) = friction_set {
                let word = self.idx_to_word.get(&i);
                let phi_bonus = match word {
                    Some(w) if neighbors.contains(w) => 1.0,
                    _ => 0.0,
                };
                lambda * cos + (1.0 - lambda) * phi_bonus
            } else {
                cos
            };
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        (best_idx, best_score)
    }

    pub fn top_k_candidates(
        &self,
        k: usize,
        last_word: Option<&str>,
        emitted: &HashSet<usize>,
    ) -> Vec<(usize, String, f64)> {
        let s_pred = self.predictive_state();
        let friction_set: Option<&Vec<String>> =
            last_word.and_then(|w| self.friction_graph.as_ref()?.get(w));
        let lambda = self.friction_lambda;

        let mut scores: Vec<(usize, f64)> = (0..self.embeddings.nrows())
            .filter(|i| !emitted.contains(i))
            .map(|i| {
                let cos = s_pred.dot(&self.embeddings.row(i));
                let score = if let Some(ref neighbors) = friction_set {
                    let word = self.idx_to_word.get(&i);
                    let phi_bonus = match word {
                        Some(w) if neighbors.contains(w) => 1.0,
                        _ => 0.0,
                    };
                    lambda * cos + (1.0 - lambda) * phi_bonus
                } else {
                    cos
                };
                (i, score)
            })
            .collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
            .into_iter()
            .map(|(idx, score)| {
                let word = self
                    .idx_to_word
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| format!("#{}", idx));
                (idx, word, score)
            })
            .collect()
    }

    pub fn generate(&mut self, max_tokens: usize, last_prompt_word: Option<&str>) -> Vec<String> {
        let mut generated: Vec<String> = Vec::new();
        let mut emitted: HashSet<usize> = HashSet::new();
        let mut last_context: Option<&str> = last_prompt_word;

        for step in 0..max_tokens {
            let (next_idx, score) = self.predict_next(last_context, &emitted);
            let next_word = match self.idx_to_word.get(&next_idx) {
                Some(w) => w.clone(),
                None => break,
            };

            // Anti-loop
            if generated.last().map_or(false, |w| *w == next_word) {
                eprintln!("  [step {}] loop on '{}', stopping", step, next_word);
                break;
            }

            generated.push(next_word.clone());
            last_context = Some(generated.last().unwrap().as_str());
            emitted.insert(next_idx);

            let word_vec = self.embeddings.row(next_idx).to_owned();
            let prev_slow = self.slow_state.clone();
            let prev_fast = self.fast_state.clone();

            // Dual-LIF update
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * &word_vec;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * &word_vec;

            if self.negation_set.contains(&next_word) {
                self.slow_state.mapv_inplace(|x| -x);
                self.fast_state.mapv_inplace(|x| -x);
            }

            self.normalize_states();

            // V7: Episodic drift control
            let alignment = self.slow_state.dot(&self.anchor_state);
            let drift = 1.0 - alignment;

            if drift > self.drift_threshold {
                let recalled = self.recall_strength;
                eprintln!(
                    "  [step {}] ⚓ drift={:.4} > {}, recalling {:.0}% anchor",
                    step, drift, self.drift_threshold, recalled * 100.0
                );
                self.slow_state = (1.0 - recalled) * &self.slow_state + recalled * &self.anchor_state;
                let n = self.slow_state.dot(&self.slow_state).sqrt();
                if n > 1e-10 { self.slow_state /= n; }
            }

            // Φ signal: measure change in predictive state
            let s_prev = {
                let p = prev_slow + self.syntax_weight * &prev_fast;
                let n = p.dot(&p).sqrt();
                if n > 1e-10 { p / n } else { p }
            };
            let s_curr = self.predictive_state();
            let delta = &s_curr - &s_prev;
            let energy: f64 = delta.mapv(|x| x * x).sum();

            let top3 = self.top_k_candidates(3, last_context, &emitted);
            let candidates_str = top3
                .iter()
                .map(|(_, w, s)| format!("{} ({:.4})", w, s))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "  [step {}] '{}' (score {:.4}, drift={:.4}, Φ={:.6})  top3: {}",
                step, next_word, score, drift, energy, candidates_str
            );

            if energy < self.stability_threshold {
                eprintln!(
                    "  [step {}] ⬇ homeostasis (Φ={:.6} < {})",
                    step, energy, self.stability_threshold
                );
                break;
            }
        }

        generated
    }
}
