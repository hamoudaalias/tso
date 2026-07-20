use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// TSO V6 Dual-LIF Generative Decoder
///
/// Two interacting LIF memories:
///   - **Slow** (α=0.9): topic/subject memory, stable across the sentence
///   - **Fast**  (α=0.5): local syntax memory, tracks the last few words
///
/// The predictive state for Inverse Motor is a weighted blend:
///     S_pred = S_slow + η · S_fast
///
/// This gives the decoder both a sense of what the sentence is *about*
/// (slow) and what is *likely to come next* (fast, biased by the friction
/// topology of the last word).
pub struct DualTSODecoder {
    pub idx_to_word: HashMap<usize, String>,
    pub word_to_idx: HashMap<String, usize>,
    pub embeddings: Array2<f64>,
    /// Slow memory — semantic topic (α=0.9)
    pub slow_state: Array1<f64>,
    /// Fast memory — local syntax (α=0.5)
    pub fast_state: Array1<f64>,
    pub alpha_slow: f64,
    pub alpha_fast: f64,
    /// How much the fast state influences the prediction.
    /// S_pred = S_slow + syntax_weight · S_fast
    /// Typical range: 0.3–0.6
    pub syntax_weight: f64,
    pub stability_threshold: f64,
    pub negation_set: Vec<String>,
    pub friction_graph: Option<HashMap<String, Vec<String>>>,
    pub friction_lambda: f64,
    pub repeat_penalty: f64,
    emission_counts: HashMap<String, usize>,
}

impl DualTSODecoder {
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
            alpha_slow,
            alpha_fast,
            syntax_weight: 0.4,
            stability_threshold: 0.001,
            negation_set: Vec::new(),
            friction_graph: None,
            friction_lambda: 0.5,
            repeat_penalty: 0.5,
            emission_counts: HashMap::new(),
        }
    }

    pub fn with_friction_graph(mut self, graph: HashMap<String, Vec<String>>) -> Self {
        self.friction_graph = Some(graph);
        self
    }

    pub fn reset(&mut self) {
        self.slow_state.fill(0.0);
        self.fast_state.fill(0.0);
        self.emission_counts.clear();
    }

    /// Build the combined predictive state: S_slow + η · S_fast
    fn predictive_state(&self) -> Array1<f64> {
        let p = self.slow_state.clone()
            + self.syntax_weight * &self.fast_state;
        let norm = p.dot(&p).sqrt();
        if norm > 1e-10 {
            p / norm
        } else {
            p
        }
    }

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
    }

    fn normalize_states(&mut self) {
        // ─ Slow state ─
        let norm_s = self.slow_state.dot(&self.slow_state).sqrt();
        if norm_s > 1e-10 {
            self.slow_state /= norm_s;
        }
        // ─ Fast state ─
        let norm_f = self.fast_state.dot(&self.fast_state).sqrt();
        if norm_f > 1e-10 {
            self.fast_state /= norm_f;
        }
    }

    /// Inverse Motor with dual-LIF predictive state + friction.
    pub fn predict_next(&self, last_word: Option<&str>) -> (usize, f64) {
        let s_pred = self.predictive_state();

        let friction_set: Option<&Vec<String>> =
            last_word.and_then(|w| self.friction_graph.as_ref()?.get(w));
        let lambda = self.friction_lambda;

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.embeddings.nrows() {
            let cos = s_pred.dot(&self.embeddings.row(i));

            let mut score = if let Some(ref neighbors) = friction_set {
                let word = self.idx_to_word.get(&i);
                let phi_bonus = match word {
                    Some(w) if neighbors.contains(w) => 1.0,
                    _ => 0.0,
                };
                lambda * cos + (1.0 - lambda) * phi_bonus
            } else {
                cos
            };

            if self.repeat_penalty > 0.0 {
                if let Some(word) = self.idx_to_word.get(&i) {
                    if let Some(&count) = self.emission_counts.get(word) {
                        score *= (1.0 - self.repeat_penalty).powi(count as i32);
                    }
                }
            }

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
    ) -> Vec<(usize, String, f64)> {
        let s_pred = self.predictive_state();
        let friction_set: Option<&Vec<String>> =
            last_word.and_then(|w| self.friction_graph.as_ref()?.get(w));
        let lambda = self.friction_lambda;

        let mut scores: Vec<(usize, f64)> = (0..self.embeddings.nrows())
            .map(|i| {
                let cos = s_pred.dot(&self.embeddings.row(i));
                let mut score = if let Some(ref neighbors) = friction_set {
                    let word = self.idx_to_word.get(&i);
                    let phi_bonus = match word {
                        Some(w) if neighbors.contains(w) => 1.0,
                        _ => 0.0,
                    };
                    lambda * cos + (1.0 - lambda) * phi_bonus
                } else {
                    cos
                };
                if self.repeat_penalty > 0.0 {
                    if let Some(word) = self.idx_to_word.get(&i) {
                        if let Some(&count) = self.emission_counts.get(word) {
                            score *= (1.0 - self.repeat_penalty).powi(count as i32);
                        }
                    }
                }
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
        let mut last_context: Option<&str> = last_prompt_word;

        for step in 0..max_tokens {
            let (next_idx, score) = self.predict_next(last_context);
            let next_word = match self.idx_to_word.get(&next_idx) {
                Some(w) => w.clone(),
                None => break,
            };

            if generated.last().map_or(false, |w| *w == next_word) {
                eprintln!("  [step {}] loop on '{}', stopping", step, next_word);
                break;
            }

            generated.push(next_word.clone());
            last_context = Some(generated.last().unwrap().as_str());

            *self.emission_counts.entry(next_word.clone()).or_insert(0) += 1;

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

            // Φ signal: measure change in the predictive state
            let s_prev = {
                let p = prev_slow + self.syntax_weight * &prev_fast;
                let n = p.dot(&p).sqrt();
                if n > 1e-10 { p / n } else { p }
            };
            let s_curr = self.predictive_state();
            let delta = &s_curr - &s_prev;
            let energy: f64 = delta.mapv(|x| x * x).sum();

            let top3 = self.top_k_candidates(3, last_context);
            let candidates_str = top3
                .iter()
                .map(|(_, w, s)| format!("{} ({:.4})", w, s))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "  [step {}] '{}' (score {:.4}, Φ={:.6})  top3: {}",
                step, next_word, score, energy, candidates_str
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
