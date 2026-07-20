use ndarray::{s, Array1, Array2};
use std::collections::{HashMap, HashSet};

/// V9.1: Volatile Syntax Inverter — morphologic scarring
///
/// When a negation marker (e.g. "not", "no") is read, the *next* word's
/// embedding is inverted *before* LIF incorporation. This creates an
/// immediate geometric contradiction in the latent space — a volatile
/// "scar" — without waiting for slow R-STDP to learn the exclusion.
///
/// The scar is ephemeral (single token). Only repeated co-occurrence
/// of the same pattern across the corpus eventually burns the edge
/// to -1 via R-STDP — the spark is syntactic, the fire is statistical.
pub struct VolatileSyntaxInverter {
    pub markers: Vec<String>,
    pub inversion_active: bool,
}

impl VolatileSyntaxInverter {
    pub fn new(markers: &[&str]) -> Self {
        Self {
            markers: markers.iter().map(|s| s.to_string()).collect(),
            inversion_active: false,
        }
    }

    /// Process a word: if the previous word was a negation marker, invert
    /// the embedding. Returns the (possibly inverted) vector.
    pub fn process(&mut self, word: &str, word_vec: &Array1<f64>) -> Array1<f64> {
        if self.inversion_active {
            self.inversion_active = false;
            return -word_vec;
        }
        if self.markers.iter().any(|m| m == word) {
            self.inversion_active = true;
        }
        word_vec.clone()
    }

    pub fn reset(&mut self) {
        self.inversion_active = false;
    }
}

impl Default for VolatileSyntaxInverter {
    fn default() -> Self {
        Self::new(&["not", "no", "never", "without"])
    }
}

/// TSO V9 Triple-LIF Dynamic Anchor Generative Decoder
///
/// Extends V7 with:
/// - **Triple-LIF**: slow (α=0.9, topic), medium (α=0.7, paragraph), fast (α=0.5, syntax).
/// - **Dynamic Anchor**: every `anchor_interval` tokens, if the medium-state
///   friction is below threshold, the anchor teleports to the current medium
///   state. The system progresses thematically instead of merely oscillating.
///
/// Predictive state:
///     S_pred = S_slow + η_m · S_medium + η_f · S_fast
pub struct AnchoredTSODecoder {
    pub idx_to_word: HashMap<usize, String>,
    pub word_to_idx: HashMap<String, usize>,
    /// Variable-dimension embeddings: each word can have its own latent size.
    /// Intersection-based dot products handle mismatched dimensions.
    pub embeddings: Vec<Array1<f64>>,

    /// Slow memory — semantic topic (α=0.9)
    pub slow_state: Array1<f64>,
    /// Medium memory — paragraph context (α=0.7)
    pub medium_state: Array1<f64>,
    /// Fast memory — local syntax (α=0.5)
    pub fast_state: Array1<f64>,
    /// Episodic anchor — teleports to medium_state when paragraph shifts
    pub anchor_state: Array1<f64>,

    pub alpha_slow: f64,
    pub alpha_medium: f64,
    pub alpha_fast: f64,
    /// S_pred = S_slow + η_m · S_medium + η_f · S_fast
    pub syntax_weight: f64,
    pub medium_weight: f64,
    /// If drift = 1 - cos(S_slow, S_anchor) > drift_threshold, reinject anchor
    pub drift_threshold: f64,
    /// Fraction of anchor reinjected during episodic recall (default 0.2)
    pub recall_strength: f64,

    /// How often (in tokens) to evaluate anchor teleportation
    pub anchor_interval: usize,
    /// Medium-state Δ below this threshold triggers anchor teleport
    pub anchor_friction_threshold: f64,
    /// Tokens emitted since last anchor teleport
    pub token_count_since_anchor: usize,
    /// Snapshot of medium_state at last anchor teleport (for friction calc)
    pub last_medium_snapshot: Array1<f64>,

    pub stability_threshold: f64,
    pub volatile_inverter: VolatileSyntaxInverter,
    pub friction_graph: Option<HashMap<String, Vec<String>>>,
    pub friction_lambda: f64,
}

impl AnchoredTSODecoder {
    /// Create a decoder from uniform-dimension embeddings (Array2).
    /// The matrix is immediately converted to variable-dimension Vec<Array1>.
    pub fn new(
        idx_to_word: HashMap<usize, String>,
        word_to_idx: HashMap<String, usize>,
        embeddings: Array2<f64>,
        alpha_slow: f64,
        alpha_fast: f64,
    ) -> Self {
        let dim = embeddings.ncols();
        let embeddings: Vec<Array1<f64>> =
            (0..embeddings.nrows()).map(|i| embeddings.row(i).to_owned()).collect();
        Self {
            idx_to_word,
            word_to_idx,
            embeddings,
            slow_state: Array1::zeros(dim),
            medium_state: Array1::zeros(dim),
            fast_state: Array1::zeros(dim),
            anchor_state: Array1::zeros(dim),
            last_medium_snapshot: Array1::zeros(dim),
            alpha_slow,
            alpha_medium: 0.7,
            alpha_fast,
            syntax_weight: 0.4,
            medium_weight: 0.3,
            drift_threshold: 0.3,
            recall_strength: 0.2,
            anchor_interval: 20,
            anchor_friction_threshold: 0.05,
            token_count_since_anchor: 0,
            stability_threshold: 0.001,
            volatile_inverter: VolatileSyntaxInverter::default(),
            friction_graph: None,
            friction_lambda: 0.5,
        }
    }

    /// Create a decoder from pre-built variable-dimension embeddings.
    /// The embedding dimension is inferred from the first vector.
    pub fn from_embeddings_vec(
        idx_to_word: HashMap<usize, String>,
        word_to_idx: HashMap<String, usize>,
        embeddings: Vec<Array1<f64>>,
        alpha_slow: f64,
        alpha_fast: f64,
    ) -> Self {
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
        Self {
            idx_to_word,
            word_to_idx,
            embeddings,
            slow_state: Array1::zeros(dim),
            medium_state: Array1::zeros(dim),
            fast_state: Array1::zeros(dim),
            anchor_state: Array1::zeros(dim),
            last_medium_snapshot: Array1::zeros(dim),
            alpha_slow,
            alpha_medium: 0.7,
            alpha_fast,
            syntax_weight: 0.4,
            medium_weight: 0.3,
            drift_threshold: 0.3,
            recall_strength: 0.2,
            anchor_interval: 20,
            anchor_friction_threshold: 0.05,
            token_count_since_anchor: 0,
            stability_threshold: 0.001,
            volatile_inverter: VolatileSyntaxInverter::default(),
            friction_graph: None,
            friction_lambda: 0.5,
        }
    }

    pub fn with_negation_markers(mut self, markers: &[&str]) -> Self {
        self.volatile_inverter = VolatileSyntaxInverter::new(markers);
        self
    }

    pub fn with_friction_graph(mut self, graph: HashMap<String, Vec<String>>) -> Self {
        self.friction_graph = Some(graph);
        self
    }

    pub fn reset(&mut self) {
        self.slow_state.fill(0.0);
        self.medium_state.fill(0.0);
        self.fast_state.fill(0.0);
        self.anchor_state.fill(0.0);
        self.last_medium_snapshot.fill(0.0);
        self.token_count_since_anchor = 0;
        self.volatile_inverter.reset();
    }

    /// V10: Expand all LIF states to at least `target_dim` dimensions.
    /// New dimensions are zero-padded. States only grow, never shrink.
    pub fn ensure_dim(&mut self, target_dim: usize) {
        if self.slow_state.len() >= target_dim { return; }
        let grow = |v: &Array1<f64>| -> Array1<f64> {
            let mut new = Array1::zeros(target_dim);
            new.slice_mut(s![..v.len()]).assign(v);
            new
        };
        self.slow_state = grow(&self.slow_state);
        self.medium_state = grow(&self.medium_state);
        self.fast_state = grow(&self.fast_state);
        self.anchor_state = grow(&self.anchor_state);
        self.last_medium_snapshot = grow(&self.last_medium_snapshot);
    }

    /// Build the combined predictive state: S_slow + η_m · S_medium + η_f · S_fast
    pub fn predictive_state(&self) -> Array1<f64> {
        let p = self.slow_state.clone()
            + self.medium_weight * &self.medium_state
            + self.syntax_weight * &self.fast_state;
        let norm = p.dot(&p).sqrt();
        if norm > 1e-10 { p / norm } else { p }
    }

    /// Ingest prompt words and freeze the anchor.
    pub fn ingest(&mut self, word_vecs: &[(String, Array1<f64>)]) {
        for (word, v) in word_vecs {
            // V10: Ensure LIF states are large enough for this word
            self.ensure_dim(v.len());
            // V9.1: Volatile syntax inversion — invert the *next* word's embedding
            let processed = self.volatile_inverter.process(word, v);
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * &processed;
            self.medium_state = self.alpha_medium * &self.medium_state + (1.0 - self.alpha_medium) * &processed;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * &processed;
        }
        self.normalize_states();
        // Episodic memory: freeze the slow state as the topic anchor
        self.anchor_state = self.slow_state.clone();
        self.last_medium_snapshot = self.medium_state.clone();
        self.token_count_since_anchor = 0;
    }

    pub fn normalize_states(&mut self) {
        let norm_s = self.slow_state.dot(&self.slow_state).sqrt();
        if norm_s > 1e-10 { self.slow_state /= norm_s; }
        let norm_m = self.medium_state.dot(&self.medium_state).sqrt();
        if norm_m > 1e-10 { self.medium_state /= norm_m; }
        let norm_f = self.fast_state.dot(&self.fast_state).sqrt();
        if norm_f > 1e-10 { self.fast_state /= norm_f; }
    }

    /// V10: Dot product on the intersection of two vectors (min dimension).
    pub fn intersection_dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let n = a.len().min(b.len());
        a.slice(s![..n]).dot(&b.slice(s![..n]))
    }

    /// Inverse Motor with friction biasing.
    pub fn predict_next(&self, last_word: Option<&str>, emitted: &HashSet<usize>) -> (usize, f64) {
        let s_pred = self.predictive_state();
        let friction_set: Option<&Vec<String>> =
            last_word.and_then(|w| self.friction_graph.as_ref()?.get(w));
        let lambda = self.friction_lambda;

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.embeddings.len() {
            if emitted.contains(&i) {
                continue;
            }
            let cos = Self::intersection_dot(&s_pred, &self.embeddings[i]);
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

        let mut scores: Vec<(usize, f64)> = (0..self.embeddings.len())
            .filter(|i| !emitted.contains(i))
            .map(|i| {
                let cos = Self::intersection_dot(&s_pred, &self.embeddings[i]);
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

            let raw_word_vec = self.embeddings[next_idx].clone();
            // V9.1: Volatile syntax inversion — inverts the word AFTER a negation
            let word_vec = self.volatile_inverter.process(&next_word, &raw_word_vec);
            let prev_slow = self.slow_state.clone();
            let prev_medium = self.medium_state.clone();
            let prev_fast = self.fast_state.clone();

            // V10: If the word has a larger dimension, grow LIF states first
            self.ensure_dim(word_vec.len());
            // V9: Triple-LIF update
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * &word_vec;
            self.medium_state = self.alpha_medium * &self.medium_state + (1.0 - self.alpha_medium) * &word_vec;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * &word_vec;

            self.normalize_states();

            // V7: Episodic drift control (slow vs anchor)
            let alignment = self.slow_state.dot(&self.anchor_state);
            let drift = 1.0 - alignment;

            if drift > self.drift_threshold {
                let recalled = self.recall_strength;
                eprintln!(
                    "  [step {}] drift={:.4} > {}, recalling {:.0}% anchor",
                    step, drift, self.drift_threshold, recalled * 100.0
                );
                self.slow_state = (1.0 - recalled) * &self.slow_state + recalled * &self.anchor_state;
                let n = self.slow_state.dot(&self.slow_state).sqrt();
                if n > 1e-10 { self.slow_state /= n; }
            }

            // V9: Dynamic anchor teleport
            self.token_count_since_anchor += 1;
            let mut anchor_teleported = false;
            if self.token_count_since_anchor >= self.anchor_interval {
                // Medium friction = Δ between current medium state and snapshot
                let m_delta = &self.medium_state - &self.last_medium_snapshot;
                let medium_friction: f64 = m_delta.mapv(|x| x * x).sum();
                if medium_friction < self.anchor_friction_threshold {
                    self.anchor_state = self.medium_state.clone();
                    self.last_medium_snapshot = self.medium_state.clone();
                    self.token_count_since_anchor = 0;
                    anchor_teleported = true;
                    eprintln!(
                        "  [step {}] anchor teleport (medium friction={:.6} < {})",
                        step, medium_friction, self.anchor_friction_threshold
                    );
                }
            }

            // Φ signal: measure change in predictive state
            let s_prev = {
                let p = prev_slow + self.medium_weight * &prev_medium + self.syntax_weight * &prev_fast;
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
            let flag = if anchor_teleported { " [ANCHOR TELEPORT]" } else { "" };
            eprintln!(
                "  [step {}] '{}' (score {:.4}, drift={:.4}, Φ={:.6}){}  top3: {}",
                step, next_word, score, drift, energy, flag, candidates_str
            );

            if energy < self.stability_threshold {
                eprintln!(
                    "  [step {}] homeostasis (Φ={:.6} < {})",
                    step, energy, self.stability_threshold
                );
                break;
            }
        }

        generated
    }
}
