use crate::friction::FatigueTracker;
use ndarray::{s, Array1, Array2};
use std::collections::{HashMap, HashSet};

/// V11: Endogenous Inversion Detector — emergent negation learning
///
/// Replaces the hardcoded `VolatileSyntaxInverter` (V9.1) with a mechanism
/// that *discovers* negation markers from the system's own dynamics.
///
/// **Bootstrapping:** A seed set of markers provides correct behavior from
/// the start. The system additionally learns new triggers by observing
/// trajectory flips: if a word consistently precedes a >90° change in the
/// predictive state, its score increases. Once a score crosses `threshold`,
/// the word becomes an automatic trigger.
///
/// **Spark (syntax) + Fire (statistics):** The seed handles the common cases
/// immediately; the learned scores capture rare or domain-specific patterns.
pub struct EndogenousInversionDetector {
    /// Hardcoded seed markers for bootstrapping (e.g. "not", "no")
    pub seed_markers: Vec<String>,
    /// Learned scores for every word encountered
    pub inversion_scores: HashMap<String, f64>,
    /// Score threshold for automatic trigger
    pub threshold: f64,
    /// Learning rate for score increments on flip detection
    pub learning_rate: f64,
    /// Current flag — if true, the next word's embedding will be inverted
    pub inversion_active: bool,
    /// The word that set `inversion_active` (for credit assignment in learn())
    pub last_trigger: Option<String>,
}

impl EndogenousInversionDetector {
    pub fn new(seed_markers: &[&str]) -> Self {
        Self {
            seed_markers: seed_markers.iter().map(|s| s.to_string()).collect(),
            inversion_scores: HashMap::new(),
            threshold: 0.5,
            learning_rate: 0.1,
            inversion_active: false,
            last_trigger: None,
        }
    }

    /// Process a word: if `inversion_active` is set, invert the embedding.
    /// Then check if this word (seed or learned) should trigger for the
    /// *next* word.
    pub fn process(&mut self, word: &str, word_vec: &Array1<f64>) -> Array1<f64> {
        if self.inversion_active {
            self.inversion_active = false;
            return -word_vec;
        }
        if self.is_trigger(word) {
            self.inversion_active = true;
            self.last_trigger = Some(word.to_string());
        }
        word_vec.clone()
    }

    /// Check if a word is a known trigger (seed marker or learned score).
    pub fn is_trigger(&self, word: &str) -> bool {
        if self.seed_markers.iter().any(|m| m == word) {
            return true;
        }
        self.inversion_scores.get(word).copied().unwrap_or(0.0) >= self.threshold
    }

    /// Observe the change in predictive state before/after a word.
    /// If the trajectory flipped by >90°, credit `last_trigger`.
    /// Call this AFTER the LIF update for the word.
    pub fn observe_flip(&mut self, before: &Array1<f64>, after: &Array1<f64>) {
        let n_b = before.dot(before).sqrt();
        let n_a = after.dot(after).sqrt();
        if n_b < 1e-10 || n_a < 1e-10 {
            self.last_trigger = None;
            return;
        }
        let cos = before.dot(after) / (n_b * n_a);
        if cos < 0.0 {
            if let Some(trigger) = self.last_trigger.take() {
                let entry = self.inversion_scores.entry(trigger).or_insert(0.0);
                *entry += self.learning_rate;
            }
        } else {
            self.last_trigger = None;
        }
    }

    pub fn reset(&mut self) {
        self.inversion_active = false;
        self.last_trigger = None;
        // Learned scores persist across resets — instincts are permanent
    }
}

impl Default for EndogenousInversionDetector {
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
    pub endogenous_inverter: EndogenousInversionDetector,
    pub word_fatigue: FatigueTracker,
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
        let n_embeds = embeddings.nrows();
        let embeddings: Vec<Array1<f64>> =
            (0..n_embeds).map(|i| embeddings.row(i).to_owned()).collect();
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
            endogenous_inverter: EndogenousInversionDetector::default(),
            word_fatigue: FatigueTracker::new(n_embeds),
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
        let n_embeds = embeddings.len();
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
            endogenous_inverter: EndogenousInversionDetector::default(),
            word_fatigue: FatigueTracker::new(n_embeds),
            friction_graph: None,
            friction_lambda: 0.5,
        }
    }

    /// Replace the seed markers for the endogenous inverter.
    pub fn with_negation_markers(mut self, markers: &[&str]) -> Self {
        self.endogenous_inverter = EndogenousInversionDetector::new(markers);
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
        self.endogenous_inverter.reset();
        self.word_fatigue.reset();
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
            // V11: Process through endogenous inverter (may invert, may learn)
            let s_before = self.predictive_state();
            let processed = self.endogenous_inverter.process(word, v);
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * &processed;
            self.medium_state = self.alpha_medium * &self.medium_state + (1.0 - self.alpha_medium) * &processed;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * &processed;
            self.normalize_states();
            let s_after = self.predictive_state();
            self.endogenous_inverter.observe_flip(&s_before, &s_after);
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
            if emitted.contains(&i) || self.word_fatigue.is_isolated(i) {
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
            .filter(|i| !emitted.contains(i) && !self.word_fatigue.is_isolated(*i))
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

            // Anti-loop via fatigue (safety net — emitted set normally prevents repeats)
            if generated.last().map_or(false, |w| *w == next_word) {
                let just_isolated = self.word_fatigue.record_action(next_idx);
                eprintln!(
                    "  [step {}] loop on '{}', fatigue={:.1}{}",
                    step, next_word, self.word_fatigue.fatigue_of(next_idx),
                    if just_isolated { " [ISOLATED]" } else { "" }
                );
            }
            self.word_fatigue.tick_decay();

            generated.push(next_word.clone());
            last_context = Some(generated.last().unwrap().as_str());
            emitted.insert(next_idx);

            let raw_word_vec = self.embeddings[next_idx].clone();
            // V10: If the word has a larger dimension, grow LIF states first
            self.ensure_dim(raw_word_vec.len());

            // V11: Process through endogenous inverter — may invert the word
            // and learn new negation triggers from trajectory flips
            let s_before_lif = self.predictive_state();
            let word_vec = self.endogenous_inverter.process(&next_word, &raw_word_vec);
            let prev_slow = self.slow_state.clone();
            let prev_medium = self.medium_state.clone();
            let prev_fast = self.fast_state.clone();

            // V9: Triple-LIF update
            self.slow_state = self.alpha_slow * &self.slow_state + (1.0 - self.alpha_slow) * &word_vec;
            self.medium_state = self.alpha_medium * &self.medium_state + (1.0 - self.alpha_medium) * &word_vec;
            self.fast_state = self.alpha_fast * &self.fast_state + (1.0 - self.alpha_fast) * &word_vec;

            self.normalize_states();

            // V11: Observe flip — if the trajectory changed by >90°, credit the trigger word
            let s_after_lif = self.predictive_state();
            self.endogenous_inverter.observe_flip(&s_before_lif, &s_after_lif);

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
