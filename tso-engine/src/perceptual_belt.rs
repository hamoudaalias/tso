/// PerceptualBelt — Deep module that fuses 5 shallow representation modules
/// PerceptualBelt — Deep module that fuses 5 shallow representation modules
/// PerceptualBelt — Deep module that fuses 5 shallow representation modules
/// into one pipe. Interface: process(), recall(), reset(), lif_state().
///
/// Encapsulates: attention, working_memory (DualLIF + associative),
/// attractor_field (prototypes), grid_cells, optional encoder, and all
/// per-concept tracking vectors (episode_trace, concept_values, etc).
///
/// Three categorization backends are selected by CognitiveConfig flags:
///   1. FPI (active inference) — use_fpi = true
///   2. Encoder (AttractorEncoder / VaeEncoder) — encoder.is_some()
///   3. AttractorField — attractor = true (fallback default)
use ndarray::Array1;
use serde::{Serialize, Deserialize};

use crate::neurons::DualLIFState;
use crate::working_memory::WorkingMemory;
use crate::attractor::AttractorField;
use crate::attention::Attention;
use crate::grid_cells::GridCells;
use crate::encoder::Encoder;
use crate::inference;

#[derive(Serialize, Deserialize)]
pub struct PerceptualBelt {
    // ── Sub-modules (internal) ──────────────────────────────────────
    pub(crate) attention: Attention,
    pub(crate) working_mem: WorkingMemory,
    pub(crate) attractor: AttractorField,
    pub(crate) grid_cells: GridCells,

    // Non-optional encoder would break serde skip; kept optional
    #[serde(skip)]
    pub(crate) encoder: Option<Box<dyn Encoder>>,

    // ── Tracking state (moved out of TsoEngine) ───────────────────
    pub(crate) current_concept_id: Option<usize>,
    pub(crate) episode_trace: Vec<usize>,
    pub(crate) concept_values: Vec<f64>,
    pub(crate) predicted_concept_id: Option<usize>,
    pub(crate) novelty_threshold: f64,
    pub(crate) concept_novelty_thresholds: Vec<f64>,
    pub(crate) concept_local_error: Vec<f64>,
    pub(crate) last_active_step: Vec<usize>,
    pub(crate) concept_maturation: Vec<usize>,

    // Percept stored for reference return
    pub(crate) last_percept: Option<Box<Percept>>,

    // Encoding dimension from environment
    dim: usize,
}

/// Output of a single process() tick: what TsoEngine needs to continue
/// the cognitive cycle (episodic prediction, graph, cerebellum).
#[derive(Clone, Serialize, Deserialize)]
pub struct Percept {
    pub concept_id: usize,
    pub gated: Array1<f64>,
    pub intrinsic: f64,
    pub shaping: f64,
    pub is_new: bool,
}

impl Default for PerceptualBelt { fn default() -> Self { Self::new(0) } }
impl PerceptualBelt {
    pub fn new(dim: usize) -> Self {
        PerceptualBelt {
            attention: Attention::new(1.0),
            working_mem: WorkingMemory::new(dim, 0.95, 0.5),
            attractor: AttractorField::new(dim, 8, 3, 0.01),
            grid_cells: GridCells::new(0, 0),
            encoder: None,
            current_concept_id: None,
            episode_trace: Vec::new(),
            concept_values: Vec::new(),
            predicted_concept_id: None,
            novelty_threshold: 0.15,
            concept_novelty_thresholds: Vec::new(),
            concept_local_error: Vec::new(),
            last_active_step: Vec::new(),
            concept_maturation: Vec::new(),
            last_percept: None,
            dim,
        }
    }

    /// Main entry point: perception → concept_id + gated copy.
    /// Handles attention gating, working memory, categorization
    /// (FPI / encoder / attractor), curiosity computation, and
    /// per-concept tracking in a single call.
    pub fn process(
        &mut self,
        perception: &Array1<f64>,
        bfs_value: Option<f64>,
        bfs_bias: &[f64],
        use_fpi: bool,
        use_attractor: bool,
        use_curiosity: bool,
        use_attention: bool,
    ) -> &Percept {
        // ── 0. SPATIAL ATTENTION ──────────────────────────────────
        let (gated, used_raw) = if use_attention {
            let predicted_proto = self.predicted_concept_id
                .and_then(|id| self.attractor.get_prototype(id).cloned());
            (self.attention.attend(perception, predicted_proto.as_ref()), false)
        } else {
            (perception.clone(), true)
        };
        // ponytail: gated accessible via last_percept

        // ── 1. WORKING MEMORY ─────────────────────────────────────
        self.working_mem.observe(&[gated.clone()]);

        // ── 2. CATEGORIZATION ─────────────────────────────────────
        let (cid, is_new, intrinsic, shaping) = if use_fpi {
            self.categorize_fpi(&gated, bfs_value)
        } else if self.encoder.is_some() {
            // Take encoder, call, put back
            let mut enc = self.encoder.take().unwrap();
            let cid = self.categorize_encoder(&gated, perception, used_raw, bfs_value, use_curiosity, &mut enc);
            self.encoder = Some(enc);
            cid
        } else if use_attractor {
            self.categorize_attractor(&gated, perception, used_raw, bfs_value, use_curiosity)
        } else {
            self.categorize_null()
        };

        self.current_concept_id = Some(cid);
        self.episode_trace.push(cid);
        self.ensure_concept_vecs(cid);

        if is_new {
            if let Some(bv) = bfs_value {
                self.concept_values[cid] = bv;
            }
        }

        // Compute shaping reward (value difference with previous concept)
        let prev = if self.episode_trace.len() >= 2 {
            Some(self.episode_trace[self.episode_trace.len() - 2])
        } else { None };
        let shaping = shaping.max(match prev {
            Some(p) if p < self.concept_values.len() && cid < self.concept_values.len() => {
                self.concept_values[cid] - self.concept_values[p]
            }
            _ => 0.0,
        });

        self.last_percept = Some(Box::new(Percept {
            concept_id: cid,
            gated,
            intrinsic,
            shaping,
            is_new,
        }));
        self.last_percept.as_ref().unwrap()
    }



    fn categorize_fpi(&mut self, gated: &Array1<f64>, _bfs_value: Option<f64>) -> (usize, bool, f64, f64) {
        let A_ident: Vec<ndarray::ArrayD<f64>> = vec![ndarray::Array2::eye(self.dim).into_dyn()];
        let obs_onehot: Vec<ndarray::Array1<f64>> = vec![gated.clone()];
        let result = inference::infer_states(&A_ident, &obs_onehot, None, 10);
        let cid = result.concept_id;
        (cid, false, 0.0, 0.0)
    }

    fn categorize_encoder(
        &self,
        gated: &Array1<f64>,
        perception: &Array1<f64>,
        used_raw: bool,
        bfs_value: Option<f64>,
        use_curiosity: bool,
        enc: &mut Box<dyn Encoder>,
    ) -> (usize, bool, f64, f64) {
        let result = enc.encode_raw(gated);
        let cid = result.category_id;
        let new_flag = result.is_new;

        let surp = if use_curiosity {
            self.compute_surprise(if used_raw { gated } else { perception }, cid, new_flag)
        } else { 0.0 };

        (cid, new_flag, surp, 0.0)
    }

    fn categorize_attractor(
        &mut self,
        gated: &Array1<f64>,
        perception: &Array1<f64>,
        used_raw: bool,
        bfs_value: Option<f64>,
        use_curiosity: bool,
    ) -> (usize, bool, f64, f64) {
        // Pre-fetch threshold before mutable borrow of self
        let (cid0, dist) = self.attractor.predict_with_distance(gated);
        let threshold = self.concept_novelty_thresholds.get(cid0).copied().unwrap_or(self.novelty_threshold);
        let new_flag = dist > threshold;
        let cid = if new_flag { self.attractor.add_class(gated) } else { cid0 };

        // Local copies to avoid simultaneous field borrows
        let lr = self.attractor.lr;
        let thr = self.novelty_threshold;
        let in_crit = cid < self.concept_maturation.len() && self.concept_maturation[cid] > 0;

        let boost_lr = if in_crit { lr * 3.0 } else { lr };
        let boost_thr = if in_crit { thr * 0.5 } else { thr };
        self.attractor.lr = boost_lr;
        self.novelty_threshold = boost_thr;

        self.adapt_novelty_threshold(cid, dist, new_flag);
        self.attractor.train_step(gated, cid);

        self.attractor.lr = lr;
        self.novelty_threshold = thr;

        let surp = if use_curiosity {
            self.compute_surprise(if used_raw { gated } else { perception }, cid, new_flag)
        } else { 0.0 };

        (cid, new_flag, surp, 0.0)
    }

    fn categorize_null(&self) -> (usize, bool, f64, f64) {
        (0, false, 0.0, 0.0)
    }

    fn ensure_concept_vecs(&mut self, cid: usize) {
        while self.concept_values.len() <= cid {
            self.concept_values.push(0.0);
        }
        while self.concept_novelty_thresholds.len() <= cid {
            self.concept_novelty_thresholds.push(self.novelty_threshold);
        }
        while self.concept_local_error.len() <= cid {
            self.concept_local_error.push(0.0);
        }
        while self.last_active_step.len() <= cid {
            self.last_active_step.push(0);
        }
        while self.concept_maturation.len() <= cid {
            self.concept_maturation.push(0);
        }
    }

    fn adapt_novelty_threshold(&mut self, concept_id: usize, dist: f64, is_new: bool) {
        if concept_id >= self.concept_local_error.len() {
            return;
        }
        let ema = 0.9 * self.concept_local_error[concept_id] + 0.1 * dist;
        self.concept_local_error[concept_id] = ema;
        let target = if is_new { 0.5 } else { 0.15 };
        let err = target - ema;
        let adjustment = 0.01 * err;
        self.concept_novelty_thresholds[concept_id] += adjustment;
        self.concept_novelty_thresholds[concept_id] =
            self.concept_novelty_thresholds[concept_id].clamp(0.01, 1.0);
    }

    fn compute_surprise(&self, perception: &Array1<f64>, concept_id: usize, is_new: bool) -> f64 {
        if is_new { return 1.0; }
        if let Some(proto) = self.attractor.get_prototype(concept_id) {
            let d = (perception - proto).mapv(|x| x * x).sum().sqrt();
            (d / self.novelty_threshold.max(0.01)).min(1.0)
        } else { 0.0 }
    }

    // ── Public accessors (Thin, for TsoEngine internal use) ──────

    pub fn recall(&self, query: &Array1<f64>) -> Option<(usize, f64)> {
        self.working_mem.recall(query)
    }

    pub fn reset(&mut self) {
        self.working_mem.reset();
        self.episode_trace.clear();
        self.current_concept_id = None;
    }

    pub fn lif_state(&self) -> &DualLIFState {
        &self.working_mem.lif
    }

    pub fn configure(&mut self, w: usize, h: usize) {
        self.grid_cells.auto_configure(w, h);
    }

    pub fn extra_dim(&self) -> usize {
        self.grid_cells.extra_dim()
    }

    pub fn num_concepts(&self) -> usize {
        self.attractor.n_classes()
    }

    pub fn set_encoder(&mut self, enc: Box<dyn Encoder>) {
        self.encoder = Some(enc);
    }

    pub fn concept_values(&self) -> &[f64] { &self.concept_values }
    pub fn concept_values_mut(&mut self) -> &mut [f64] { &mut self.concept_values }
    pub fn predicted_concept_id(&self) -> Option<usize> { self.predicted_concept_id }
    pub fn set_predicted_concept_id(&mut self, id: Option<usize>) { self.predicted_concept_id = id; }
    pub fn get_prototype(&self, id: usize) -> Option<&Array1<f64>> {
        self.attractor.get_prototype(id)
    }
    pub fn episode_trace(&self) -> &[usize] { &self.episode_trace }
    pub fn concept_maturation(&self) -> &[usize] { &self.concept_maturation }
    pub fn concept_maturation_mut(&mut self) -> &mut [usize] { &mut self.concept_maturation }
    pub fn last_active_step_mut(&mut self) -> &mut [usize] { &mut self.last_active_step }
    pub fn dim(&self) -> usize { self.dim }
}
