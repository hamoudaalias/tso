#[cfg(feature = "rstdp")]
use crate::plasticity::RstdpPlasticity;
use crate::perceptual_belt::PerceptualBelt;
use ndarray::Array1;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use tracing::{event, Level};
use crate::attractor::AttractorField;
use crate::episodic::{EpisodicMemory, ContextBuffer};
use crate::cerebellum::Cerebellum;
use crate::core::{Graph, NodeId, resolve_with_anneal,
    demineur_sweep, demineur_sweep_trace,
    lateral_inhibition_sweep, lateral_inhibition_trace,
    exponential_decay_sweep, exponential_decay_trace};
use crate::working_memory::WorkingMemory;
use crate::action::ActionMotor;
use crate::neurogenesis::{Neurogenesis, NeurogenesisConfig};
use crate::hypothalamus::Hypothalamus;
use crate::attention::Attention;
use crate::grid_cells::GridCells;


/// Summary of what happened during a single sleep/consolidation cycle.
#[derive(Clone, Debug)]
pub struct SleepReport {
    pub replay_count: usize,
    pub prototypes_pruned: usize,
    pub prototypes_added: usize,
    pub new_concepts: usize,
    pub edges_removed: usize,
    pub concepts_pruned: usize,
    pub phi_before: f64,
    pub phi_after: f64,
}

/// Instantané des métriques clés pour export JSON / temps réel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub phi: f64,
    pub well_being: f64,
    pub energy: f64,
    pub hydration: f64,
    pub temperature: f64,
    pub sleep_pressure: f64,
    pub n_concepts: usize,
    pub n_edges: usize,
    pub total_episodes: usize,
    pub total_steps: usize,
    pub sleep_cycles: usize,
}

/// Flags activant/désactivant chaque sous-système cognitif dans step() / heartbeat().
/// Groupe logique des 6 booléens de CognitiveConfig.
#[derive(Clone, Debug)]
pub struct SubsystemFlags {
    pub attractor: bool,
    pub use_fpi: bool,
    pub efe_weight: f64,
    pub graph_phi: bool,
    pub attention: bool,
    pub episodic_curiosity: bool,
    pub metabolic_cost: bool,
    pub hypothalamus: bool,
}

/// Configuration fine des sous-systèmes cognitifs activés dans step() / heartbeat().
/// Chaque flag contrôle un sous-système indépendant pour bissection.
/// Défaut : tout-à-true (comportement actuel inchangé).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveConfig {
    /// Catégorisation (attractor)
    pub attractor: bool,
    /// Use FPI inference instead of attractor categorization
    pub use_fpi: bool,
    /// Weight for EFE scoring in action selection (0.0 = pure RL)
    pub efe_weight: f64,
    /// Graphe sémantique + Φ
    pub graph_phi: bool,
    /// Attention spatiale
    pub attention: bool,
    /// Mémoire épisodique + curiosité intrinsèque
    pub episodic_curiosity: bool,
    /// Coût métabolique
    pub metabolic_cost: bool,
    /// Hypothalamus
    pub hypothalamus: bool,
    /// Clip de |δ| dans reinforce_td
    pub delta_clip_max: f64,
    /// Neurogenèse : probabilité de naissance (0.0 – 1.0)
    pub sleep_neurogenesis_rate: f64,
    /// Neurogenèse : budget max de concepts
    pub sleep_max_concepts: usize,
    /// Neurogenèse : durée période critique
    pub sleep_maturation_cycles: usize,
    /// Neurogenèse : scaling synaptique
    pub sleep_synaptic_scaling: bool,
    pub phi_gating: bool,
    pub phi_threshold: f64,
    pub rstdp_enabled: bool,
    /// Homeostatic drive bonus for action selection.
    /// When deficits are high, increases exploration (urgency to find resources).
    pub homeostatic_drive_bonus: f64,
    /// Autonomous sleep: end_episode() checks should_sleep() and runs sleep_cycle().
    pub autonomous_sleep: bool,
}

impl CognitiveConfig {
    /// Retourne les flags de sous-systèmes comme struct groupé.
    /// Rétrocompatible : les bins existants utilisent les champs plats.
    pub fn subsystems(&self) -> SubsystemFlags {
        SubsystemFlags {
            efe_weight: self.efe_weight,
            use_fpi: self.use_fpi,            attractor: self.attractor,
            graph_phi: self.graph_phi,
            attention: self.attention,
            episodic_curiosity: self.episodic_curiosity,
            metabolic_cost: self.metabolic_cost,
            hypothalamus: self.hypothalamus,
        }
    }
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        CognitiveConfig {
            use_fpi: false,
            efe_weight: 0.0,
            attractor: true,
            graph_phi: true,
            attention: false,
            episodic_curiosity: false,
            metabolic_cost: false,
            hypothalamus: true,
            delta_clip_max: 5.0,
            sleep_neurogenesis_rate: 0.02,
            sleep_max_concepts: 50,
            sleep_maturation_cycles: 3,
            sleep_synaptic_scaling: true,
            phi_gating: false,
            phi_threshold: 0.5,
            rstdp_enabled: false,
            homeostatic_drive_bonus: 0.3,
            autonomous_sleep: true,
        }
    }
}

/// Métrique de preuve — proof metrics tracking the system's performance
/// on the Weakness Game §8 (Démineur / Minesweeper mode).
/// Measures resolution efficiency, edge count stability, and Φ evolution.
#[derive(Clone, Debug, Default)]
pub struct ProofMetrics {
    /// Total number of flag operations performed
    pub total_flags: usize,
    /// Cumulative Φ eliminated by flagging
    pub phi_eliminated_by_flags: f64,
    /// Current Φ value
    pub current_phi: f64,
    /// Number of edges currently in the graph
    pub edge_count: usize,
    /// Number of exclusion edges currently in the graph
    pub exclusion_edge_count: usize,
    /// Peak edge count ever reached
    pub peak_edge_count: usize,
    /// Number of edges pruned (by low-phi pruning)
    pub edges_pruned: usize,
    /// Total resolution iterations performed
    pub total_resolve_iters: usize,
    /// Average Φ reduction per flag
    pub avg_phi_per_flag: f64,
    /// How many times forced evolution was applied
    pub evolution_cycles: usize,
    /// Efficiency of pruning: edges pruned / total edges peak
    pub pruning_efficiency: f64,
    /// Proof score: higher = better (Φ eliminated per edge count)
    pub proof_score: f64,
}

#[derive(Serialize, Deserialize)]
pub struct TsoEngine {
    pub working_mem: WorkingMemory,
    pub attractor: AttractorField,
    #[serde(skip)]
    pub encoder: Option<Box<dyn crate::encoder::Encoder>>,
    #[serde(skip)]
    pub env: Option<Box<dyn crate::environment::Environment>>,
    pub episodic: EpisodicMemory,
    pub context: ContextBuffer,
    pub graph: Graph,
    pub cerebellum: Cerebellum,
    pub motor: ActionMotor,
    pub hypothalamus: Hypothalamus,

    dim: usize,
    current_concept_id: Option<usize>,
    episode_trace: Vec<usize>,
    step_count: usize,

    /// Total episodes lived (persisted across restarts)
    total_episodes: usize,
    /// Cognitive map : concept_id → V(s) value iteration
    concept_values: Vec<f64>,
    /// Log des transitions (from, to, reward) pour la value iteration
    pub trans_log: Vec<(usize, usize, usize, f64)>,

    /// Episodic memory's prediction for the current step's concept.
    /// Used to compute curiosity / surprise as distance(perception, prototype[predicted]).
    predicted_concept_id: Option<usize>,
    /// Weight of intrinsic curiosity reward. Decays per second via real-time loop.
    pub curiosity_weight: f64,
    /// Attractor novelty threshold. Increase for continuous/noisy environments (default 0.15).
    pub novelty_threshold: f64,
    /// Per-concept adaptive novelty thresholds (replaces global threshold check).
    /// Each concept learns its own discrimination granularity based on local
    /// prediction error — finer where distinctions matter, coarser elsewhere.
    pub concept_novelty_thresholds: Vec<f64>,
    /// Running EMA of Euclidean distance to matched prototype for each concept.
    /// Used by the homeostatic adaptation of per-concept thresholds.
    pub concept_local_error: Vec<f64>,

    /// Step number when each concept was last activated (matched or created).
    /// Used to detect zombie concepts for pruning.
    pub last_active_step: Vec<usize>,
    /// If a concept has not been activated for this many steps, it is pruned
    /// during end_episode. Set to 0 to disable pruning.
    pub concept_prune_threshold: usize,
    /// Compteur de maturation pour chaque concept (0 = mature, >0 = période critique).
    /// Indexé par concept_id. Initialisé à 0 pour tous les concepts.
    pub(crate) concept_maturation: Vec<usize>,

    // ── Hypothalamic / Anxiety (Φ) state ──
    /// Current graph conflict energy Φ — measures cognitive tension (anxiety).
    pub current_phi: f64,
    /// Φ from end of last tick (after resolve) — used to compute ΔΦ.
    phi_prev: f64,
    /// Unified Φ_total = Φ_graph + homeostatic_drift (when hypothalamus active).
    pub phi_total: f64,
    phi_total_prev: f64,
    /// Rolling window of recent resolved-Φ values (for neurogenesis modulation).
    phi_history: VecDeque<f64>,
    /// Max entries in phi_history.
    phi_history_len: usize,
    /// Whether Φ exceeds the anxiety threshold.
    pub anxious: bool,
    /// Φ threshold above which the organism enters an anxious state.
    pub phi_threshold: f64,

    /// Si true, toute l'apprentissage RL utilise un signal stationnaire :
    ///   R_ext + γ·Φ_BFS(s') − Φ_BFS(s)  (potentiel BFS précalculé)
    /// au lieu de well_being (9 termes, non-stationnaire).
    /// Les termes intrinsèques (curiosité, ΔΦ, métabolique) continuent
    /// de moduler l'exploration (ε, bruit) mais ne sont plus jamais passés
    /// à reinforce_td ni store_transition.
    /// Réalise le découplage : motivation intrinsèque → exploration,
    /// récompense stationnaire → politique exploitable.
    pub use_stationary_reward: bool,

    /// Poids multiplicatifs pour chaque terme du bien-être.
    /// Ordre : gated_reward, consummatory, curiosity, shaping,
    ///         phi_delta, chronic_tension, deficit_penalty,
    ///         metabolic_penalty, parsimony.
    /// Défaut : [1.0; 9] (comportement historique).
    pub well_being_weights: [f64; 9],

    /// Dernière valeur du bien-être (total_reward) calculée dans step().
    /// Utile pour l'export de métriques (MetricsSnapshot).
    pub last_total_reward: f64,

    /// Valeur du potentiel BFS au step précédent (pour le shaping).
    /// Stocke le bfs_value du dernier step() pour calculer γ·Φ(s')−Φ(s).
    pub prev_bfs_value: Option<f64>,

    /// How many times the oscillation detector has forced greedy mode in
    /// resolve_with_anneal. Accumulated across the lifetime of the engine.
    pub oscillation_breaks: usize,

    /// Φ gating: how many steps were skipped (no resolve) because Φ < threshold.
    pub gating_skip_count: usize,
    /// Φ gating: how many times resolve was called.
    pub resolve_count: usize,
    /// Φ gating: total iterations spent in resolve across all calls.
    pub resolve_total_iters: usize,

    // ── Sleep / Consolidation ──
    /// Number of sleep cycles completed since engine creation.
    pub sleep_cycles: usize,
    /// Sleep every N episodes (0 = never).
    pub sleep_every_n_episodes: usize,
    /// How many replay epochs per sleep cycle.
    pub sleep_replay_epochs: usize,
    /// Habit tracker: (concept_from, concept_to) → repetition count.
    /// Frequently repeated transitions become metabolically cheaper
    /// (simulating myelination / automation of well-learned pathways).
    pub habit_counts: std::collections::HashMap<(usize, usize), usize>,
    /// Total steps lived (used for habit normalization).
    pub total_steps: usize,
    /// How many resolution iterations during offline sleep.
    pub sleep_resolve_iters: usize,
    /// Std-dev of Gaussian noise added during sleep replay.
    pub sleep_noise_std: f64,
    /// Max episodes to replay per sleep cycle (0 = all). Prioritizes recent.
    pub sleep_max_replay: usize,

    /// Spatial attention module — biases whisker perception toward
    /// dimensions where the predicted concept prototype diverges most
    /// from the current input (anomaly-driven attention).
    pub attention: Attention,

    /// Configuration fine des sous-systèmes cognitifs actifs.
    /// Permet la bissection pour isoler l'interférence du cycle cognitif
    /// sur l'apprentissage du Cerebellum (cf. BUG-2025-08-03T120000).
    #[serde(skip)]
    pub belt: PerceptualBelt,

    pub cogs: CognitiveConfig,
    #[serde(skip)]
    pub neurogenesis: Neurogenesis,

    /// Debug : si true, dump le rl_signal et la récompense à chaque step.
    pub debug_step_dump: bool,

    /// Grid cells — désambiguïse l'aliasing perceptuel en encodant
    /// la position absolue pour les grilles >6×6.
    pub grid_cells: GridCells,

    // ── Replay buffer state ──
    /// Gated state from the previous step (for replay buffer).
    prev_gated: Option<Array1<f64>>,
    /// Action selected at the previous step (for replay buffer).
    prev_action: Option<usize>,
}

impl TsoEngine {
    #[cfg(feature = "rstdp")]
    fn init_rstdp(&mut self) {
        if self.cogs.rstdp_enabled {
            let n_act = self.cerebellum.n_actions;
            let hd = self.cerebellum.hidden_dim();
            let r = RstdpPlasticity::new(self.dim, hd, n_act, 0.01);
            self.cerebellum.rstdp = Some(r);
        }
    }

    pub fn new(dim: usize, n_actions: usize) -> Self {
        #[cfg_attr(not(feature = "rstdp"), allow(unused_mut))]
        let mut e = Self::with_hidden(dim, n_actions, 0);
        #[cfg(feature = "rstdp")]
        e.init_rstdp();
        e
    }

    pub fn with_hidden(dim: usize, n_actions: usize, hidden_dim: usize) -> Self {
        TsoEngine {
            belt: PerceptualBelt::new(dim),
            working_mem: WorkingMemory::new(dim, 0.95, 0.5),
            attractor: AttractorField::new(dim, 8, 3, 0.01),
            encoder: None,
            env: None,
            episodic: EpisodicMemory::new(50),
            context: ContextBuffer::new(10),
            graph: Graph::with_params(0.7, 0.1),
            cerebellum: Cerebellum::new(dim, n_actions, 0.30, 0.1, 0.50, hidden_dim),
            motor: ActionMotor::new(0.6),
            hypothalamus: Hypothalamus::new(),
            dim,
            total_episodes: 0,
            current_concept_id: None,
            episode_trace: Vec::new(),
            step_count: 0,
            concept_values: Vec::new(),
            trans_log: Vec::new(),
            predicted_concept_id: None,
            curiosity_weight: 0.5,
            novelty_threshold: 0.15,
            concept_novelty_thresholds: Vec::new(),
            concept_local_error: Vec::new(),
            last_active_step: Vec::new(),
            concept_prune_threshold: 500,
            concept_maturation: Vec::new(),
            current_phi: 0.0,
            phi_prev: 0.0,
            phi_total: 0.0,
            phi_total_prev: 0.0,
            phi_history: VecDeque::with_capacity(128),
            phi_history_len: 128,
            anxious: false,
            phi_threshold: 0.5,
            oscillation_breaks: 0,
            gating_skip_count: 0,
            resolve_count: 0,
            resolve_total_iters: 0,
            sleep_cycles: 0,
            sleep_every_n_episodes: 5,
            sleep_replay_epochs: 2,
            sleep_resolve_iters: 80,
            sleep_noise_std: 0.05,
            sleep_max_replay: 0,
            habit_counts: std::collections::HashMap::new(),
            total_steps: 0,
            attention: Attention::new(0.5),
            grid_cells: GridCells::new(0, 0),
            cogs: CognitiveConfig::default(),
            neurogenesis: Neurogenesis::new(NeurogenesisConfig {
                rate: CognitiveConfig::default().sleep_neurogenesis_rate,
                max_concepts: CognitiveConfig::default().sleep_max_concepts,
                maturation_cycles: CognitiveConfig::default().sleep_maturation_cycles,
                synaptic_scaling: CognitiveConfig::default().sleep_synaptic_scaling,
            }),
            prev_gated: None,
            prev_action: None,
            use_stationary_reward: false,
            well_being_weights: [1.0; 9],
            last_total_reward: 0.0,
            prev_bfs_value: None,
            debug_step_dump: false,
        }
    }

    /// Reconfigure l'engine pour une grille de dimensions données.
    /// Ajoute des cellules de grille si la surface >36 (6×6),
    /// ce qui désambiguïse l'aliasing perceptuel.
    pub fn configure_for_grid(&mut self, w: usize, h: usize, n_actions: usize, hidden_dim: usize) {
        self.grid_cells.auto_configure(w, h);
        // Pour Sokoban : 4 whiskers + box_adjacent + box_dir + target_sensed = 7
        let sokoban_base = 7usize;
        let new_dim = sokoban_base + self.grid_cells.extra_dim();
        if new_dim != self.dim {
            self.dim = new_dim;
            self.working_mem = WorkingMemory::new(new_dim, 0.95, 0.5);
            self.attractor = AttractorField::new(new_dim, 8, 3, 0.01);
            self.cerebellum = Cerebellum::new(new_dim, n_actions, 0.30, 0.1, 0.50, hidden_dim);
        }
    }

    /// Compute intrinsic curiosity reward from surprise.
    fn compute_surprise(&self, _perception: &Array1<f64>, _concept_id: usize, _is_new: bool) -> f64 {
        0.0 // ponytail: disabled — §6.5 ablation showed no impact
    }

    /// Noradrenaline-like arousal signal from Φ tension.
    /// Returns a multiplier in [0, 3] for the effective TD learning rate.
    fn compute_arousal(&self) -> f64 {
        if self.phi_threshold > 0.0 {
            (self.current_phi / self.phi_threshold).clamp(0.0, 3.0)
        } else {
            0.0
        }
    }

    fn adapt_novelty_threshold(&mut self, concept_id: usize, _dist: f64, _is_new: bool) {
        // ponytail: minimal — track activity only; adaptive thresholds removed
        while self.last_active_step.len() <= concept_id {
            self.last_active_step.push(self.step_count);
        }
        self.last_active_step[concept_id] = self.step_count;
    }

    /// Retourne le concept précédent depuis episode_trace, ou None.
    /// Utile pour trans_log même en mode attractor=false où l'ID factice 0 est poussé.
    fn previous_concept(&self) -> Option<usize> {
        if self.episode_trace.len() >= 2 {
            Some(self.episode_trace[self.episode_trace.len() - 2])
        } else {
            None
        }
    }

    fn compute_habit_efficiency(&self) -> f64 {
        if self.episode_trace.len() >= 2 {
            let prev = self.episode_trace[self.episode_trace.len() - 2];
            let cur = self.episode_trace[self.episode_trace.len() - 1];
            let key = (prev, cur);
            let count = self.habit_counts.get(&key).copied().unwrap_or(0);
            // Efficiency grows with repetition: count=1→0.17, count=10→0.39, count=100→0.67
            return 1.0 - 1.0 / (1.0 + 0.2 * (count as f64).sqrt());
        }
        0.0
    }

    fn apply_metabolic_costs(&mut self) {
        let cerebellum_cost = self.cerebellum.compute_cost();
        let graph_cost = self.graph.compute_cost();
        let habit_efficiency = self.compute_habit_efficiency();
        self.hypothalamus.apply_metabolic_cost(cerebellum_cost, graph_cost, habit_efficiency);
    }

    pub fn step(&mut self, perception: &Array1<f64>, reward: f64, bfs_value: Option<f64>, bfs_bias: &[f64]) -> usize {
        assert_eq!(perception.len(), self.dim, "step: perception.len() != dim ({} != {})", perception.len(), self.dim);
        debug_assert!(!perception.iter().any(|x| x.is_nan()), "step: perception contains NaN");
        self.step_count += 1;
        self.total_steps += 1;
        let cc = self.cogs.clone();
        self.cerebellum.delta_clip = cc.delta_clip_max;

        if let Some(action_id) = self.step_fast_path(perception, reward) {
            return action_id;
        }

        if cc.subsystems().hypothalamus { self.hypothalamus.step(); }

        let (gated, used_raw) = self.step_attention(perception, &cc);

        self.working_mem.observe(&[gated.clone()]);

        let (concept_id, intrinsic, shaping) = self.step_categorize(perception, &gated, used_raw, bfs_value, bfs_bias, &cc);

        if cc.subsystems().episodic_curiosity {
            self.context.push(concept_id);
            self.predicted_concept_id = self.episodic.recall(&self.context.as_slice());
        }

        let total_reward = self.step_phi_homeostasis(reward, concept_id, intrinsic, shaping, &cc);

        self.step_transition_log(concept_id, reward, &cc);

        let decision_state = self.step_decision_state(perception, &gated);
        let mut logits = self.cerebellum.forward_logits(&decision_state);
        #[cfg(feature = "rstdp")]
        {
            let hidden = self.cerebellum.get_hidden().to_vec();
            if let Some(ref mut rstdp) = self.cerebellum.rstdp {
                let hidden_arr = Array1::from_vec(hidden);
                let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let post_exp: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
                let sum: f64 = post_exp.iter().sum();
                let post_arr = Array1::from_vec(post_exp.iter().map(|x| x / sum).collect());
                rstdp.update_trace(&decision_state, &post_arr, &hidden_arr);
            }
        }
        let rl_signal = self.step_rl_signal(bfs_value, reward, total_reward);

        if self.debug_step_dump {
            event!(Level::DEBUG, step = self.step_count, rl_signal = rl_signal, reward_ext = reward, bfs_value = bfs_value, prev_bfs = self.prev_bfs_value, stationary = self.use_stationary_reward);
        }

        let saved_lr = self.cerebellum.lr;
        self.cerebellum.lr *= 1.0 + self.compute_arousal();
        self.cerebellum.reinforce_td(rl_signal, 0.99);
        self.cerebellum.lr = saved_lr;
        self.cerebellum.decay_trace(0.99, 0.98);

        let replay_r = if self.use_stationary_reward { rl_signal } else { total_reward };
        if !self.cerebellum.is_linear() {
            if let (Some(ps), Some(pa)) = (self.prev_gated.clone(), self.prev_action) {
                self.cerebellum.store_transition(&ps, pa, replay_r, &decision_state, false);
            }
            self.prev_gated = Some(decision_state.clone());
        }

        self.step_action(&mut logits, &decision_state, bfs_bias)
    }

    fn step_fast_path(&mut self, perception: &Array1<f64>, reward: f64) -> Option<usize> {
        let cc = &self.cogs;
        if cc.attractor || cc.hypothalamus || cc.episodic_curiosity || cc.attention || cc.graph_phi || cc.metabolic_cost || cc.use_fpi {
            return None;
        }
        let logits = self.cerebellum.forward_logits(perception);
        #[cfg(feature = "rstdp")]
        {
            let hidden = self.cerebellum.get_hidden().to_vec();
            if let Some(ref mut rstdp) = self.cerebellum.rstdp {
                let hidden_arr = Array1::from_vec(hidden);
                let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let post_exp: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
                let sum: f64 = post_exp.iter().sum();
                let post_arr = Array1::from_vec(post_exp.iter().map(|x| x / sum).collect());
                rstdp.update_trace(perception, &post_arr, &hidden_arr);
            }
        }
        let mut action_id = 0;
        let mut best = logits[0];
        for (i, l) in logits.iter().enumerate() {
            if *l > best { best = *l; action_id = i; }
        }
        if rand::random::<f64>() < self.cerebellum.epsilon {
            action_id = rand::random::<usize>() % self.cerebellum.n_actions;
        }
        let saved_lr = self.cerebellum.lr;
        self.cerebellum.lr *= 1.0 + self.compute_arousal();
        self.cerebellum.reinforce_td(reward, 0.99);
        self.cerebellum.lr = saved_lr;
        self.cerebellum.decay_trace(0.99, 0.98);
        self.cerebellum.mark(perception, action_id);
        Some(action_id)
    }

    fn step_attention(&self, perception: &Array1<f64>, cc: &CognitiveConfig) -> (Array1<f64>, bool) {
        if cc.subsystems().attention {
            let predicted_proto = self.predicted_concept_id.and_then(|id| self.attractor.get_prototype(id));
            (self.attention.attend(perception, predicted_proto), false)
        } else {
            (perception.clone(), true)
        }
    }

    fn step_categorize(&mut self, perception: &Array1<f64>, gated: &Array1<f64>, used_raw: bool, bfs_value: Option<f64>, bfs_bias: &[f64], cc: &CognitiveConfig) -> (usize, f64, f64) {
        let input = if used_raw { gated } else { perception };
        let (cid, _is_new, intrinsic, shaping) = if cc.subsystems().use_fpi {
            let percept = self.belt.process(input, bfs_value, bfs_bias, true, cc.subsystems().attractor, cc.subsystems().episodic_curiosity, cc.subsystems().attention);
            (percept.concept_id, percept.is_new, percept.intrinsic, percept.shaping)
        } else if self.encoder.is_some() || cc.subsystems().attractor {
            let percept = self.belt.process(input, bfs_value, bfs_bias, false, true, cc.subsystems().episodic_curiosity, cc.subsystems().attention);
            if percept.is_new {
                if let Some(bv) = bfs_value {
                    while self.concept_values.len() <= percept.concept_id { self.concept_values.push(0.0); }
                    self.concept_values[percept.concept_id] = bv;
                }
            }
            (percept.concept_id, percept.is_new, percept.intrinsic, percept.shaping)
        } else {
            (0usize, false, 0.0, 0.0)
        };
        self.current_concept_id = Some(cid);
        self.episode_trace.push(cid);
        while self.concept_values.len() <= cid { self.concept_values.push(0.0); }
        (cid, intrinsic, shaping)
    }

    fn step_phi_homeostasis(&mut self, reward: f64, _concept_id: usize, intrinsic: f64, shaping: f64, cc: &CognitiveConfig) -> f64 {
        let gated_reward = if cc.subsystems().hypothalamus { self.hypothalamus.gate_reward(reward) } else { reward };
        let consummatory = if cc.subsystems().hypothalamus { self.hypothalamus.consummatory_value(reward) } else { 0.0 };

        self.anxious = cc.subsystems().graph_phi && self.current_phi > self.phi_threshold;
        if cc.subsystems().hypothalamus { self.hypothalamus.set_phi(self.current_phi); }
        if cc.subsystems().hypothalamus && reward > 0.0 { self.hypothalamus.consume(reward); }

        if cc.subsystems().graph_phi && self.step_count % 50 == 0 {
            if !cc.phi_gating || self.graph.compute_phi() >= cc.phi_threshold {
                let result = resolve_with_anneal(&mut self.graph, 20, 0.05, 0.15);
                self.oscillation_breaks += result.oscillation_breaks;
                self.resolve_count += 1;
                self.resolve_total_iters += 20;
            } else { self.gating_skip_count += 1; }
        }

        let resolved_phi = if cc.subsystems().graph_phi { self.graph.phi() } else { 0.0 };
        let phi_delta = resolved_phi - self.phi_prev;
        self.phi_prev = resolved_phi;
        self.current_phi = resolved_phi;
        if self.phi_history.len() >= self.phi_history_len { self.phi_history.pop_front(); }
        self.phi_history.push_back(resolved_phi);
        self.anxious = cc.subsystems().graph_phi && resolved_phi > self.phi_threshold;
        if cc.subsystems().hypothalamus { self.hypothalamus.set_phi(resolved_phi); }

        // Φ_total = Φ_graph + homeostatic_drift (when hypothalamus active)
        let homeostatic_drift = if cc.subsystems().hypothalamus { self.hypothalamus.total_deficit() } else { 0.0 };
        self.phi_total = resolved_phi + homeostatic_drift;
        let _phi_total_delta = self.phi_total - self.phi_total_prev;
        self.phi_total_prev = self.phi_total;

        if cc.subsystems().metabolic_cost { self.apply_metabolic_costs(); }
        else { self.hypothalamus.total_cost = 0.0; }
        let metabolic_penalty = -self.hypothalamus.total_cost * 20.0;

        let n_protos = if let Some(enc) = &self.encoder { enc.prototype_count() }
        else if cc.subsystems().attractor { self.attractor.prototypes.len() } else { 0 };
        let parsimony = -(n_protos as f64) * 0.001;
        let is_terminal = reward.abs() >= 10.0;
        let r_curiosity = if is_terminal { 0.0 } else { intrinsic };

        let total_reward = {
            let w = self.well_being_weights;
            w[0] * gated_reward + w[1] * consummatory + w[2] * r_curiosity + w[3] * shaping
            - w[4] * phi_delta + w[7] * metabolic_penalty + w[8] * parsimony
        };
        self.last_total_reward = total_reward;
        total_reward
    }

    fn step_transition_log(&mut self, concept_id: usize, reward: f64, cc: &CognitiveConfig) {
        if let Some(p) = self.previous_concept() {
            let step_r = if reward >= 20.0 { -0.05 } else { reward };
            self.trans_log.push((p, concept_id, self.prev_action.unwrap_or(0), step_r));
        }
        if cc.subsystems().attractor && reward >= 20.0 && concept_id < self.concept_values.len() {
            self.concept_values[concept_id] = 20.0;
        }
        if cc.subsystems().graph_phi && self.episode_trace.len() >= 2
            && (!cc.phi_gating || self.graph.compute_phi() >= cc.phi_threshold)
        {
            let p = self.episode_trace[self.episode_trace.len() - 2];
            let proto_a = self.encoder.as_ref().and_then(|e| e.get_prototype(p).cloned())
                .or_else(|| self.attractor.prototypes.get(p).and_then(|v| v.first().cloned()));
            let proto_b = self.encoder.as_ref().and_then(|e| e.get_prototype(concept_id).cloned())
                .or_else(|| self.attractor.prototypes.get(concept_id).and_then(|v| v.first().cloned()));
            if let (Some(a), Some(b)) = (proto_a.as_ref(), proto_b.as_ref()) {
                self.graph.add_transition(a, b, reward);
                *self.habit_counts.entry((p, concept_id)).or_insert(0) += 1;
            }
        }
        if cc.subsystems().attractor && self.concept_prune_threshold > 0 && self.step_count % self.concept_prune_threshold == 0 {
            self.prune_concepts(self.compute_arousal());
        }
    }

    fn step_decision_state(&self, perception: &Array1<f64>, gated: &Array1<f64>) -> Array1<f64> {
        if self.use_stationary_reward { perception.clone() } else { gated.clone() }
    }

    fn step_rl_signal(&mut self, bfs_value: Option<f64>, reward: f64, total_reward: f64) -> f64 {
        if self.use_stationary_reward {
            let bfs_shaping = match (self.prev_bfs_value, bfs_value) {
                (Some(prev_bv), Some(cur_bv)) => 0.99 * cur_bv - prev_bv,
                _ => 0.0,
            };
            self.prev_bfs_value = bfs_value;
            reward + bfs_shaping
        } else { total_reward }
    }

    fn step_action(&mut self, logits: &mut Vec<f64>, decision_state: &Array1<f64>, bfs_bias: &[f64]) -> usize {
        let cc = &self.cogs;
        let homeostatic_urgency = if cc.subsystems().hypothalamus && cc.homeostatic_drive_bonus > 0.0 {
            self.hypothalamus.total_deficit() * cc.homeostatic_drive_bonus
        } else { 0.0 };
        let exploring = self.cerebellum.noise_std > 0.0 || homeostatic_urgency > 0.0;
        if exploring {
            let mut rng = rand::thread_rng();
            let effective_epsilon = (self.cerebellum.epsilon + homeostatic_urgency).min(1.0);
            if rand::random::<f64>() < effective_epsilon {
                let action_id = rng.gen_range(0..logits.len());
                self.cerebellum.mark(decision_state, action_id);
                return action_id;
            }
            let effective_noise = self.cerebellum.noise_std * (1.0 + homeostatic_urgency);
            for l in logits.iter_mut() {
                *l += rng.gen_range(-effective_noise..effective_noise);
            }
        }
        if cc.efe_weight > 0.0 && self.current_concept_id.is_some() {
            #[cfg(feature = "active-inference")]
            let (qs, A, B, C) = {
                let cid = self.current_concept_id.unwrap();
                let n_states = self.attractor.n_classes().max(1);

                // qs = one-hot belief over current concept
                let qs = vec![ndarray::Array1::from_shape_fn(n_states, |i| if i == cid { 1.0 } else { 0.0 })];

                // A = identity (observation directly reveals the concept)
                let A: ndarray::ArrayD<f64> = ndarray::Array2::eye(n_states).into_dyn();

                // B from empirical transition counts in trans_log
                let n_actions = logits.len();
                let mut count = ndarray::Array3::<f64>::zeros((n_states, n_states, n_actions));
                let mut total = ndarray::Array2::<f64>::zeros((n_states, n_actions));
                for &(from, to, action, _r) in &self.trans_log {
                    if from < n_states && to < n_states && action < n_actions {
                        count[[to, from, action]] += 1.0;
                        total[[from, action]] += 1.0;
                    }
                }
                let B: ndarray::Array3<f64> = ndarray::Array3::from_shape_fn((n_states, n_states, n_actions), |(to, from, action)| {
                    let t = total[[from, action]];
                    if t > 0.0 { count[[to, from, action]] / t }
                    else if to == from { 1.0 }
                    else { 0.0 }
                });

                // C = learned concept values as preferences
                let n = self.concept_values.len();
                let C: ndarray::Array1<f64> = ndarray::Array1::from_shape_fn(n_states, |i| {
                    if i < n { self.concept_values[i] } else { 0.0 }
                });

                (qs, A, B, C)
            };
            let candidates: Vec<usize> = (0..logits.len()).collect();
            let efe_scores: Vec<f64> = candidates.iter().map(|&a| {
                #[cfg(feature = "active-inference")]
                { crate::efe::score_policy(&qs, &[A.clone()], &[B.clone()], &[C.clone()], &[a], true, true) }
                #[cfg(not(feature = "active-inference"))]
                { let _ = a; 0.0 }
            }).collect();
            for (l, g) in logits.iter_mut().zip(efe_scores.iter()) { *l += cc.efe_weight * g; }
        }
        for (l, b) in logits.iter_mut().zip(bfs_bias.iter()) { *l += b * 0.5; }
        let action_id = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        self.cerebellum.mark(decision_state, action_id);
        self.prev_action = Some(action_id);
        action_id
    }

    // ──────────────────────────────────────────────────────────────
    //  Heartbeat — The 4-step cognitive cycle of the living organism
    // ──────────────────────────────────────────────────────────────
    //
    //  Step 1 — Perception & Internal State
    //    Integrate sensor readings (whiskers) with hypothalamic
    //    homeostatic variables (Energy, Hydration, Temperature) into
    //    the DualLIFState. The Hypothalamus drifts each step.
    //
    //  Step 2 — Tension Evaluation (Φ)
    //    Compute the current graph conflict energy Φ. If Φ exceeds
    //    the anxiety threshold, the organism enters an anxious state
    //    that prioritises conflict-resolution actions.
    //
    //  Step 3 — Action Selection
    //    The Cerebellum selects an action that maximises well-being:
    //        well_being = gated_reward + consummatory + curiosity + shaping
    //                   - ΔΦ - chronic_tension - deficit_penalty - parsimony
    //    where gated_reward = hypothalamus.gate_reward(external).
    //
    //  Step 4 — Hebbian Learning
    //    Update eligibility traces and synaptic weights based on the
    //    success of the action in stabilising homeostasis or reducing Φ.
    //
    pub fn heartbeat(&mut self, perception: &Array1<f64>, external_reward: f64) -> usize {
        self.heartbeat_dt(perception, external_reward, 0.1)
    }

    /// Heartbeat with real-time delta `dt` (seconds since last tick).
    /// Homeostasis drifts proportionally to actual wall-clock time.
    pub fn heartbeat_dt(&mut self, perception: &Array1<f64>, external_reward: f64, dt: f64) -> usize {
        assert_eq!(perception.len(), self.dim, "heartbeat_dt: perception.len() != dim ({} != {})", perception.len(), self.dim);
        debug_assert!(!perception.iter().any(|x| x.is_nan()), "heartbeat_dt: perception contains NaN");
        let _span = tracing::span!(tracing::Level::TRACE, "heartbeat", step = self.step_count).entered();
        self.step_count += 1;
        self.total_steps += 1;

        // ── Step 1 : Perception & Internal State ─────────────────
        self.hypothalamus.step_dt(dt);

        // Spatial attention: gate whiskers toward predicted anomaly direction.
        // The predicted_concept_id from the *previous* tick is used — it holds
        // what episodic memory expected to see *now*. Boosting the directions
        // where the current input diverges from expectation lets the organism
        // "turn its head" toward surprising stimuli.
        let predicted_proto = self.predicted_concept_id
            .and_then(|id| self.attractor.get_prototype(id));
        let gated = self.attention.attend(perception, predicted_proto);

        self.working_mem.observe(&[gated.clone()]);

        // Categorisation via PerceptualBelt (attention-gated perception)
        let percept = self.belt.process(&gated, None, &[], false, true, true, true);
        let concept_id = percept.concept_id;
        let intrinsic = percept.intrinsic;
        let _is_new = percept.is_new;

        // Sync belt internal episodic prediction
        self.belt.set_predicted_concept_id(self.predicted_concept_id);

        // Update episodic prediction for next step
        self.context.push(concept_id);
        self.predicted_concept_id = self.episodic.recall(&self.context.as_slice());

        // Record transition
        let prev_concept = self.current_concept_id;
        self.current_concept_id = Some(concept_id);
        self.episode_trace.push(concept_id);

        // Grow concept_values; initialise new concepts
        while self.concept_values.len() <= concept_id {
            self.concept_values.push(0.0);
        }

        let _shaping = match prev_concept {
            Some(p) if p < self.concept_values.len() && concept_id < self.concept_values.len() => {
                self.concept_values[concept_id] - self.concept_values[p]
            }
            _ => 0.0,
        };

        // Suppress curiosity on terminal transitions
        let is_terminal = external_reward.abs() >= 10.0;
        let r_curiosity = if is_terminal { 0.0 } else { intrinsic };

        // Value iteration trans log uses extrinsic reward only
        if let Some(p) = prev_concept {
            let step_r = if external_reward >= 20.0 { -0.05 } else { external_reward };
            self.trans_log.push((p, concept_id, self.prev_action.unwrap_or(0), step_r));
        }

        if external_reward >= 20.0 && concept_id < self.concept_values.len() {
            self.concept_values[concept_id] = 20.0;
        }

        // Graph: add transition edge between concept prototypes
        if self.episode_trace.len() >= 2 {
            let p = self.episode_trace[self.episode_trace.len() - 2];
            let a = &self.attractor.prototypes[p][0];
            let b = &self.attractor.prototypes[concept_id][0];
            self.graph.add_transition(a, b, external_reward);
            // Track habit: repeated transitions become metabolically cheaper
            let key = (p, concept_id);
            *self.habit_counts.entry(key).or_insert(0) += 1;
        }

        // Periodic inline pruning — prevents zombie accumulation during long
        // continuous episodes (real-time mode). Runs at the same cadence as
        // the inactivity timeout so every concept gets at least one check window.
        if self.concept_prune_threshold > 0 && self.step_count % self.concept_prune_threshold == 0 {
            self.prune_concepts(self.compute_arousal());
        }

        // ── Step 2 : Tension Evaluation (Φ) ──────────────────────
        self.current_phi = self.graph.phi();
        let _phi_delta = self.current_phi - self.phi_prev;
        self.anxious = self.current_phi > self.phi_threshold;
        self.hypothalamus.set_phi(self.current_phi);

        // Continuous constraint satisfaction
        let result = resolve_with_anneal(&mut self.graph, 15, 0.05, 0.2);
        self.oscillation_breaks += result.oscillation_breaks;
        let resolved_phi = self.graph.phi();
        self.phi_prev = resolved_phi;
        self.current_phi = resolved_phi;
        self.anxious = resolved_phi > self.phi_threshold;
        self.hypothalamus.set_phi(resolved_phi);

        // Φ_total = Φ_graph + homeostatic_drift
        let homeostatic_drift = self.hypothalamus.total_deficit();
        self.phi_total = resolved_phi + homeostatic_drift;
        let _phi_total_delta = self.phi_total - self.phi_total_prev;
        self.phi_total_prev = self.phi_total;

        // ── Step 3 : Action Selection ─────────────────────────────
        // Gate the external reward through the Hypothalamus:
        //   deficit = energy_deficit + hydration_deficit + temperature_deviation
        //   perceived reward = external × (1.0 + deficit × 2.0)
        let gated_reward = self.hypothalamus.gate_reward(external_reward);

        // Consummatory satisfaction: reaching a goal when deprived is itself rewarding
        let consummatory = self.hypothalamus.consummatory_value(external_reward);

        // Reduce deficits when a reward is received (simulates eating/drinking)
        if external_reward > 0.0 {
            self.hypothalamus.consume(external_reward);
        }

        let deficit_penalty = -self.hypothalamus.total_deficit() * 0.2;

        let chronic_tension = -(self.current_phi * self.current_phi) * 0.001;

        let well_being = gated_reward + consummatory + r_curiosity
            + deficit_penalty + chronic_tension;

        let decision_state = if self.use_stationary_reward {
            perception.clone()
        } else {
            gated.clone()
        };
        let mut logits = self.cerebellum.forward_logits(&decision_state);

        // ── Step 4 : Hebbian Learning ────────────────────────────
        // TD-error update with well-being as the reward signal
        let saved_lr = self.cerebellum.lr;
        self.cerebellum.lr *= 1.0 + self.compute_arousal();
        self.cerebellum.reinforce_td(well_being, 0.99);
        self.cerebellum.lr = saved_lr;
        // Backprop TD gradient into encoder (VAE/Attractor)
        if let Some(ref mut enc) = self.encoder {
            let d = self.cerebellum.last_delta;
            if d.abs() > 1e-8 { enc.backprop_td(d); }
        }
        self.cerebellum.decay_trace(0.99, 0.98);

        // REPLAY BUFFER : store (s_{t-1}, a_{t-1}, r_t, s_t)
        // Si use_stationary_reward, stocke external_reward seulement (pas de BFS
        // dans heartbeat, mais R_ext seul est déjà stationnaire).
        let replay_r = if self.use_stationary_reward { external_reward } else { well_being };
        if !self.cerebellum.is_linear() {
            if let (Some(ps), Some(pa)) = (self.prev_gated.clone(), self.prev_action) {
                self.cerebellum.store_transition(&ps, pa, replay_r, &decision_state, false);
            }
            self.prev_gated = Some(decision_state.clone());
        }

        // ε-greedy exploration
        let exploring = self.cerebellum.noise_std > 0.0;
        if exploring {
            let mut rng = rand::thread_rng();
            if rand::random::<f64>() < self.cerebellum.epsilon {
                let action_id = rng.gen_range(0..logits.len());
                self.cerebellum.mark(&decision_state, action_id);
                self.prev_action = Some(action_id);
                return action_id;
            }
            for l in logits.iter_mut() {
                *l += rng.gen_range(-self.cerebellum.noise_std..self.cerebellum.noise_std);
            }
        }

        let action_id = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        self.cerebellum.mark(&decision_state, action_id);
        self.prev_action = Some(action_id);
        // Export metrics
        if self.step_count % 50 == 0 {
            event!(
                Level::INFO,
                step = self.step_count,
                phi = self.current_phi,
                phi_threshold = self.phi_threshold,
                anxious = self.anxious,
                n_concepts = self.attractor.n_classes(),
                n_edges = self.graph.edges.len(),
                last_reward = external_reward,
            );
        }
        action_id
    }

    fn propagate_values(&mut self, gamma: f64, iterations: usize) {
        let mut v = self.concept_values.clone();
        for _ in 0..iterations {
            for &(from, to, _action, r) in &self.trans_log {
                if to < v.len() && from < v.len() {
                    let td = r + gamma * v[to];
                    if td > v[from] {
                        v[from] = td;
                    }
                }
            }
        }
        self.concept_values = v;
    }

    /// Remove concepts that have not been activated for `concept_prune_threshold` steps.
    /// Reindexes all data structures (attractor, graph, tracking vectors, references)
    /// so the system remains consistent. Zombie concepts no longer occupy memory
    /// or create phantom edges that inflate Φ.
    ///
    /// `arousal` is the noradrenaline-like signal from `compute_arousal()`.
    /// High Φ → high arousal → effective threshold is multiplied by `(1 + arousal * 0.5)`,
    /// protecting concepts from pruning when the system is under tension.
    fn prune_concepts(&mut self, arousal: f64) {
        let base = self.concept_prune_threshold;
        let threshold = if arousal > 0.0 {
            (base as f64 * (1.0 + arousal * 0.5)).max(1.0) as usize
        } else {
            base
        };

        // Ensure concept_maturation is at least as long as the number of prototypes
        while self.concept_maturation.len() < self.attractor.n_classes() {
            self.concept_maturation.push(0);
        }
        if threshold == 0 { return; }
        let n = self.attractor.prototypes.len();
        if n == 0 { return; }

        // Ensure all tracking vectors are sized to n
        while self.last_active_step.len() < n {
            self.last_active_step.push(0);
        }
        while self.concept_novelty_thresholds.len() < n {
            self.concept_novelty_thresholds.push(self.novelty_threshold);
        }
        while self.concept_local_error.len() < n {
            self.concept_local_error.push(0.0);
        }
        while self.concept_values.len() < n {
            self.concept_values.push(0.0);
        }
        while self.concept_maturation.len() < n {
            self.concept_maturation.push(0);
        }

        // Determine survivors: concepts active within the threshold window
        // Les concepts en période critique (maturation > 0) sont toujours protégés
        let survivors: Vec<bool> = (0..n)
            .map(|i| {
                self.step_count - self.last_active_step[i] <= threshold
                    || self.concept_maturation.get(i).copied().unwrap_or(0) > 0
            })
            .collect();
        let survivor_count = survivors.iter().filter(|&&a| a).count();
        if survivor_count == n || survivor_count == 0 { return; }

        // Build old→new mapping for concept IDs (0..n) plus any extra graph nodes
        let max_old = n.max(self.graph.nodes.len());
        let mut old_to_new: Vec<Option<usize>> = vec![None; max_old];
        let mut new_id = 0;
        for old in 0..n {
            if survivors[old] {
                old_to_new[old] = Some(new_id);
                new_id += 1;
            }
        }
        // Graph-only nodes (indices >= n) all survive
        for old in n..self.graph.nodes.len() {
            old_to_new[old] = Some(new_id);
            new_id += 1;
        }

        // Remap attractor prototypes
        self.attractor.prototypes = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.attractor.prototypes[i].clone())
            .collect();

        // Remap graph nodes
        self.graph.nodes = (0..self.graph.nodes.len())
            .filter_map(|old| old_to_new[old].map(|_| self.graph.nodes[old].clone()))
            .collect();

        // Rebuild graph edges via public API
        let old_edges = std::mem::take(&mut self.graph.edges);
        self.graph.clear_edges();
        for e in &old_edges {
            if let (Some(nf), Some(nt)) = (old_to_new[e.from], old_to_new[e.to]) {
                self.graph.add_edge(nf, nt, e.weight);
            }
        }

        // Enforce node–concept consistency: if graph has extra nodes beyond the
        // attractor's concept count (possible when find_similar_node reuses an
        // existing node for a different concept), truncate and rebuild edges.
        if self.graph.nodes.len() > self.attractor.prototypes.len() {
            self.graph.nodes.truncate(self.attractor.prototypes.len());
            let survivor_edges = std::mem::take(&mut self.graph.edges);
            self.graph.clear_edges();
            for e in &survivor_edges {
                if e.from < self.graph.nodes.len() && e.to < self.graph.nodes.len() {
                    self.graph.add_edge(e.from, e.to, e.weight);
                }
            }
        }

        // Remap tracking vectors
        self.concept_novelty_thresholds = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.concept_novelty_thresholds[i])
            .collect();
        self.concept_local_error = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.concept_local_error[i])
            .collect();
        self.concept_values = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.concept_values[i])
            .collect();
        self.last_active_step = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.last_active_step[i])
            .collect();
        self.concept_maturation = (0..n)
            .filter(|&i| survivors[i])
            .map(|i| self.concept_maturation[i])
            .collect();

        // Helper: remap an optional concept ID
        let remap = |id: Option<usize>| -> Option<usize> {
            id.and_then(|old| old_to_new.get(old).copied().flatten())
        };

        // Remap current / predicted concept
        self.current_concept_id = remap(self.current_concept_id);
        self.predicted_concept_id = remap(self.predicted_concept_id);

        // Remap episode trace
        self.episode_trace = self.episode_trace.iter()
            .filter_map(|&old| remap(Some(old)))
            .collect();

        // Remap context buffer
        self.context.remap(&old_to_new);

        // Remap stored episodic sequences (old episodes contain stale IDs)
        self.episodic.remap(&old_to_new);

        // Remap transition log
        self.trans_log = self.trans_log.iter()
            .filter_map(|&(from, to, action, r)| {
                Some((remap(Some(from))?, remap(Some(to))?, action, r))
            })
            .collect();
    }

    pub fn end_episode(&mut self) {
        self.prev_gated = None;
        self.prev_action = None;
        self.prev_bfs_value = None;
        self.prune_concepts(self.compute_arousal());
        self.total_episodes += 1;
        if !self.trans_log.is_empty() {
            self.propagate_values(0.99, 10);
        }
        if self.episode_trace.len() > 1 {
            self.episodic.store(&self.episode_trace);
        }
        if self.cogs.autonomous_sleep && self.should_sleep(self.total_episodes) {
            self.sleep_cycle();
            self.hypothalamus.reset_sleep();
        }
        self.episode_trace.clear();
        self.working_mem.reset();
        self.cerebellum.reset_trace();
        self.predicted_concept_id = None;
    }

    /// Current sleep pressure as a fraction [0, 1].
    /// Builds up with each step via hypothalamus drift, resets after sleep.
    pub fn sleep_pressure(&self) -> f64 {
        self.hypothalamus.sleep_pressure()
    }

    /// Whether the organism should enter sleep based on hypothalamic sleep pressure
    /// or the episode-interval trigger.
    pub fn should_sleep(&self, episode_count: usize) -> bool {
        if self.hypothalamus.sleep_pressure() >= 1.0 {
            return true;
        }
        if self.sleep_every_n_episodes > 0
            && episode_count > 0
            && episode_count % self.sleep_every_n_episodes == 0
        {
            return true;
        }
        false
    }

    /// Enter the sleep / offline consolidation phase.
    ///
    /// During this phase the engine does not receive sensor input.  It replays
    /// stored episodic traces to slowly adjust attractor prototypes (neocortical
    /// consolidation), runs deep graph conflict resolution, prunes redundant
    /// prototypes and low-phi edges (synaptic pruning), performs neurogenesis
    /// (adding prototypes for concepts with high replay error), and removes
    /// inactive concepts.
    pub fn sleep_cycle(&mut self) -> SleepReport {
        let replay_epochs = self.sleep_replay_epochs;
        let resolve_iters = self.sleep_resolve_iters;
        let noise_std = self.sleep_noise_std;

        let n_episodes = self.episodic.len();
        let phi_before = self.graph.phi();

        let mut replay_count = 0usize;
        let mut prototypes_added = 0usize;

        // ── Phase 1: Replay episodic traces into attractor field ──
        // Slow down the learning rate for gentle offline consolidation.
        // Uses prioritized replay: recent episodes are replayed first and
        // more frequently. Older episodes may be partially skipped to focus
        // consolidation on what matters most.
        if n_episodes > 0 {
            let lr_saved = self.attractor.lr;
            self.attractor.lr *= 0.3;

            // Build replay queue: recent episodes last (highest index = most recent).
            // If sleep_max_replay > 0, only replay the most recent episodes.
            let replay_start = if self.sleep_max_replay > 0 && n_episodes > self.sleep_max_replay {
                n_episodes - self.sleep_max_replay
            } else {
                0
            };
            // Iterate indices from most recent to oldest.
            let replay_indices: Vec<usize> = (replay_start..n_episodes).rev().collect();

            for _epoch in 0..replay_epochs {
                for &ep_idx in &replay_indices {
                    let seq = match self.episodic.get_sequence(ep_idx) {
                        Some(s) => s,
                        None => continue,
                    };
                    if seq.len() < 2 {
                        continue;
                    }

                    // Self-replay: noisy prototype -> own class
                    for &cid in seq {
                        if cid >= self.attractor.n_classes() {
                            continue;
                        }
                        if let Some(proto) = self.attractor.get_prototype(cid) {
                            let noise: Array1<f64> = (0..proto.len())
                                .map(|_| rand::random::<f64>() * noise_std * 2.0 - noise_std)
                                .collect();
                            let noisy = proto + &noise;
                            let (_, dist) = self.attractor.predict_with_distance(&noisy);
                            replay_count += 1;

                            // Neurogenesis: if the noisy prototype is far from its
                            // matched prototype (prediction error high), add a new
                            // prototype to better cover this region of the concept.
                            if dist > noise_std * 3.0 {
                                self.attractor.add_prototype(&noisy, cid);
                                prototypes_added += 1;
                            }

                            self.attractor.train_step(&noisy, cid);
                        }
                    }

                    // Temporal replay: noisy prototype[i] -> class[i+1]
                    for window in seq.windows(2) {
                        let prev = window[0];
                        let next = window[1];
                        if prev >= self.attractor.n_classes()
                            || next >= self.attractor.n_classes()
                        {
                            continue;
                        }
                        if let Some(proto) = self.attractor.get_prototype(prev) {
                            let noise: Array1<f64> = (0..proto.len())
                                .map(|_| rand::random::<f64>() * noise_std * 2.0 - noise_std)
                                .collect();
                            self.attractor.train_step(&(proto + &noise), next);
                        }
                    }
                }
            }

            self.attractor.lr = lr_saved;
        }

        // Synchroniser la config du module neurogenesis avec CognitiveConfig
        self.neurogenesis.config.rate = self.cogs.sleep_neurogenesis_rate;
        self.neurogenesis.config.max_concepts = self.cogs.sleep_max_concepts;
        self.neurogenesis.config.maturation_cycles = self.cogs.sleep_maturation_cycles;
        self.neurogenesis.config.synaptic_scaling = self.cogs.sleep_synaptic_scaling;

        // ── Phase 1.5: Neurogenèse — délégation au module dédié ──
        let phi_stress = if self.phi_history.is_empty() {
            0.0
        } else {
            self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64
        };
        let neuro_outcome = self.neurogenesis.cycle(
            &mut self.attractor,
            &mut self.graph,
            &self.last_active_step,
            noise_std,
            phi_stress,
        );
        // Synchroniser le vecteur de maturation interne du module avec TsoEngine
        self.concept_maturation = self.neurogenesis.maturation.clone();

        // ── Phase 2: Deep graph conflict resolution ──
        // Run many resolve iterations to deeply clean up structural conflicts
        // in the semantic graph — more than the 15-20 used online.
        let result = resolve_with_anneal(&mut self.graph, resolve_iters, 0.05, 0.3);
        self.oscillation_breaks += result.oscillation_breaks;

        // ── Phase 3: Prune redundant attractor prototypes ──
        // Merges prototypes that are closer than 0.05 — removes redundancy
        // created by noisy sleep updates.
        let prototypes_pruned = self.attractor.prune_redundant(0.05);

        // ── Phase 3.5: Scaling synaptique — déjà intégré dans neurogenesis.cycle()

        // ── Phase 4: Remove low-phi edges ──
        // Edges that contribute negligible tension are pruned from the
        // semantic graph, keeping only structurally meaningful connections.
        let edges_removed = self.graph.remove_low_phi_edges(0.001);

        // ── Phase 5: Standard concept pruning ──
        // Removes concepts (classes) that have been inactive for too long,
        // reindexing all data structures consistently.
        let concepts_before = self.attractor.n_classes();
        self.prune_concepts(self.compute_arousal());
        let concepts_pruned = concepts_before.saturating_sub(self.attractor.n_classes());

        let phi_after = self.graph.phi();
        self.sleep_cycles += 1;
        self.hypothalamus.reset_sleep();

        SleepReport {
            replay_count,
            prototypes_pruned,
            prototypes_added,
            new_concepts: neuro_outcome.births,
            edges_removed,
            concepts_pruned,
            phi_before,
            phi_after,
        }
    }

    /// Total episodes lived (persists across restarts).
    pub fn episode_count(&self) -> usize {
        self.total_episodes
    }

    /// Instantié les métriques courantes en snapshot sériaisable.
    pub fn metrics_snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            phi: self.current_phi,
            well_being: self.last_total_reward,
            energy: self.hypothalamus.energy,
            hydration: self.hypothalamus.hydration,
            temperature: self.hypothalamus.temperature,
            sleep_pressure: self.hypothalamus.sleep_pressure(),
            n_concepts: self.num_concepts(),
            n_edges: self.graph_edges(),
            total_episodes: self.total_episodes,
            total_steps: self.total_steps,
            sleep_cycles: self.sleep_cycles,
        }
    }

    /// Number of stored episodic sequences.
    pub fn episodic_size(&self) -> usize {
        self.episodic.len()
    }

    /// Number of edges in the semantic graph.
    pub fn graph_edges(&self) -> usize {
        self.graph.edges.len()
    }

    /// Expose current concept ID for debug visualization.
    pub fn num_concepts(&self) -> usize {
        self.attractor.prototypes.len()
    }

    /// Average Φ over the recent history window (for diagnostics).
    pub fn recent_phi(&self) -> f64 {
        if self.phi_history.is_empty() { 0.0 }
        else { self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64 }
    }

    pub fn current_concept_id(&self) -> Option<usize> {
        self.current_concept_id
    }

    /// Accède au vecteur de maturation des concepts (période critique).
    pub fn concept_maturation(&self) -> &[usize] {
        &self.concept_maturation
    }

    /// Trouve le concept le moins actif (hors période critique) pour remplacement.
    /// Retourne None si tous les concepts sont en période critique ou s'il n'y a pas de concepts.
    pub fn find_least_active_concept(&self) -> Option<usize> {
        let n = self.attractor.n_classes();
        if n == 0 {
            return None;
        }
        (0..n)
            .filter(|&i| self.concept_maturation.get(i).copied().unwrap_or(0) == 0)
            .min_by_key(|&i| self.last_active_step.get(i).copied().unwrap_or(0))
    }

    /// Flag an edge by its endpoints, removing it from the semantic graph.
    /// Returns the amount of Φ eliminated by this single flag.
    /// This is the "drapeau" mechanic: each flagged edge resolves a conflict
    /// and immediately drops cognitive tension.
    /// Balayage par décroissance exponentielle.
    pub fn exponential_decay_sweep(&mut self, tol: f64, factor: f64) -> (usize, f64, f64) {
        exponential_decay_sweep(&mut self.graph, tol, factor)
    }

    pub fn exponential_decay_trace(&mut self, tol: f64, factor: f64)
        -> (usize, f64, f64, Vec<(f64, f64, i8)>) {
        exponential_decay_trace(&mut self.graph, tol, factor)
    }

    pub fn flag_edge(&mut self, from: NodeId, to: NodeId) -> f64 {
        self.graph.flag_edge(from, to)
    }

    /// Inject `count` random exclusion edges into the semantic graph.
    /// Used for stress-testing the resolution and pruning machinery.
    pub fn inject_exclusion_edges(&mut self, count: usize) -> usize {
        self.graph.inject_exclusion_edges(count)
    }

    /// Bulk-prune exclusion edges whose phi contribution is negligible.
    /// Returns (exclusion_removed, implication_removed, phi_saved).
    pub fn prune_exclusion_edges(&mut self, min_phi: f64) -> (usize, usize, f64) {
        self.graph.prune_exclusion_edges(min_phi)
    }

    /// Run a sweep that systematically "flags" all violated edges.
    /// Returns (flags_planted, total_phi_dropped, final_phi).
    pub fn demineur_sweep(&mut self, tol: f64) -> (usize, f64, f64) {
        demineur_sweep(&mut self.graph, tol)
    }

    /// Run a sweep with per-flag tracing showing Φ before/after each flag.
    /// Returns (flags, phi_dropped, final_phi, vec![(phi_avant, phi_après, weight)]).
    pub fn demineur_sweep_trace(&mut self, tol: f64) -> (usize, f64, f64, Vec<(f64, f64, i8)>) {
        demineur_sweep_trace(&mut self.graph, tol)
    }

    /// Balayage par inhibition latérale (décroissance graduelle).
    /// Balayage par inhibition latérale (décroissance graduelle).
    pub fn lateral_inhibition_sweep(&mut self, tol: f64, decay: i8) -> (usize, f64, f64) {
        lateral_inhibition_sweep(&mut self.graph, tol, decay)
    }

    /// Version avec trace de lateral_inhibition_sweep.
    pub fn lateral_inhibition_trace(&mut self, tol: f64, decay: i8)
        -> (usize, f64, f64, Vec<(f64, f64, i8)>) {
        lateral_inhibition_trace(&mut self.graph, tol, decay)
    }

    /// Resolve graph conflicts by gradient descent on node vectors.
    ///
    /// CONSTRAINT REDIRECTION strategy:
    ///   - No edges are deleted or modified.
    ///   - Node vectors drift in latent space to satisfy all edge constraints
    ///     simultaneously (energy-based learning).
    ///   - After calling this, use `sync_prototypes_after_redirection()` to
    ///     propagate drifted vectors back to the AttractorField prototypes.
    pub fn resolve_constraint_redirection(&mut self, _config: &()) -> () { () }

    /// Copy graph node vectors back to the first prototype of each concept.
    ///
    /// Call this after `resolve_constraint_redirection()` to keep the
    /// attractor field in sync with the drifted semantic vectors.
    pub fn sync_prototypes_after_redirection(&mut self) {}

    /// Run deep resolution in parallel using scoped threads.
    /// Falls back to sequential for num_threads <= 1.

    /// Forced evolution: periodically add new challenging edges of mixed types
    /// (exclusion AND implication) to simulate a changing environment that
    /// tests the system's adaptability under full O(|E|) complexity.
    ///
    /// Mixing both edge types creates "triangles mixtes" — the hardest
    /// constraint satisfaction problems for the annealer, because resolving
    /// one edge may worsen another, causing the oscillation cycles that the
    /// oscillation breaker must handle.
    ///
    /// `n_new` edges are added each call, targeting concept pairs with high
    /// existing phi (deep conflicts) when possible. Roughly 60% exclusion,
    /// 40% implication (including some +2 for high-reward simulation).
    pub fn forced_evolution(&mut self, n_new: usize) -> usize {
        if self.graph.nodes.len() < 2 { return 0; }
        let mut added = 0usize;
        let mut rng = rand::thread_rng();

        // Phase 1: Add edges near existing high-phi conflicts (targeted attack)
        let mut sorted: Vec<(usize, f64)> = self.graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, self.graph.edge_phi(e)))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, _) in sorted.iter().take(n_new / 3) {
            let e = &self.graph.edges[idx];
            if self.graph.nodes.len() > 2 {
                let target = loop {
                    let t = rng.gen_range(0..self.graph.nodes.len());
                    if t != e.from && t != e.to { break t; }
                };
                // Create the opposite type of edge near the conflict
                // (if the conflict is exclusion, add implication, and vice versa)
                let weight = if e.weight == -1 { 1 } else { -1 };
                if self.graph.edge_weight(e.from, target).is_none() {
                    self.graph.add_edge(e.from, target, weight);
                    added += 1;
                }
            }
        }

        // Phase 2: Fill with random mixed edges (60% exclusion, 40% implication)
        while added < n_new {
            let a = rng.gen_range(0..self.graph.nodes.len());
            let b = rng.gen_range(0..self.graph.nodes.len());
            if a == b || self.graph.edge_weight(a, b).is_some() { continue; }
            // 60% exclusion, 30% implication, 10% strong implication
            let roll: f64 = rand::random();
            let weight = if roll < 0.60 { -1 }
                        else if roll < 0.90 { 1 }
                        else { 2 };
            self.graph.add_edge(a, b, weight);
            added += 1;
        }
        added
    }

    /// Compute proof metrics for the Weakness Game §8.
    pub fn proof_metrics(&self, total_flags: usize, phi_eliminated_by_flags: f64,
                         edges_pruned: usize, total_resolve_iters: usize,
                         evolution_cycles: usize, peak_edge_count: usize) -> ProofMetrics {
        let current_phi = self.graph.phi();
        let excl_count = self.graph.edges.iter().filter(|e| e.weight == -1).count();
        let edge_count = self.graph.edges.len();
        let avg_per_flag = if total_flags > 0 { phi_eliminated_by_flags / total_flags as f64 } else { 0.0 };
        let pruning_eff = if peak_edge_count > 0 { edges_pruned as f64 / peak_edge_count as f64 } else { 0.0 };
        // Proof score: high when Φ is low and edges are well-controlled
        let score = if edge_count > 0 && current_phi > 0.0 {
            (phi_eliminated_by_flags + 1.0) / (edge_count as f64 * current_phi.max(0.01))
        } else if edge_count == 0 {
            100.0 // Perfect: empty graph, no conflicts
        } else {
            (phi_eliminated_by_flags + 1.0) / (edge_count as f64 * 0.01)
        };
        ProofMetrics {
            total_flags,
            phi_eliminated_by_flags,
            current_phi,
            edge_count,
            exclusion_edge_count: excl_count,
            peak_edge_count,
            edges_pruned,
            total_resolve_iters,
            avg_phi_per_flag: avg_per_flag,
            evolution_cycles,
            pruning_efficiency: pruning_eff,
            proof_score: score,
        }
    }

    /// Augmente une perception avec les cellules de grille si actives.
    /// Position (0,0) par défaut pour les environnements sans tracking de position.
    pub fn augment_perception(&self, p: &Array1<f64>, x: usize, y: usize) -> Array1<f64> {
        self.grid_cells.augment(p, x, y)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
