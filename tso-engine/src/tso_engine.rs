use ndarray::Array1;
use rand::Rng;
use crate::attractor::AttractorField;
use crate::episodic::{EpisodicMemory, ContextBuffer};
use crate::cerebellum::Cerebellum;
use crate::core::{Graph, resolve};
use crate::working_memory::WorkingMemory;
use crate::action::ActionMotor;

pub struct TsoEngine {
    pub working_mem: WorkingMemory,
    pub attractor: AttractorField,
    pub episodic: EpisodicMemory,
    pub context: ContextBuffer,
    pub graph: Graph,
    pub cerebellum: Cerebellum,
    pub motor: ActionMotor,

    dim: usize,
    current_concept_id: Option<usize>,
    episode_trace: Vec<usize>,
    step_count: usize,

    /// Cognitive map : concept_id → V(s) value iteration
    concept_values: Vec<f64>,
    /// Log des transitions (from, to, reward) pour la value iteration
    pub trans_log: Vec<(usize, usize, f64)>,
}

impl TsoEngine {
    pub fn new(dim: usize, n_actions: usize) -> Self {
        TsoEngine {
            working_mem: WorkingMemory::new(dim, 0.95, 0.5),
            attractor: AttractorField::new(dim, 8, 3, 0.01),
            episodic: EpisodicMemory::new(50),
            context: ContextBuffer::new(10),
            graph: Graph::with_params(0.7, 0.1),
            cerebellum: Cerebellum::new(dim, n_actions, 0.10, 0.1, 0.50, 0),
            motor: ActionMotor::new(0.6),
            dim,
            current_concept_id: None,
            episode_trace: Vec::new(),
            step_count: 0,
            concept_values: Vec::new(),
            trans_log: Vec::new(),
        }
    }

    pub fn step(&mut self, perception: &Array1<f64>, reward: f64, bfs_value: Option<f64>, bfs_bias: &[f64]) -> usize {
        self.step_count += 1;

        // 1. PERCEPTION & WORKING MEMORY
        self.working_mem.observe(&[perception.clone()]);

        // 2. CATEGORIZATION (on raw perception, not LIF-smoothed)
        let (concept_id, dist) = self.attractor.predict_with_distance(perception);
        let is_new = dist > 0.15;
        let concept_id = if is_new { self.attractor.add_class(perception) } else { concept_id };

        self.attractor.train_step(perception, concept_id);

        // 3. RECORD TRANSITION + SHAPING REWARD
        let prev_concept = self.current_concept_id;
        self.current_concept_id = Some(concept_id);
        self.episode_trace.push(concept_id);
        self.context.push(concept_id);

        // Grow concept_values; initialize new concepts with BFS value
        while self.concept_values.len() <= concept_id {
            self.concept_values.push(0.0);
        }
        if is_new {
            if let Some(bv) = bfs_value {
                self.concept_values[concept_id] = bv;
            }
        }

        let intrinsic = 0.0;
        let shaping = match prev_concept {
            Some(p) if p < self.concept_values.len() && concept_id < self.concept_values.len() => {
                self.concept_values[concept_id] - self.concept_values[p]
            }
            _ => 0.0,
        };
        let total_reward = reward + intrinsic + shaping;

        // Store step_reward for value iteration (no intrinsic — évite la propagation
        // artificielle de valeur dans la carte cognitive).
        if let Some(p) = prev_concept {
            let step_r = if reward >= 20.0 { -0.05 } else { reward };
            self.trans_log.push((p, concept_id, step_r));
        }

        // Set goal value when found
        if reward >= 20.0 && concept_id < self.concept_values.len() {
            self.concept_values[concept_id] = 20.0;
        }

        // 4. GRAPH
        if self.episode_trace.len() >= 2 {
            let p = self.episode_trace[self.episode_trace.len() - 2];
            let a = &self.attractor.prototypes[p][0];
            let b = &self.attractor.prototypes[concept_id][0];
            self.graph.add_transition(a, b, reward);
        }
        if self.step_count % 50 == 0 { let _ = resolve(&mut self.graph, 10, 0.05); }

        // 5. REINFORCE (with shaping included)
        self.cerebellum.reinforce(total_reward);
        self.cerebellum.decay_trace(0.99, 0.98);

        // 6. DECISION STATE (raw perception — pas de LIF pour préserver le signal BFS)
        let decision_state = perception.clone();

        // 7. ACTION (logits + BFS bias + ε-greedy + noise)
        let mut logits = self.cerebellum.forward_logits(&decision_state);
        let exploring = self.cerebellum.noise_std > 0.0;
        if exploring {
            let mut rng = rand::thread_rng();
            if rand::random::<f64>() < self.cerebellum.epsilon {
                let action_id = rng.gen_range(0..logits.len());
                // Need hidden for mark() even with random action
                self.cerebellum.forward_with_hidden(&decision_state);
                self.cerebellum.mark(&decision_state, action_id);
                return action_id;
            }
            for l in logits.iter_mut() {
                *l += rng.gen_range(-self.cerebellum.noise_std..self.cerebellum.noise_std);
            }
        }
        // Add BFS bias (amplified by 0.5 to push exploration toward goal)
        for (l, b) in logits.iter_mut().zip(bfs_bias.iter()) {
            *l += b * 0.5;
        }
        let action_id = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        self.cerebellum.forward_with_hidden(&decision_state);
        self.cerebellum.mark(&decision_state, action_id);
        action_id
    }

    fn propagate_values(&mut self, gamma: f64, iterations: usize) {
        let mut v = self.concept_values.clone();
        for _ in 0..iterations {
            for &(from, to, r) in &self.trans_log {
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

    pub fn end_episode(&mut self) {
        // Background planning : propager la valeur du but dans la carte cognitive
        if !self.trans_log.is_empty() {
            self.propagate_values(0.99, 10);
        }
        if self.episode_trace.len() > 1 {
            self.episodic.store(&self.episode_trace);
        }
        // ε-decay : 0.50 → 0.05 en ~460 épisodes
        self.cerebellum.epsilon *= 0.995;
        self.cerebellum.epsilon = self.cerebellum.epsilon.max(0.05);
        self.episode_trace.clear();
        self.working_mem.reset();
        self.cerebellum.reset_trace();
    }
}
