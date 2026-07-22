use std::collections::HashMap;
use ndarray::Array1;
use tso_engine::core::{Graph, NodeId};
use tso_engine::neurons::LIFState;
use tso_engine::attractor::AttractorField;

use crate::grid::{GridWorld, Action, StepResult, ACTIONS};
use crate::encoder::StateEncoder;

pub struct TSOAgent {
    pub lif_slow: LIFState,
    pub lif_fast: LIFState,
    pub graph: Graph,
    pub field: AttractorField,
    pub encoder: StateEncoder,
    pub state_node: Option<NodeId>,
    pub episode_nodes: Vec<NodeId>,
    pub dim: usize,
    pub epsilon: f64,
    pub lr: f64,
    pub gamma: f64,
    step_count: usize,
    q_table: HashMap<(usize, usize, usize), f64>,
    prev_pos: Option<(usize, usize)>,
    prev_action: Option<usize>,
}

const N_CLASSES_BASE: usize = 2;

impl TSOAgent {
    pub fn new(dim: usize) -> Self {
        let slow = LIFState::new(dim, 0.9);
        let fast = LIFState::new(dim, 0.5);
        let graph = Graph::new();
        let field = AttractorField::new(dim, N_CLASSES_BASE, 3, 0.08);
        let encoder = StateEncoder::new();

        TSOAgent {
            lif_slow: slow,
            lif_fast: fast,
            graph,
            field,
            encoder,
            state_node: None,
            episode_nodes: Vec::new(),
            dim,
            epsilon: 0.5,
            lr: 0.15,
            gamma: 0.9,
            step_count: 0,
            q_table: HashMap::new(),
            prev_pos: None,
            prev_action: None,
        }
    }

    pub fn reset(&mut self) {
        self.lif_slow = LIFState::new(self.dim, 0.9);
        self.lif_fast = LIFState::new(self.dim, 0.5);
        self.state_node = None;
        self.episode_nodes.clear();
        self.step_count = 0;
        self.prev_pos = None;
        self.prev_action = None;
    }

    pub fn act(&mut self, gw: &GridWorld) -> Action {
        let pos = (gw.agent_x, gw.agent_y);
        self.prev_pos = Some(pos);
        self.step_count += 1;

        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.r#gen::<f64>() < self.epsilon {
            let ai = rng.gen_range(0..ACTIONS.len());
            self.prev_action = Some(ai);
            return ACTIONS[ai];
        }

        let mut best_ai = 0;
        let mut best_q = f64::NEG_INFINITY;
        for (ai, _) in ACTIONS.iter().enumerate() {
            let q = self.q_table.get(&(pos.0, pos.1, ai)).copied().unwrap_or(0.0);
            if q > best_q {
                best_q = q;
                best_ai = ai;
            }
        }
        self.prev_action = Some(best_ai);
        ACTIONS[best_ai]
    }

    pub fn q_update(&mut self, result: StepResult, next_pos: (usize, usize)) {
        let (px, py) = match self.prev_pos {
            Some(p) => p,
            None => return,
        };
        let ai = match self.prev_action {
            Some(a) => a,
            None => return,
        };

        let mut max_next_q = 0.0;
        if result == StepResult::Move {
            for (nai, _) in ACTIONS.iter().enumerate() {
                let nq = self.q_table.get(&(next_pos.0, next_pos.1, nai)).copied().unwrap_or(0.0);
                if nq > max_next_q { max_next_q = nq; }
            }
        }

        let reward = match result {
            StepResult::Collision => -10.0,
            StepResult::Goal => 50.0,
            StepResult::Move => -0.1,
        };

        let old_q = self.q_table.get(&(px, py, ai)).copied().unwrap_or(0.0);
        let new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q);
        self.q_table.insert((px, py, ai), new_q);
    }

    pub fn learn(&mut self, state: &Array1<f64>, action: Action, result: StepResult, next_state: &Array1<f64>) {
        match result {
            StepResult::Collision => {
                let dc = if self.field.n_classes() > 1 { 1 } else { self.field.add_class(state) };
                if self.field.n_classes() <= dc {
                    self.field.add_class(state);
                } else {
                    self.field.add_prototype(state, 1);
                }
            }
            StepResult::Goal => {
                while self.field.n_classes() <= 0 {
                    self.field.add_class(state);
                }
                self.field.add_prototype(state, 0);
            }
            StepResult::Move => {}
        }

        if let Some(&prev_node) = self.state_node.as_ref() {
            let next_node = self.graph.nodes.len();
            self.graph.add_node(next_state.clone());
            self.episode_nodes.push(next_node);

            let weight = match result {
                StepResult::Move => 1,
                StepResult::Goal => 2,
                StepResult::Collision => -1,
            };
            self.graph.add_edge(prev_node, next_node, weight);
        }

        let pos = (0, 0); // placeholder — the actual position is tracked elsewhere
        self.q_update(result, pos);
    }
}
