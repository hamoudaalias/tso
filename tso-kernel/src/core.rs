use crate::friction::FrictionCalculator;
use crate::neurons::LIFCluster;
use crate::plasticity::RSTDPPlasticity;
use ndarray::Array1;
use std::collections::{HashMap, VecDeque};

#[derive(Clone)]
pub struct TSOCore {
    pub d: usize,
    pub max_clusters: usize,
    pub gamma: f64,
    pub epsilon: f64,
    pub base_theta_c: f64,
    pub inertia_threshold: f64,
    pub clusters: Vec<LIFCluster>,
    pub labels: HashMap<String, usize>,
    pub edges: Vec<(usize, usize, f64, f64)>,
    pub friction: FrictionCalculator,
    pub plasticity: RSTDPPlasticity,
    pub phi_history: Vec<f64>,
    pub time: f64,
    pub activity_history: VecDeque<f64>,
    pub slow_trace_history: VecDeque<f64>,
    pub dynamic_theta_c: f64,
    pub history_size: usize,
}

impl TSOCore {
    pub fn new(
        max_clusters: usize,
        d: usize,
        gamma: f64,
        epsilon: f64,
        history_size: usize,
        base_theta_c: f64,
        inertia_threshold: f64,
    ) -> Self {
        Self {
            d,
            max_clusters,
            gamma,
            epsilon,
            base_theta_c,
            inertia_threshold,
            clusters: Vec::new(),
            labels: HashMap::new(),
            edges: Vec::new(),
            friction: FrictionCalculator::new(gamma, epsilon),
            plasticity: RSTDPPlasticity::new(max_clusters, 0.05, 0.02, 0.05),
            phi_history: Vec::new(),
            time: 0.0,
            activity_history: VecDeque::with_capacity(history_size),
            slow_trace_history: VecDeque::with_capacity(history_size),
            dynamic_theta_c: base_theta_c,
            history_size,
        }
    }

    pub fn add_cluster(&mut self, label: &str) -> isize {
        if self.clusters.len() >= self.max_clusters {
            return -1;
        }
        let idx = self.clusters.len();
        self.clusters.push(LIFCluster::new(self.d));
        self.labels.insert(label.to_string(), idx);
        idx as isize
    }

    pub fn add_edge(&mut self, i: usize, j: usize, w: f64, strength: f64) {
        self.edges.push((i, j, w, strength));
    }

    pub fn update_homeostasis(&mut self, current_activity: f64, current_slow_trace: f64) {
        self.activity_history.push_back(current_activity);
        self.slow_trace_history.push_back(current_slow_trace);

        if self.activity_history.len() > self.history_size {
            self.activity_history.pop_front();
            self.slow_trace_history.pop_front();
        }

        if self.activity_history.len() > 10 {
            let mean: f64 = self.activity_history.iter().sum::<f64>() / self.activity_history.len() as f64;
            let variance: f64 = self
                .activity_history
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / self.activity_history.len() as f64;
            let std_act = variance.sqrt();
            self.dynamic_theta_c = self.base_theta_c + std_act * 0.5;
        }
    }

    pub fn check_inertia_gate(&self) -> bool {
        if self.slow_trace_history.len() < 10 {
            return false;
        }
        let mean: f64 =
            self.slow_trace_history.iter().sum::<f64>() / self.slow_trace_history.len() as f64;
        let variance: f64 = self
            .slow_trace_history
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / self.slow_trace_history.len() as f64;
        variance < self.inertia_threshold
    }

    pub fn should_trigger_expansion(&self, current_phi: f64, min_implication: f64) -> bool {
        if current_phi < self.dynamic_theta_c {
            return false;
        }
        if min_implication < self.gamma {
            return false;
        }
        if !self.check_inertia_gate() {
            return false;
        }
        true
    }

    pub fn step(&mut self, i_ext: &Array1<f64>, dt: f64) -> (f64, Array1<f64>, u32) {
        let n = self.clusters.len();
        let mut rates = Array1::zeros(n);
        let mut total_spikes: u32 = 0;

        for (ci, c) in self.clusters.iter_mut().enumerate() {
            let val = if ci < i_ext.len() { i_ext[ci] } else { 0.0 };
            let s = c.step(&Array1::from_elem(c.n, val), dt);
            rates[ci] = c.rate;
            total_spikes += (s * c.n as f64) as u32;
        }

        let phi = self.friction.compute_phi(&rates, &self.edges);
        self.phi_history.push(phi);
        self.time += dt;

        if self.phi_history.len() > 1 {
            let dphi = phi - self.phi_history[self.phi_history.len() - 2];
            self.plasticity.reward_modulate(phi, dphi);
        }

        let mean_activity = rates.mean().unwrap_or(0.0);
        let slow_trace: f64 = self.clusters.iter().map(|c| c.rate).sum::<f64>()
            / self.clusters.len() as f64;
        self.update_homeostasis(mean_activity, slow_trace);

        (phi, rates, total_spikes)
    }

    pub fn reset(&mut self) {
        for c in &mut self.clusters {
            c.reset();
        }
        self.phi_history.clear();
        self.activity_history.clear();
        self.slow_trace_history.clear();
        self.dynamic_theta_c = self.base_theta_c;
        self.time = 0.0;
    }

    pub fn phi_gradient(&self, window: usize) -> f64 {
        if self.phi_history.len() < window {
            return 0.0;
        }
        (self.phi_history[self.phi_history.len() - 1]
            - self.phi_history[self.phi_history.len().saturating_sub(window)])
            / window as f64
    }
}
