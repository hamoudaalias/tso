use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Clone)]
pub struct EligibilityTrace {
    pub tau_fast: f64,
    pub tau_slow: f64,
    pub e_fast: Array2<f64>,
    pub e_slow: Array2<f64>,
}

impl EligibilityTrace {
    pub fn new(shape: (usize, usize), tau_fast: f64, tau_slow: f64) -> Self {
        Self {
            tau_fast,
            tau_slow,
            e_fast: Array2::zeros(shape),
            e_slow: Array2::zeros(shape),
        }
    }

    pub fn update(&mut self, pre: &Array1<f64>, post: &Array1<f64>, dt: f64) {
        let a_fast = 1.0 - (-dt / self.tau_fast).exp();
        let a_slow = 1.0 - (-dt / self.tau_slow).exp();
        for i in 0..pre.len() {
            for j in 0..post.len() {
                let hebb = pre[i] * post[j];
                self.e_fast[[i, j]] += a_fast * (hebb - self.e_fast[[i, j]]);
                self.e_slow[[i, j]] += a_slow * (hebb - self.e_slow[[i, j]]);
            }
        }
    }

    pub fn get(&self, beta: f64) -> Array2<f64> {
        beta * &self.e_fast + (1.0 - beta) * &self.e_slow
    }

    pub fn decay(&mut self, dt: f64) {
        let df = (-dt / self.tau_fast).exp();
        let ds = (-dt / self.tau_slow).exp();
        self.e_fast.mapv_inplace(|v| v * df);
        self.e_slow.mapv_inplace(|v| v * ds);
    }

    pub fn reset(&mut self) {
        self.e_fast.fill(0.0);
        self.e_slow.fill(0.0);
    }
}

#[derive(Clone)]
pub struct RSTDPPlasticity {
    pub n: usize,
    pub alpha_p: f64,
    pub alpha_n: f64,
    pub inhib_factor: f64,
    pub w: Array2<f64>,
    pub z_targets: HashMap<usize, Array1<f64>>,
    pub eligibility: Array1<f64>,
    pub decay: f64,
}

impl RSTDPPlasticity {
    pub fn new(n_clusters: usize, alpha_p: f64, alpha_n: f64, inhib_factor: f64) -> Self {
        Self {
            n: n_clusters,
            alpha_p,
            alpha_n,
            inhib_factor,
            w: Array2::zeros((n_clusters, n_clusters)),
            z_targets: HashMap::new(),
            eligibility: Array1::zeros(n_clusters),
            decay: f64::exp(-1.0 / 20.0),
        }
    }

    pub fn register_target(&mut self, idx: usize, z_vec: Array1<f64>) {
        self.z_targets.insert(idx, z_vec);
    }

    pub fn consolidate(&mut self, pre_idx: usize, post_idx: usize) {
        self.consolidate_with_rates(pre_idx, post_idx, 1.0, 1.0);
    }

    pub fn consolidate_with_rates(
        &mut self,
        pre_idx: usize,
        post_idx: usize,
        rate_pre: f64,
        rate_post: f64,
    ) {
        let el = self.eligibility[pre_idx] * self.decay + rate_pre * rate_post;
        self.eligibility[pre_idx] = el;

        if let (Some(a), Some(b)) = (self.z_targets.get(&pre_idx), self.z_targets.get(&post_idx)) {
            let dot = a.dot(b);
            let na = a.dot(a).sqrt();
            let nb = b.dot(b).sqrt();
            let sim = dot / (na * nb + 1e-8);
            if sim < 0.0 {
                self.w[[pre_idx, post_idx]] -= self.inhib_factor * el;
                self.w[[pre_idx, post_idx]] = self.w[[pre_idx, post_idx]].max(0.0);
                return;
            }
        }

        self.w[[pre_idx, post_idx]] += self.alpha_p * el;
        self.w[[pre_idx, post_idx]] = self.w[[pre_idx, post_idx]].clamp(0.0, 1.5);
    }

    pub fn reward_modulate(&mut self, _phi: f64, delta_phi: f64) {
        let scale = if delta_phi < 0.0 { 1.0 } else { 0.1 };
        self.w.mapv_inplace(|v| v * (1.0 - self.alpha_n * scale));
        self.w.mapv_inplace(|v| v.clamp(0.0, 1.5));
    }

    pub fn reset(&mut self) {
        self.w.fill(0.0);
        self.eligibility.fill(0.0);
    }
}
