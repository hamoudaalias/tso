use ndarray::Array1;
use rand::Rng;

pub struct Cerebellum {
    /// Poids synaptiques [concept_dim × n_actions]
    pub w: Vec<Vec<f64>>,
    pub lr: f64,
    pub noise_std: f64,
    dim: usize,
    n_actions: usize,
}

impl Cerebellum {
    pub fn new(dim: usize, n_actions: usize, lr: f64, noise_std: f64) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..dim)
            .map(|_| (0..n_actions).map(|_| rng.gen_range(-0.01..0.01)).collect())
            .collect();
        Cerebellum { w, lr, noise_std, dim, n_actions }
    }

    pub fn forward(&self, concept: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();
        let mut logits = vec![0.0; self.n_actions];
        for a in 0..self.n_actions {
            for i in 0..self.dim {
                logits[a] += concept[i] * self.w[i][a];
            }
            logits[a] += rng.gen_range(-self.noise_std..self.noise_std);
        }
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn learn(&mut self, concept: &Array1<f64>, action: usize, reward: f64) {
        if reward.abs() < 1e-6 {
            return;
        }
        let step = self.lr * reward.abs();
        if reward > 0.0 {
            for i in 0..self.dim {
                self.w[i][action] += step * concept[i];
            }
        } else {
            for i in 0..self.dim {
                self.w[i][action] -= step * concept[i];
            }
        }
        // Normalisation des colonnes pour stabilité
        let mut norm = 0.0;
        for i in 0..self.dim {
            norm += self.w[i][action] * self.w[i][action];
        }
        norm = norm.sqrt();
        if norm > 1.0 {
            for i in 0..self.dim {
                self.w[i][action] /= norm;
            }
        }
    }

    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.dim {
            for a in 0..self.n_actions {
                self.w[i][a] = rng.gen_range(-0.01..0.01);
            }
        }
    }
}
