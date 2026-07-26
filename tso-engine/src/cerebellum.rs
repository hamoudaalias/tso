use ndarray::Array1;
use rand::Rng;

pub struct Cerebellum {
    pub lr: f64,
    pub noise_std: f64,
    pub epsilon: f64,
    dim: usize,
    n_actions: usize,
    hidden_dim: usize,
    is_linear: bool,

    /// Linear weights [dim × n_actions] (used when is_linear)
    w_lin: Vec<Vec<f64>>,
    e_lin: Vec<Vec<f64>>,

    /// MLP hidden weights [hidden_dim × dim]
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    /// MLP output weights [n_actions × hidden_dim]
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Cached hidden activations [hidden_dim]
    h: Vec<f64>,
    /// MLP eligibility traces
    e1: Vec<Vec<f64>>,
    e2: Vec<Vec<f64>>,
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_deriv(y: f64) -> f64 {
    1.0 - y * y
}

fn l2_norm_col(w: &[Vec<f64>], col: usize) -> f64 {
    let mut s = 0.0;
    for r in w {
        s += r[col] * r[col];
    }
    s.sqrt()
}

fn normalize_col(w: &mut [Vec<f64>], col: usize) {
    let norm = l2_norm_col(w, col);
    if norm > 1.0 {
        for r in w.iter_mut() {
            r[col] /= norm;
        }
    }
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize(v: &mut [f64]) {
    let norm = l2_norm(v);
    if norm > 1.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl Cerebellum {
    /// hidden_dim = 0 → linear (backward compatible)
    pub fn new(dim: usize, n_actions: usize, lr: f64, noise_std: f64, epsilon: f64, hidden_dim: usize) -> Self {
        let is_linear = hidden_dim == 0;
        let hd = if is_linear { 0 } else { hidden_dim };
        let mut rng = rand::thread_rng();

        let w_lin = if is_linear {
            (0..dim)
                .map(|_| (0..n_actions).map(|_| rng.gen_range(-0.01..0.01)).collect())
                .collect()
        } else {
            vec![]
        };
        let e_lin = if is_linear {
            (0..dim).map(|_| vec![0.0; n_actions]).collect()
        } else {
            vec![]
        };

        let w1 = if is_linear {
            vec![]
        } else {
            (0..hd)
                .map(|_| (0..dim).map(|_| rng.gen_range(-0.01..0.01)).collect())
                .collect()
        };
        let b1 = if is_linear { vec![] } else { vec![0.0; hd] };
        let w2 = if is_linear {
            vec![]
        } else {
            (0..n_actions)
                .map(|_| (0..hd).map(|_| rng.gen_range(-0.01..0.01)).collect())
                .collect()
        };
        let b2 = if is_linear { vec![] } else { vec![0.0; n_actions] };
        let h = if is_linear { vec![] } else { vec![0.0; hd] };
        let e1 = if is_linear {
            vec![]
        } else {
            (0..hd).map(|_| vec![0.0; dim]).collect()
        };
        let e2 = if is_linear {
            vec![]
        } else {
            (0..n_actions).map(|_| vec![0.0; hd]).collect()
        };

        Cerebellum {
            lr, noise_std, epsilon, dim, n_actions, hidden_dim: hd, is_linear,
            w_lin, e_lin, w1, b1, w2, b2, h, e1, e2,
        }
    }

    pub fn forward(&mut self, concept: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();
        let exploring = self.noise_std > 0.0;
        if exploring && rand::random::<f64>() < self.epsilon {
            return rng.gen_range(0..self.n_actions);
        }
        let add_noise = exploring;

        if self.is_linear {
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                for i in 0..self.dim {
                    logits[a] += concept[i] * self.w_lin[i][a];
                }
                if add_noise {
                    logits[a] += rng.gen_range(-self.noise_std..self.noise_std);
                }
            }
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap()
        } else {
            // MLP forward: h = tanh(W1 @ concept + b1)
            let mut hloc = vec![0.0; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut s = self.b1[j];
                for i in 0..self.dim {
                    s += self.w1[j][i] * concept[i];
                }
                hloc[j] = tanh(s);
            }
            self.h = hloc;
            // output = W2 @ h + b2
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                logits[a] = self.b2[a];
                for j in 0..self.hidden_dim {
                    logits[a] += self.w2[a][j] * self.h[j];
                }
                if add_noise {
                    logits[a] += rng.gen_range(-self.noise_std..self.noise_std);
                }
            }
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap()
        }
    }

    /// Returns copy of hidden activations (for use by mark/reinforce after forward)
    pub fn forward_with_hidden(&mut self, concept: &Array1<f64>) -> (usize, Vec<f64>) {
        let action = self.forward(concept);
        let hloc = self.h.clone();
        (action, hloc)
    }

    /// Retourne les logits bruts (Q-valeurs) sans argmax.
    /// Utile pour le blending avec un modèle du monde.
    pub fn forward_logits(&mut self, concept: &Array1<f64>) -> Vec<f64> {
        if self.is_linear {
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                for i in 0..self.dim {
                    logits[a] += concept[i] * self.w_lin[i][a];
                }
            }
            logits
        } else {
            let mut hloc = vec![0.0; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut s = self.b1[j];
                for i in 0..self.dim {
                    s += self.w1[j][i] * concept[i];
                }
                hloc[j] = tanh(s);
            }
            self.h = hloc;
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                logits[a] = self.b2[a];
                for j in 0..self.hidden_dim {
                    logits[a] += self.w2[a][j] * self.h[j];
                }
            }
            logits
        }
    }

    pub fn learn(&mut self, concept: &Array1<f64>, action: usize, reward: f64) {
        if reward.abs() < 1e-6 { return; }
        let step = self.lr * reward.abs();

        if self.is_linear {
            if reward > 0.0 {
                for i in 0..self.dim { self.w_lin[i][action] += step * concept[i]; }
            } else {
                for i in 0..self.dim { self.w_lin[i][action] -= step * concept[i]; }
            }
            normalize_col(&mut self.w_lin, action);
        } else {
            let sign = if reward > 0.0 { 1.0 } else { -1.0 };
            // Update output layer using cached hidden activations
            for j in 0..self.hidden_dim {
                self.w2[action][j] += sign * step * self.h[j];
            }
            self.b2[action] += sign * step;
            normalize(&mut self.w2[action]);
            // Update hidden layer via backpropagated Hebbian
            for j in 0..self.hidden_dim {
                let delta_h = sign * step * tanh_deriv(self.h[j]);
                for i in 0..self.dim {
                    self.w1[j][i] += delta_h * concept[i];
                }
                self.b1[j] += delta_h;
                normalize(&mut self.w1[j]);
            }
        }
    }

    // --- Eligibility trace methods ---

    pub fn mark(&mut self, concept: &Array1<f64>, action: usize) {
        if self.is_linear {
            for i in 0..self.dim { self.e_lin[i][action] += concept[i]; }
        } else {
            // Mark output trace using cached hidden activations
            for j in 0..self.hidden_dim {
                self.e2[action][j] += self.h[j];
            }
            // Mark hidden trace: e1[j][i] += concept[i]
            for j in 0..self.hidden_dim {
                for i in 0..self.dim {
                    self.e1[j][i] += concept[i];
                }
            }
        }
    }

    pub fn decay_trace(&mut self, gamma: f64, lambda: f64) {
        let decay = gamma * lambda;
        if self.is_linear {
            for i in 0..self.dim {
                for a in 0..self.n_actions { self.e_lin[i][a] *= decay; }
            }
        } else {
            for j in 0..self.hidden_dim {
                for i in 0..self.dim { self.e1[j][i] *= decay; }
            }
            for a in 0..self.n_actions {
                for j in 0..self.hidden_dim { self.e2[a][j] *= decay; }
            }
        }
    }

    pub fn reinforce(&mut self, reward: f64) {
        if reward.abs() < 1e-6 { return; }
        let step = self.lr * reward.abs();
        let sign = if reward > 0.0 { 1.0 } else { -1.0 };

        if self.is_linear {
            for i in 0..self.dim {
                for a in 0..self.n_actions { self.w_lin[i][a] += sign * step * self.e_lin[i][a]; }
            }
            for a in 0..self.n_actions { normalize_col(&mut self.w_lin, a); }
        } else {
            // Update output layer via e2
            for a in 0..self.n_actions {
                for j in 0..self.hidden_dim { self.w2[a][j] += sign * step * self.e2[a][j]; }
                normalize(&mut self.w2[a]);
            }
            // Update hidden layer via e1
            for j in 0..self.hidden_dim {
                for i in 0..self.dim { self.w1[j][i] += sign * step * self.e1[j][i]; }
                normalize(&mut self.w1[j]);
            }
        }
    }

    pub fn reset_trace(&mut self) {
        if self.is_linear {
            for i in 0..self.dim {
                for a in 0..self.n_actions { self.e_lin[i][a] = 0.0; }
            }
        } else {
            for j in 0..self.hidden_dim {
                for i in 0..self.dim { self.e1[j][i] = 0.0; }
            }
            for a in 0..self.n_actions {
                for j in 0..self.hidden_dim { self.e2[a][j] = 0.0; }
            }
        }
    }

    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        if self.is_linear {
            for i in 0..self.dim {
                for a in 0..self.n_actions { self.w_lin[i][a] = rng.gen_range(-0.01..0.01); }
            }
        } else {
            for j in 0..self.hidden_dim {
                for i in 0..self.dim { self.w1[j][i] = rng.gen_range(-0.01..0.01); }
            }
            for a in 0..self.n_actions {
                for j in 0..self.hidden_dim { self.w2[a][j] = rng.gen_range(-0.01..0.01); }
            }
        }
        self.reset_trace();
    }

    // --- Inspection helpers ---
    pub fn get_lin_weight(&self, i: usize, a: usize) -> f64 {
        if self.is_linear && i < self.w_lin.len() && a < self.w_lin[0].len() {
            self.w_lin[i][a]
        } else {
            0.0
        }
    }

    pub fn get_hidden_weight(&self, j: usize, i: usize) -> f64 {
        if !self.is_linear && j < self.w1.len() && i < self.w1[0].len() {
            self.w1[j][i]
        } else {
            0.0
        }
    }

    pub fn get_out_weight(&self, a: usize, j: usize) -> f64 {
        if !self.is_linear && a < self.w2.len() && j < self.w2[0].len() {
            self.w2[a][j]
        } else {
            0.0
        }
    }
}
