use crate::plasticity::RstdpPlasticity;
use ndarray::Array1;
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::replay_buffer::ReplayBuffer;

#[derive(Serialize, Deserialize)]
pub struct Cerebellum {
    pub lr: f64,
    pub noise_std: f64,
    pub epsilon: f64,
    dim: usize,
    pub n_actions: usize,
    hidden_dim: usize,
    is_linear: bool,

    /// Linear weights [dim × n_actions] (used when is_linear)
    w_lin: Vec<Vec<f64>>,
    e_lin: Vec<Vec<f64>>,

    /// MLP hidden weights [hidden_dim × dim]
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    /// MLP output weights (actor) [n_actions × hidden_dim]
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    /// Cached hidden activations
    h: Vec<f64>,
    /// MLP eligibility traces
    e1: Vec<Vec<f64>>,
    e2: Vec<Vec<f64>>,

    /// Critic weights [hidden_dim]  — V(h) = w_v · h + b_v
    w_v: Vec<f64>,
    b_v: f64,
    /// Critic learning rate (defaults to 1.0 × actor lr)
    lr_critic: f64,
    /// Cached V(h_t) from the last mark() — used for TD-error computation
    v_prev: f64,
    pub rstdp: Option<RstdpPlasticity>,

    /// Experience replay buffer — stocke les transitions pour
    /// un apprentissage TD stable sans bruit d'exploration.
    pub replay: ReplayBuffer,
    /// Taux d'apprentissage pour le replay TD.
    pub replay_lr: f64,
    /// Si true, désactive le TD en ligne (reinforce_td).
    /// L'apprentissage se fait uniquement via replay_train().
    pub replay_only: bool,
    /// Clip le |δ| utilisé dans step_a = lr * min(|δ|, delta_clip).
    /// 0.0 = pas de clip (comportement original).
    pub delta_clip: f64,
}

fn tanh(x: f64) -> f64 { x.tanh() }
fn tanh_deriv(y: f64) -> f64 { 1.0 - y * y }

fn l2_norm_col(w: &[Vec<f64>], col: usize) -> f64 {
    let mut s = 0.0;
    for r in w { s += r[col] * r[col]; }
    s.sqrt()
}

fn normalize_col(w: &mut [Vec<f64>], col: usize) {
    let norm = l2_norm_col(w, col);
    if norm > 1.0 {
        for r in w.iter_mut() { r[col] /= norm; }
    }
}

impl Cerebellum {
    /// hidden_dim = 0 → linear (backward compatible)
    pub fn new(dim: usize, n_actions: usize, lr: f64, noise_std: f64, epsilon: f64, hidden_dim: usize) -> Self {
        let is_linear = hidden_dim == 0;
        let hd = if is_linear { 0 } else { hidden_dim };
        let mut rng = rand::thread_rng();

        let w_lin = if is_linear {
            (0..dim).map(|_| (0..n_actions).map(|_| rng.gen_range(-0.01..0.01)).collect()).collect()
        } else { vec![] };
        let e_lin = if is_linear {
            (0..dim).map(|_| vec![0.0; n_actions]).collect()
        } else { vec![] };

        let w1 = if is_linear { vec![] } else {
            (0..hd).map(|_| (0..dim).map(|_| rng.gen_range(-0.01..0.01)).collect()).collect()
        };
        let b1 = if is_linear { vec![] } else { vec![0.0; hd] };
        let w2 = if is_linear { vec![] } else {
            (0..n_actions).map(|_| (0..hd).map(|_| rng.gen_range(-0.01..0.01)).collect()).collect()
        };
        let b2 = if is_linear { vec![] } else { vec![0.0; n_actions] };
        let h = if is_linear { vec![] } else { vec![0.0; hd] };
        let e1 = if is_linear { vec![] } else {
            (0..hd).map(|_| vec![0.0; dim]).collect()
        };
        let e2 = if is_linear { vec![] } else {
            (0..n_actions).map(|_| vec![0.0; hd]).collect()
        };
        let w_v = if is_linear { vec![] } else { vec![0.0; hd] };
        let b_v = 0.0;

        Cerebellum { lr, noise_std, epsilon, dim, n_actions, hidden_dim: hd, is_linear,
            w_lin, e_lin, w1, b1, w2, b2, h, e1, e2, w_v, b_v, lr_critic: lr, v_prev: 0.0,
            rstdp: None,
            replay: ReplayBuffer::new(10000), replay_lr: 0.05, replay_only: false, delta_clip: 0.0 }
    }

    /// Set a different learning rate for the critic (default: same as actor).
    pub fn set_lr_critic(&mut self, lr: f64) { self.lr_critic = lr; }

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
                for i in 0..self.dim { logits[a] += concept[i] * self.w_lin[i][a]; }
                if add_noise { logits[a] += rng.gen_range(-self.noise_std..self.noise_std); }
            }
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap()
        } else {
            let mut hloc = vec![0.0; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut s = self.b1[j];
                for i in 0..self.dim { s += self.w1[j][i] * concept[i]; }
                hloc[j] = tanh(s);
            }
            self.h = hloc;
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                logits[a] = self.b2[a];
                for j in 0..self.hidden_dim { logits[a] += self.w2[a][j] * self.h[j]; }
                if add_noise { logits[a] += rng.gen_range(-self.noise_std..self.noise_std); }
            }
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap()
        }
    }

    /// Returns (action, hidden_activations_copy) — hidden is empty for linear mode.
    pub fn forward_with_hidden(&mut self, concept: &Array1<f64>) -> (usize, Vec<f64>) {
        let action = self.forward(concept);
        (action, self.h.clone())
    }

    /// Retourne les logits bruts sans argmax ni bruit d'exploration.
    /// Sets self.h (MLP) for subsequent value prediction or mark.
    pub fn forward_logits(&mut self, concept: &Array1<f64>) -> Vec<f64> {
        if self.is_linear {
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                for i in 0..self.dim { logits[a] += concept[i] * self.w_lin[i][a]; }
            }
            logits
        } else {
            let mut hloc = vec![0.0; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut s = self.b1[j];
                for i in 0..self.dim { s += self.w1[j][i] * concept[i]; }
                hloc[j] = tanh(s);
            }
            self.h = hloc;
            let mut logits = vec![0.0; self.n_actions];
            for a in 0..self.n_actions {
                logits[a] = self.b2[a];
                for j in 0..self.hidden_dim { logits[a] += self.w2[a][j] * self.h[j]; }
            }
            logits
        }
    }

    /// Expose hidden activations (for external critic use).
    pub fn get_hidden(&self) -> &[f64] { &self.h }

    /// Critic: V(h) = w_v · h + b_v
    pub fn predict_value(&self) -> f64 {
        if self.is_linear { return 0.0; }
        let mut v = self.b_v;
        for j in 0..self.hidden_dim { v += self.w_v[j] * self.h[j]; }
        v
    }

    /// Average |δ| over last N steps (monitoring).
    pub fn critic_learning_rate(&self) -> f64 { self.lr_critic }

    // --- Eligibility trace methods ---

    /// Mark the selected action with the correct gradients.
    /// Caches V(h) for the next TD step.
    pub fn mark(&mut self, concept: &Array1<f64>, action: usize) {
        if self.is_linear {
            for i in 0..self.dim { self.e_lin[i][action] += concept[i]; }
        } else {
            // Cache V(h_t) BEFORE updating h (h is from current forward)
            self.v_prev = self.predict_value();
            for j in 0..self.hidden_dim {
                self.e2[action][j] += self.h[j];
            }
            for j in 0..self.hidden_dim {
                let dh = self.w2[action][j] * tanh_deriv(self.h[j]);
                for i in 0..self.dim {
                    self.e1[j][i] += dh * concept[i];
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

    // --- REINFORCE (Monte Carlo) — backward compatible ---

    /// Classic REINFORCE with reward R.
    /// Linear mode uses this. MLP mode can use it too (treats traces as-is).
    pub fn reinforce(&mut self, reward: f64) {
        if reward.abs() < 1e-6 { return; }
        let step = self.lr * reward.abs();
        let sign = if reward > 0.0 { 1.0 } else { -1.0 };

        if self.is_linear {
            for i in 0..self.dim {
                for a in 0..self.n_actions {
                    self.w_lin[i][a] += sign * step * self.e_lin[i][a];
                }
            }
            for a in 0..self.n_actions { normalize_col(&mut self.w_lin, a); }
        } else {
            for a in 0..self.n_actions {
                for j in 0..self.hidden_dim {
                    self.w2[a][j] += sign * step * self.e2[a][j];
                }
                soft_normalize_row(&mut self.w2[a], 1.2, 0.01);
            }
            for j in 0..self.hidden_dim {
                for i in 0..self.dim {
                    self.w1[j][i] += sign * step * self.e1[j][i];
                }
                soft_normalize_row(&mut self.w1[j], 1.2, 0.01);
            }
        }
    }

    // --- Actor-Critic (TD) ---

    /// TD-Error update: uses δ = R + γ·V(h') − V(h) to update both actor and critic.
    ///
    /// Call order:
    ///   1. forward_logits(x)         — sets self.h = h_t (or h_{t+1})
    ///   2. predict_value()           — V(h_current)
    ///   3. mark(x, a)                — traces accumulate
    /// Then next step:
    ///   4. forward_logits(x')        — self.h = h_{t+1}
    ///   5. predict_value()           — V(h_{t+1})
    ///   6. reinforce_td(R, γ)        — computes δ, updates actor traces + critic
    pub fn reinforce_td(&mut self, reward: f64, gamma: f64) {
        if self.is_linear { return self.reinforce(reward); }
        if self.replay_only { return; }
        if reward.abs() < 1e-6 { return; }

        let v_next = self.predict_value();
        // v_prev was cached during last mark()
        let delta = reward + gamma * v_next - self.v_prev;
        if delta.abs() < 1e-8 { return; }

        // --- Actor update with δ ---
        let clipped_delta = if self.delta_clip > 0.0 { delta.abs().min(self.delta_clip) } else { delta.abs() };
        let step_a = self.lr * clipped_delta;
        // R-STDP
        if let Some(ref r) = self.rstdp { r.apply(&mut self.w_lin, &mut self.w1, &mut self.w2, delta); }

        let sign_a = if delta > 0.0 { 1.0 } else { -1.0 };

        for a in 0..self.n_actions {
            for j in 0..self.hidden_dim {
                self.w2[a][j] += sign_a * step_a * self.e2[a][j];
            }
            soft_normalize_row(&mut self.w2[a], 1.2, 0.01);
        }
        for j in 0..self.hidden_dim {
            for i in 0..self.dim {
                self.w1[j][i] += sign_a * step_a * self.e1[j][i];
            }
            soft_normalize_row(&mut self.w1[j], 1.2, 0.01);
        }

        // --- Critic update with asymmetric LR ---
        // δ > 0 (bonne surprise) : apprentissage rapide pour intégrer les rares succès
        // δ < 0 (mauvaise surprise) : apprentissage lent pour ne pas noyer les signaux rares
        let lr = if delta > 0.0 { self.lr_critic * 5.0 } else { self.lr_critic };
        let step_c = lr * delta;
        for j in 0..self.hidden_dim {
            self.w_v[j] += step_c * self.h[j];
        }
        self.b_v += step_c;
    }

    /// Cache current V(h) for next TD step.
    /// Called automatically at the end of mark().
    pub fn cache_value(&mut self) {
        if !self.is_linear {
            self.v_prev = self.predict_value();
        }
    }

    // --- Replay Buffer Training ---

    /// Stocke une transition dans le replay buffer.
    pub fn store_transition(&mut self, state: &Array1<f64>, action: usize, reward: f64, next_state: &Array1<f64>, done: bool) {
        if !self.is_linear {
            self.replay.store(state, action, reward, next_state, done);
        }
    }

    /// Entraîne le réseau sur un mini-batch tiré du replay buffer.
    /// Effectue `steps` mises à jour. Retourne l'erreur TD moyenne.
    pub fn replay_train(&mut self, batch_size: usize, gamma: f64, steps: usize) -> f64 {
        if self.is_linear || self.replay.len() < batch_size { return 0.0; }
        let mut total_delta = 0.0;
        let mut count = 0usize;

        for _ in 0..steps {
            let batch = self.replay.sample(batch_size);
            for t in &batch {
                let state = Array1::from_vec(t.state.clone());
                let next_state = Array1::from_vec(t.next_state.clone());

                // Forward pass on current state
                let mut hloc = vec![0.0; self.hidden_dim];
                for j in 0..self.hidden_dim {
                    let mut s = self.b1[j];
                    for i in 0..self.dim { s += self.w1[j][i] * state[i]; }
                    hloc[j] = tanh(s);
                }

                // V(s)
                let mut vs = self.b_v;
                for j in 0..self.hidden_dim { vs += self.w_v[j] * hloc[j]; }

                // Forward pass on next state
                let mut hloc_next = vec![0.0; self.hidden_dim];
                for j in 0..self.hidden_dim {
                    let mut s = self.b1[j];
                    for i in 0..self.dim { s += self.w1[j][i] * next_state[i]; }
                    hloc_next[j] = tanh(s);
                }

                // V(s')
                let mut v_next = self.b_v;
                for j in 0..self.hidden_dim { v_next += self.w_v[j] * hloc_next[j]; }

                // TD target
                let target = if t.done { t.reward } else { t.reward + gamma * v_next };
                let delta = target - vs;

                // Critic update
                let step_c = self.replay_lr * delta;
                for j in 0..self.hidden_dim { self.w_v[j] += step_c * hloc[j]; }
                self.b_v += step_c;

                // Actor update
                let step_a = self.replay_lr * delta * 0.5;
                for j in 0..self.hidden_dim {
                    self.w2[t.action][j] += step_a * hloc[j];
                }
                soft_normalize_row(&mut self.w2[t.action], 1.2, 0.01);

                total_delta += delta.abs();
                count += 1;
            }
        }
        if count > 0 { total_delta / count as f64 } else { 0.0 }
    }

    // --- Reset ---

    pub fn reset_trace(&mut self) {
        self.v_prev = 0.0;
        if self.is_linear {
            for i in 0..self.dim { for a in 0..self.n_actions { self.e_lin[i][a] = 0.0; } }
        } else {
            for j in 0..self.hidden_dim { for i in 0..self.dim { self.e1[j][i] = 0.0; } }
            for a in 0..self.n_actions { for j in 0..self.hidden_dim { self.e2[a][j] = 0.0; } }
        }
    }

    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        if self.is_linear {
            for i in 0..self.dim { for a in 0..self.n_actions { self.w_lin[i][a] = rng.gen_range(-0.01..0.01); } }
        } else {
            for j in 0..self.hidden_dim { for i in 0..self.dim { self.w1[j][i] = rng.gen_range(-0.01..0.01); } }
            for a in 0..self.n_actions { for j in 0..self.hidden_dim { self.w2[a][j] = rng.gen_range(-0.01..0.01); } }
            self.w_v = vec![0.0; self.hidden_dim];
            self.b_v = 0.0;
        }
        self.reset_trace();
        self.v_prev = 0.0;
    }

    // --- Legacy learn method (used by graph actor in core.rs) ---
    pub fn learn(&mut self, concept: &Array1<f64>, action: usize, reward: f64) {
        if reward.abs() < 1e-6 { return; }
        let step = self.lr * reward.abs();
        let sign = if reward > 0.0 { 1.0 } else { -1.0 };

        if self.is_linear {
            for i in 0..self.dim { self.w_lin[i][action] += sign * step * concept[i]; }
        } else {
            for j in 0..self.hidden_dim {
                self.w2[action][j] += sign * step * self.h[j];
            }
            self.b2[action] += sign * step;
            for j in 0..self.hidden_dim {
                let dh = sign * step * tanh_deriv(self.h[j]);
                for i in 0..self.dim { self.w1[j][i] += dh * concept[i]; }
                self.b1[j] += dh;
            }
        }
    }

    /// Metabolic cost per tick.
    /// Linear mode is cheap (simple direct mapping).
    /// MLP mode is expensive (many synapses to activate).
    pub fn is_linear(&self) -> bool { self.is_linear }

    pub fn compute_cost(&self) -> f64 {
        if self.is_linear {
            1.0
        } else {
            2.0
        }
    }

    // --- Inspection helpers ---
    pub fn get_lin_weight(&self, i: usize, a: usize) -> f64 {
        if self.is_linear && i < self.w_lin.len() && a < self.w_lin[0].len() { self.w_lin[i][a] } else { 0.0 }
    }
    pub fn get_hidden_weight(&self, j: usize, i: usize) -> f64 {
        if !self.is_linear && j < self.w1.len() && i < self.w1[0].len() { self.w1[j][i] } else { 0.0 }
    }
    pub fn get_out_weight(&self, a: usize, j: usize) -> f64 {
        if !self.is_linear && a < self.w2.len() && j < self.w2[0].len() { self.w2[a][j] } else { 0.0 }
    }
}

fn soft_normalize_row(row: &mut [f64], threshold: f64, rate: f64) {
    let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > threshold {
        let scale = 1.0 / (1.0 + rate * (norm - threshold));
        for x in row.iter_mut() { *x *= scale; }
    }
}
