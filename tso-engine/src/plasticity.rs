//! Plasticite locale (R-STDP).

use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct RstdpPlasticity {
    pub learning_rate: f64,
    pub trace_decay: f64,
    pub e_lin: Vec<Vec<f64>>,
    pub e1: Vec<Vec<f64>>,
    pub e2: Vec<Vec<f64>>,
}

impl RstdpPlasticity {
    pub fn new(dim: usize, hidden_dim: usize, n_actions: usize, lr: f64) -> Self {
        RstdpPlasticity {
            learning_rate: lr,
            trace_decay: 0.98,
            e_lin: if hidden_dim == 0 { vec![vec![0.0; n_actions]; dim] } else { vec![] },
            e1: if hidden_dim > 0 { vec![vec![0.0; dim]; hidden_dim] } else { vec![] },
            e2: if hidden_dim > 0 { vec![vec![0.0; hidden_dim]; n_actions] } else { vec![] },
        }
    }

    /// Met a jour les traces avec pre*post (STDP). Appele a chaque mark().
    pub fn update_trace(&mut self, pre: &[f64], post: &[f64], hidden: &[f64]) {
        if !self.e_lin.is_empty() {
            for i in 0..self.e_lin.len() {
                for a in 0..self.e_lin[i].len() {
                    self.e_lin[i][a] = self.trace_decay * self.e_lin[i][a] + pre[i] * post[a];
                }
            }
        } else {
            for a in 0..self.e2.len() {
                for j in 0..self.e2[a].len() {
                    self.e2[a][j] = self.trace_decay * self.e2[a][j] + post[a] * hidden[j];
                }
            }
            for j in 0..self.e1.len() {
                for i in 0..self.e1[j].len() {
                    self.e1[j][i] = self.trace_decay * self.e1[j][i] + hidden[j] * pre[i];
                }
            }
        }
    }

    /// Applique les traces cumulees via delta_w = lr * delta * trace.
    pub fn apply(&self, w_lin: &mut [Vec<f64>], w1: &mut [Vec<f64>], w2: &mut [Vec<f64>], delta: f64) {
        let step = self.learning_rate * delta.abs().min(5.0);
        let sign = if delta > 0.0 { 1.0 } else { -1.0 };
        if !w_lin.is_empty() {
            for i in 0..w_lin.len() {
                for a in 0..w_lin[i].len() {
                    w_lin[i][a] += sign * step * self.e_lin[i][a];
                }
            }
        } else {
            for a in 0..w2.len() {
                for j in 0..w2[a].len() {
                    w2[a][j] += sign * step * self.e2[a][j];
                }
            }
            for j in 0..w1.len() {
                for i in 0..w1[j].len() {
                    w1[j][i] += sign * step * self.e1[j][i];
                }
            }
        }
    }

    pub fn reset(&mut self) {
        for row in self.e_lin.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
        for row in self.e1.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
        for row in self.e2.iter_mut() { for v in row.iter_mut() { *v = 0.0; } }
    }
}