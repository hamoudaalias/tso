use ndarray::{Array2, Array1};
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct RstdpPlasticity {
    pub learning_rate: f64,
    pub trace_decay: f64,
    pub e_lin: Array2<f64>,
    pub e1: Array2<f64>,
    pub e2: Array2<f64>,
}

impl RstdpPlasticity {
    pub fn new(dim: usize, hidden_dim: usize, n_actions: usize, lr: f64) -> Self {
        RstdpPlasticity {
            learning_rate: lr,
            trace_decay: 0.98,
            e_lin: if hidden_dim == 0 { Array2::zeros((dim, n_actions)) } else { Array2::zeros((0, 0)) },
            e1: if hidden_dim > 0 { Array2::zeros((hidden_dim, dim)) } else { Array2::zeros((0, 0)) },
            e2: if hidden_dim > 0 { Array2::zeros((n_actions, hidden_dim)) } else { Array2::zeros((0, 0)) },
        }
    }

    pub fn update_trace(&mut self, pre: &Array1<f64>, post: &Array1<f64>, hidden: &Array1<f64>) {
        let td = self.trace_decay;
        if self.e_lin.len() > 0 {
            for i in 0..self.e_lin.shape()[0] {
                for a in 0..self.e_lin.shape()[1] {
                    self.e_lin[[i, a]] = td * self.e_lin[[i, a]] + pre[i] * post[a];
                }
            }
        } else {
            for a in 0..self.e2.shape()[0] {
                for j in 0..self.e2.shape()[1] {
                    self.e2[[a, j]] = td * self.e2[[a, j]] + post[a] * hidden[j];
                }
            }
            for j in 0..self.e1.shape()[0] {
                for i in 0..self.e1.shape()[1] {
                    self.e1[[j, i]] = td * self.e1[[j, i]] + hidden[j] * pre[i];
                }
            }
        }
    }

    pub fn apply(&self, w_lin: &mut [Vec<f64>], w1: &mut [Vec<f64>], w2: &mut [Vec<f64>], delta: f64) {
        let step = self.learning_rate * delta.abs().min(5.0);
        let sign = if delta > 0.0 { 1.0 } else { -1.0 };
        if !w_lin.is_empty() {
            for i in 0..w_lin.len() {
                for a in 0..w_lin[i].len() {
                    w_lin[i][a] += sign * step * self.e_lin[[i, a]];
                }
            }
        } else {
            for a in 0..w2.len() {
                for j in 0..w2[a].len() {
                    w2[a][j] += sign * step * self.e2[[a, j]];
                }
            }
            for j in 0..w1.len() {
                for i in 0..w1[j].len() {
                    w1[j][i] += sign * step * self.e1[[j, i]];
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.e_lin.fill(0.0);
        self.e1.fill(0.0);
        self.e2.fill(0.0);
    }
}
