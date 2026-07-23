use ndarray::Array1;
use crate::neurons::DualLIFState;

pub struct ActionMotor {
    pub beta: f64,
}

impl ActionMotor {
    pub fn new(beta: f64) -> Self {
        ActionMotor { beta }
    }

    pub fn select(&self, context: &DualLIFState, actions: &[Array1<f64>]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, action_vec) in actions.iter().enumerate() {
            let score = context.alignment(action_vec, self.beta);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        (best_idx, best_score)
    }

    pub fn select_with_bonus(
        &self,
        context: &DualLIFState,
        actions: &[Array1<f64>],
        bonuses: &[f64],
    ) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, action_vec) in actions.iter().enumerate() {
            let align = context.alignment(action_vec, self.beta);
            let bonus = bonuses.get(i).copied().unwrap_or(0.0);
            let score = align + bonus;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        (best_idx, best_score)
    }
}
