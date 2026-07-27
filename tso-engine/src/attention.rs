use ndarray::Array1;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Attention {
    pub temperature: f64,
}

impl Attention {
    pub fn new(temperature: f64) -> Self {
        Attention { temperature }
    }

    /// Apply spatial attention to 4 whisker dimensions.
    /// Boosts dimensions where the perception differs most from the predicted
    /// concept prototype, simulating the organism "turning its head" toward
    /// unexpected stimuli. Non-whisker dimensions (e.g. BFS) pass through unchanged.
    pub fn attend(
        &self,
        perception: &Array1<f64>,
        predicted_prototype: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let n_dims = perception.len();
        let whisker_dims = n_dims.min(4);

        let weights = match predicted_prototype {
            Some(proto) if proto.len() >= whisker_dims => {
                let mut diffs: Vec<f64> = (0..whisker_dims)
                    .map(|i| (perception[i] - proto[i]).abs())
                    .collect();
                let max_diff = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_diff.is_finite() && max_diff > 1e-12 {
                    for d in diffs.iter_mut() {
                        *d -= max_diff;
                    }
                } else {
                    diffs.iter_mut().for_each(|d| *d = 0.0);
                }
                let exps: Vec<f64> = diffs.iter()
                    .map(|d| (d / self.temperature).exp())
                    .collect();
                let sum_exp: f64 = exps.iter().sum();
                if sum_exp > 0.0 {
                    let softmax: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();
                    let mean = softmax.iter().sum::<f64>() / whisker_dims as f64;
                    if mean > 0.0 {
                        softmax.iter().map(|w| w / mean).collect()
                    } else {
                        vec![1.0; whisker_dims]
                    }
                } else {
                    vec![1.0; whisker_dims]
                }
            }
            _ => vec![1.0; whisker_dims],
        };

        let mut gated = perception.clone();
        for i in 0..whisker_dims {
            gated[i] = (gated[i] * weights[i]).clamp(0.0, 2.0);
        }
        gated
    }
}
