use ndarray::Array1;
use crate::grid::GridWorld;

const EMBED_DIM: usize = 16;

pub struct StateEncoder {
    pub dim: usize,
}

impl StateEncoder {
    pub fn new() -> Self {
        StateEncoder { dim: EMBED_DIM }
    }

    pub fn encode(&self, gw: &GridWorld) -> Array1<f64> {
        let mut v = Array1::zeros(EMBED_DIM);

        let idx = gw.agent_y * gw.width + gw.agent_x;

        if idx < EMBED_DIM {
            v[idx] = 1.0;
        }

        let gx_norm = gw.goal_x as f64 / (gw.width - 1).max(1) as f64;
        let gy_norm = gw.goal_y as f64 / (gw.height - 1).max(1) as f64;

        let half = EMBED_DIM / 2;
        let goal_bucket_x = (gx_norm * (half - 1) as f64).round() as usize;
        let goal_bucket_y = (gy_norm * (half - 1) as f64).round() as usize;
        if half + goal_bucket_x < EMBED_DIM {
            v[half + goal_bucket_x] = 0.5;
        }
        if goal_bucket_y < half {
            v[goal_bucket_y] += 0.3;
        }

        v
    }
}
