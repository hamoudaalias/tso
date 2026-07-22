use ndarray::Array1;

pub struct LIFState {
    pub state: Array1<f64>,
    pub alpha: f64,
}

impl LIFState {
    pub fn new(dim: usize, alpha: f64) -> Self {
        LIFState { state: Array1::zeros(dim), alpha }
    }

    pub fn step(&mut self, embedding: &Array1<f64>, negate: bool) {
        let e = if negate { -embedding } else { embedding.clone() };
        self.state = self.alpha * &self.state + (1.0 - self.alpha) * e;
    }
}

pub struct DualLIFState {
    pub slow: LIFState,
    pub fast: LIFState,
}

impl DualLIFState {
    pub fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self {
        DualLIFState {
            slow: LIFState::new(dim, alpha_slow),
            fast: LIFState::new(dim, alpha_fast),
        }
    }

    pub fn step(&mut self, embedding: &Array1<f64>, negate: bool) {
        self.slow.step(embedding, negate);
        self.fast.step(embedding, negate);
    }

    pub fn alignment(&self, embedding: &Array1<f64>, beta: f64) -> f64 {
        beta * self.slow.state.dot(embedding) + (1.0 - beta) * self.fast.state.dot(embedding)
    }
}
