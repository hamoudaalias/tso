use ndarray::Array1;
use crate::neurons::DualLIFState;
use crate::memory::AssociativeMemory;

pub struct WorkingMemory {
    pub lif: DualLIFState,
    pub assoc: AssociativeMemory,
    dim: usize,
    locked: bool,
}

impl WorkingMemory {
    pub fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self {
        WorkingMemory {
            lif: DualLIFState::new(dim, alpha_slow, alpha_fast),
            assoc: AssociativeMemory::new(),
            dim,
            locked: false,
        }
    }

    pub fn observe(&mut self, objects: &[Array1<f64>]) -> Option<(usize, f64)> {
        for obj in objects {
            self.lif.step(obj, false);
        }
        if let Some(first) = objects.first() {
            if self.assoc.size() == 0 {
                self.assoc.store(first, 0);
                self.locked = true;
                return None;
            }
        }
        let mut best: Option<(usize, f64)> = None;
        for obj in objects {
            if let Some(result) = self.assoc.recall_with_sim(obj) {
                if best.map_or(true, |(_, s)| result.1 > s) {
                    best = Some(result);
                }
            }
        }
        best
    }

    pub fn recall(&self, query: &Array1<f64>) -> Option<(usize, f64)> {
        self.assoc.recall_with_sim(query)
    }

    pub fn reset(&mut self) {
        self.lif = DualLIFState::new(self.dim, 0.99, 0.5);
        self.assoc = AssociativeMemory::new();
        self.locked = false;
    }

    pub fn store(&mut self, vector: &Array1<f64>, data: usize) {
        self.assoc.store(vector, data);
        self.locked = true;
    }

    pub fn has_target(&self) -> bool {
        self.assoc.size() > 0
    }
}
