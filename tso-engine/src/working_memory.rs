use ndarray::Array1;
use serde::{Serialize, Deserialize};
use crate::neurons::DualLIFState;


#[derive(Serialize, Deserialize)]
pub struct WorkingMemory {
    pub lif: DualLIFState,
    pub assoc: AssociativeMemory,
    dim: usize,
    locked: bool,
    pub cue_latch: f64,
}

impl WorkingMemory {
    pub fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self {
        WorkingMemory {
            lif: DualLIFState::new(dim, alpha_slow, alpha_fast),
            assoc: AssociativeMemory::new(),
            dim,
            locked: false,
            cue_latch: 0.0,
        }
    }

    pub fn observe(&mut self, objects: &[Array1<f64>]) -> Option<(usize, f64)> {
        for obj in objects {
            self.lif.step(obj, false);
        }
        // Latch the cue from the first perception for POMDP tasks.
        // Le cue (non-nul dans la première observation) est maintenu artificiellement
        // dans cue_latch pour que le cervelet puisse le voir à chaque pas.
        if self.cue_latch == 0.0 {
            if let Some(first) = objects.first() {
                if first.len() > 4 {
                    self.cue_latch = first[4];
                }
            }
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
        self.cue_latch = 0.0;
    }

    pub fn store(&mut self, vector: &Array1<f64>, data: usize) {
        self.assoc.store(vector, data);
        self.locked = true;
    }

    pub fn has_target(&self) -> bool {
        self.assoc.size() > 0
    }
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub vector: Array1<f64>,
    pub data: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AssociativeMemory {
    pub entries: Vec<Entry>,
}

impl AssociativeMemory {
    pub fn new() -> Self {
        AssociativeMemory { entries: Vec::new() }
    }

    pub fn store(&mut self, vector: &Array1<f64>, data: usize) {
        self.entries.push(Entry {
            vector: vector.clone(),
            data,
        });
    }

    pub fn recall(&self, query: &Array1<f64>) -> Option<usize> {
        let mut best_sim = -1.0;
        let mut best_data = None;
        for e in &self.entries {
            let sim = cosine_sim(query, &e.vector);
            if sim > best_sim {
                best_sim = sim;
                best_data = Some(e.data);
            }
        }
        best_data
    }

    pub fn recall_with_sim(&self, query: &Array1<f64>) -> Option<(usize, f64)> {
        let mut best_sim = -1.0;
        let mut best_data = None;
        for e in &self.entries {
            let sim = cosine_sim(query, &e.vector);
            if sim > best_sim {
                best_sim = sim;
                best_data = Some(e.data);
            }
        }
        best_data.map(|d| (d, best_sim))
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

fn cosine_sim(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let na = a.dot(a).sqrt().max(1e-12);
    let nb = b.dot(b).sqrt().max(1e-12);
    dot / (na * nb)
}
