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

    pub fn membrane_potential(&self) -> (Array1<f64>, Array1<f64>) {
        (self.lif.slow.state.clone(), self.lif.fast.state.clone())
    }

    pub fn spike_rate(&self) -> (f64, f64) {
        let slow_rate = self.lif.slow.state.mapv(|x| x.max(0.0)).mean().unwrap_or(0.0);
        let fast_rate = self.lif.fast.state.mapv(|x| x.max(0.0)).mean().unwrap_or(0.0);
        (slow_rate, fast_rate)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_pulse_response_fast_dominates() {
        let mut wm = WorkingMemory::new(4, 0.9, 0.5);

        let (s0, f0) = wm.membrane_potential();
        assert!(s0.iter().all(|x| *x == 0.0));
        assert!(f0.iter().all(|x| *x == 0.0));

        let pulse = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        wm.observe(&[pulse]);

        let (slow, fast) = wm.membrane_potential();
        let slow_mag = slow.dot(&slow);
        let fast_mag = fast.dot(&fast);

        assert!(fast_mag > slow_mag, "fast should integrate more of the pulse than slow");
        assert!(slow[0] > 0.0, "slow should have non-zero state after pulse");
        assert!(fast[0] > 0.0, "fast should have non-zero state after pulse");

        assert!(approx_eq(slow[0], (1.0 - 0.9) * 1.0, 1e-6));
        assert!(approx_eq(fast[0], (1.0 - 0.5) * 1.0, 1e-6));
    }

    #[test]
    fn test_decay_rates_differ() {
        let mut wm = WorkingMemory::new(4, 0.9, 0.5);

        let pulse = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        wm.observe(&[pulse]);

        let zero = Array1::zeros(4);
        for _ in 0..10 {
            wm.observe(&[zero.clone()]);
        }

        let (slow, fast) = wm.membrane_potential();
        let slow_mag = slow.dot(&slow);
        let fast_mag = fast.dot(&fast);

        assert!(slow_mag > fast_mag, "after decay, slow should retain more than fast");
        assert!(fast_mag < 0.01, "fast should decay near zero: {}", fast_mag);
        assert!(slow[0] > 0.0, "slow should retain some signal after 10 steps");
    }

    #[test]
    fn test_spike_rate_after_pulse() {
        let mut wm = WorkingMemory::new(4, 0.9, 0.5);

        let (sr, fr) = wm.spike_rate();
        assert!(approx_eq(sr, 0.0, 1e-10));
        assert!(approx_eq(fr, 0.0, 1e-10));

        let pulse = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
        wm.observe(&[pulse]);

        let (sr, fr) = wm.spike_rate();
        assert!(sr > 0.0, "slow spike rate should be positive");
        assert!(fr > 0.0, "fast spike rate should be positive");
        assert!(fr > sr, "fast spike rate should exceed slow for a fresh pulse");
    }

    #[test]
    fn test_observe_and_recall() {
        let mut wm = WorkingMemory::new(4, 0.9, 0.5);

        let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);

        wm.store(&v1, 42);
        wm.store(&v2, 99);

        let r1 = wm.recall(&v1);
        assert_eq!(r1, Some((42, 1.0)));

        let partial = Array1::from_vec(vec![0.0, 0.8, 0.0, 0.0]);
        let r2 = wm.recall(&partial);
        assert_eq!(r2.map(|r| r.0), Some(99));
    }

    #[test]
    fn test_reset_clears_state() {
        let mut wm = WorkingMemory::new(4, 0.9, 0.5);

        let pulse = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        wm.observe(&[pulse]);
        wm.reset();

        let (slow, fast) = wm.membrane_potential();
        assert!(slow.iter().all(|x| *x == 0.0));
        assert!(fast.iter().all(|x| *x == 0.0));
        assert_eq!(wm.assoc.size(), 0);
    }
}
