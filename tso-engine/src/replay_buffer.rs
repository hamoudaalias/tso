use ndarray::Array1;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Transition stockée dans le buffer de replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Replay buffer — stocke les transitions récentes et échantillonne
/// des mini-batchs pour stabiliser l'apprentissage TD.
///
/// Résout le problème des récompenses rares/différées en permettant
/// au réseau de rejouer les expériences passées plusieurs fois.
#[derive(Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    capacity: usize,
    buffer: Vec<Transition>,
    pos: usize,
    size: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buffer: Vec::with_capacity(capacity),
            pos: 0,
            size: 0,
        }
    }

    pub fn store(&mut self, state: &Array1<f64>, action: usize, reward: f64, next_state: &Array1<f64>, done: bool) {
        let t = Transition {
            state: state.to_vec(),
            action,
            reward,
            next_state: next_state.to_vec(),
            done,
        };
        if self.size < self.capacity {
            self.buffer.push(t);
            self.size += 1;
        } else {
            self.buffer[self.pos] = t;
        }
        self.pos = (self.pos + 1) % self.capacity;
    }

    /// Échantillonne un mini-batch aléatoire.
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::thread_rng();
        (0..batch_size.min(self.size))
            .map(|_| &self.buffer[rng.gen_range(0..self.size)])
            .collect()
    }

    pub fn len(&self) -> usize { self.size }
    pub fn capacity(&self) -> usize { self.capacity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_state(v: f64) -> Array1<f64> {
        Array1::from_vec(vec![v; 4])
    }

    #[test]
    fn test_new_is_empty() {
        let rb = ReplayBuffer::new(100);
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 100);
    }

    #[test]
    fn test_store_and_grow() {
        let mut rb = ReplayBuffer::new(10);
        for i in 0..5 {
            rb.store(&make_state(i as f64), i % 2, i as f64, &make_state(i as f64 + 1.0), false);
        }
        assert_eq!(rb.len(), 5);
    }

    #[test]
    fn test_circular_overwrite() {
        let mut rb = ReplayBuffer::new(3);
        for i in 0..6 {
            rb.store(&make_state(i as f64), 0, i as f64, &make_state(i as f64), false);
        }
        assert_eq!(rb.len(), 3, "should cap at 3");
    }

    #[test]
    fn test_sample_returns_requested_size() {
        let mut rb = ReplayBuffer::new(20);
        for i in 0..10 {
            rb.store(&make_state(i as f64), 0, i as f64, &make_state(i as f64), false);
        }
        let batch = rb.sample(5);
        assert_eq!(batch.len(), 5, "should return 5 transitions");
    }

    #[test]
    fn test_sample_limited_by_size() {
        let mut rb = ReplayBuffer::new(20);
        for i in 0..3 {
            rb.store(&make_state(i as f64), 0, i as f64, &make_state(i as f64), false);
        }
        let batch = rb.sample(10);
        assert_eq!(batch.len(), 3, "should return at most 3 (size)");
    }

    #[test]
    fn test_store_preserves_data() {
        let mut rb = ReplayBuffer::new(10);
        rb.store(&make_state(1.0), 2, 3.0, &make_state(4.0), true);
        assert_eq!(rb.len(), 1);
        let batch = rb.sample(1);
        assert_eq!(batch[0].action, 2);
        assert!((batch[0].reward - 3.0).abs() < 1e-6);
        assert!(batch[0].done);
    }

    #[test]
    fn test_empty_sample_returns_empty() {
        let rb: ReplayBuffer = ReplayBuffer::new(10);
        let batch = rb.sample(5);
        assert!(batch.is_empty());
    }
}
