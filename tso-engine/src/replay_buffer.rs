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
