use ndarray::Array1;
use rand::Rng;
use crate::replay_buffer::ReplayBuffer;

/// Simple DQN agent with target network and replay buffer.
pub struct DqnAgent {
    pub dim: usize,
    pub n_actions: usize,
    pub hidden_dim: usize,
    pub lr: f64,
    pub epsilon: f64,
    pub gamma: f64,

    /// Online Q-network
    pub q: QNetwork,
    /// Target Q-network
    pub q_target: QNetwork,

    pub replay: ReplayBuffer,
    pub batch_size: usize,
    step_count: usize,
    target_update_freq: usize,
}

pub struct QNetwork {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}

fn tanh(x: f64) -> f64 { x.tanh() }

impl QNetwork {
    pub fn new(dim: usize, hidden_dim: usize, n_actions: usize) -> Self {
        let r = 0.01;
        let mut rng = rand::thread_rng();
        QNetwork {
            w1: (0..hidden_dim).map(|_| (0..dim).map(|_| rng.gen_range(-r..r)).collect()).collect(),
            b1: vec![0.0; hidden_dim],
            w2: (0..n_actions).map(|_| (0..hidden_dim).map(|_| rng.gen_range(-r..r)).collect()).collect(),
            b2: vec![0.0; n_actions],
        }
    }

    pub fn forward(&self, obs: &Array1<f64>) -> Vec<f64> {
        let mut h = vec![0.0; self.b1.len()];
        for j in 0..h.len() {
            let mut s = self.b1[j];
            for i in 0..obs.len() {
                s += self.w1[j][i] * obs[i];
            }
            h[j] = tanh(s);
        }
        let mut qvals = vec![0.0; self.w2.len()];
        for a in 0..qvals.len() {
            let mut s = self.b2[a];
            for j in 0..h.len() {
                s += self.w2[a][j] * h[j];
            }
            qvals[a] = s;
        }
        qvals
    }

    pub fn forward_with_hidden(&self, obs: &Array1<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut h = vec![0.0; self.b1.len()];
        for j in 0..h.len() {
            let mut s = self.b1[j];
            for i in 0..obs.len() {
                s += self.w1[j][i] * obs[i];
            }
            h[j] = tanh(s);
        }
        let mut qvals = vec![0.0; self.w2.len()];
        for a in 0..qvals.len() {
            let mut s = self.b2[a];
            for j in 0..h.len() {
                s += self.w2[a][j] * h[j];
            }
            qvals[a] = s;
        }
        (qvals, h)
    }
}

impl DqnAgent {
    pub fn new(dim: usize, n_actions: usize, hidden_dim: usize, lr: f64, epsilon: f64) -> Self {
        DqnAgent {
            dim,
            n_actions,
            hidden_dim,
            lr,
            epsilon,
            gamma: 0.99,
            q: QNetwork::new(dim, hidden_dim, n_actions),
            q_target: QNetwork::new(dim, hidden_dim, n_actions),
            replay: ReplayBuffer::new(100_000),
            batch_size: 32,
            step_count: 0,
            target_update_freq: 100,
        }
    }

    pub fn act(&mut self, obs: &Array1<f64>) -> usize {
        if rand::random::<f64>() < self.epsilon {
            rand::random::<usize>() % self.n_actions
        } else {
            let qvals = self.q.forward(obs);
            qvals.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }

    pub fn store(&mut self, obs: &Array1<f64>, action: usize, reward: f64, next_obs: &Array1<f64>, done: bool) {
        self.replay.store(obs, action, reward, next_obs, done);
    }

    pub fn update_target(&mut self, tau: f64) {
        for j in 0..self.hidden_dim {
            for i in 0..self.dim {
                self.q_target.w1[j][i] = tau * self.q.w1[j][i] + (1.0 - tau) * self.q_target.w1[j][i];
            }
            self.q_target.b1[j] = tau * self.q.b1[j] + (1.0 - tau) * self.q_target.b1[j];
        }
        for a in 0..self.n_actions {
            for j in 0..self.hidden_dim {
                self.q_target.w2[a][j] = tau * self.q.w2[a][j] + (1.0 - tau) * self.q_target.w2[a][j];
            }
            self.q_target.b2[a] = tau * self.q.b2[a] + (1.0 - tau) * self.q_target.b2[a];
        }
    }

    pub fn train(&mut self, batch_size: usize) -> f64 {
        if self.replay.len() < batch_size { return 0.0; }
        let batch = self.replay.sample(batch_size);
        let mut total_loss = 0.0;

        for t in batch {
            let obs = Array1::from_vec(t.state.clone());
            let action = t.action;
            let reward = t.reward;
            let next_obs = Array1::from_vec(t.next_state.clone());
            let done = t.done;

            let (qvals, h) = self.q.forward_with_hidden(&obs);
            let q_current = qvals[action];

            let q_next = if done {
                0.0
            } else {
                let q_target_vals = self.q_target.forward(&next_obs);
                q_target_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };
            let target = reward + self.gamma * q_next;
            let td_error = target - q_current;
            total_loss += td_error * td_error;

            let lr = self.lr;
            let dh = td_error * lr;

            // w2 gradient
            for j in 0..self.hidden_dim {
                self.q.w2[action][j] += dh * h[j];
            }
            self.q.b2[action] += dh;

            // w1 gradient (backprop through tanh)
            for j in 0..self.hidden_dim {
                let tanh_deriv = 1.0 - h[j] * h[j];
                let grad_w1 = dh * self.q.w2[action][j] * tanh_deriv;
                for i in 0..self.dim {
                    self.q.w1[j][i] += lr * grad_w1 * obs[i];
                }
                self.q.b1[j] += lr * grad_w1;
            }
        }

        self.step_count += 1;
        if self.step_count % self.target_update_freq == 0 {
            self.update_target(1.0);
        }

        total_loss / batch_size as f64
    }
}
