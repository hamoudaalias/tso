/// Rotating-T Aliased: observation identique pour plusieurs buts.
/// L'agent doit utiliser la mémoire épisodique pour désambiguïser.
use ndarray::Array1;

const W: usize = 5;
const H: usize = 5;

pub struct RotatingT {
    pub agent: (usize, usize),
    pub goal: (usize, usize),
    pub episode: usize,
    pub switch_every: usize,
    pub steps: usize,
    pub max_steps: usize,
    pub phase_count: usize,
    phase_ep_count: usize,
    // Cache pour envoyer au engine
    pub prev_concept: Option<usize>,
    pub prev_obs: Array1<f64>,
}

impl RotatingT {
    pub fn new(switch_every: usize) -> Self {
        RotatingT {
            agent: (0, 2),
            goal: (4, 0),
            episode: 0,
            switch_every,
            steps: 0,
            max_steps: 20,
            phase_count: 0,
            phase_ep_count: 0,
            prev_concept: None,
            prev_obs: Array1::zeros(4),
        }
    }

    pub fn reset(&mut self) {
        self.agent = (0, 2);
        self.steps = 0;
        self.episode += 1;
        self.phase_ep_count += 1;

        if self.phase_ep_count >= self.switch_every {
            self.phase_ep_count = 0;
            self.phase_count += 1;
            // 4 goal positions, only 2 distinct observations
            let goals = [(4,0), (0,4), (4,4), (0,0)];
            self.goal = goals[self.phase_count % 4];
        }
        self.prev_obs = self.observation();
    }

    pub fn step(&mut self, action: usize) -> (f64, Array1<f64>, bool) {
        self.steps += 1;
        let (x, y) = self.agent;
        let (nx, ny) = match action {
            0 if y > 0 => (x, y - 1),
            1 if x < W - 1 => (x + 1, y),
            2 if y < H - 1 => (x, y + 1),
            3 if x > 0 => (x - 1, y),
            _ => (x, y),
        };
        self.agent = (nx, ny);
        let done = self.steps >= self.max_steps || (nx, ny) == self.goal;
        let reward = if (nx, ny) == self.goal { 10.0 } else { -0.1 };

        let obs = self.observation();
        self.prev_obs = obs.clone();

        (reward, obs, done)
    }

    /// Aliased observation: 4 whiskers only, NO goal direction.
    /// (0,4) and (4,0) produce SAME observation from start position (0,2).
    /// Agent must use episodic memory to know which goal is active.
    pub fn observation(&self) -> Array1<f64> {
        let (x, y) = self.agent;
        Array1::from_vec(vec![
            if y == 0 { 1.0 } else { 0.0 },
            if x == W - 1 { 1.0 } else { 0.0 },
            if y == H - 1 { 1.0 } else { 0.0 },
            if x == 0 { 1.0 } else { 0.0 },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_observation_dim() {
        let rt = RotatingT::new(50);
        assert_eq!(rt.observation().len(), 4);
    }
    #[test]
    fn test_goal_aliasing() {
        // (4,0) and (0,4) should give same observation from (0,2)
        let mut rt = RotatingT::new(50);
        let obs1 = rt.observation();
        rt.goal = (0, 4);
        let obs2 = rt.observation();
        assert_eq!(obs1, obs2);
    }
}
