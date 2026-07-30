/// Pure Rust MiniGrid: 7x7 grid with visual observation (RGB 7x7x3).
use ndarray::Array1;

pub const DEFAULT_W: usize = 7;
pub const DEFAULT_H: usize = 7;



pub struct MiniGridEnv {
    pub w: usize,
    pub h: usize,
    pub max_steps: usize,
    pub agent: (usize, usize),
    pub goal: (usize, usize),
    pub door: (usize, usize),
    pub key: (usize, usize),
    pub has_key: bool,
    pub steps: usize,
    pub done: bool,
    pub obs_buf: Array1<f64>,
}

impl MiniGridEnv {
    pub fn new() -> Self {
        Self::with_size(DEFAULT_W, DEFAULT_H)
    }

    pub fn with_size(w: usize, h: usize) -> Self {
        MiniGridEnv {
            w, h,
            max_steps: w * h,
            agent: (1, 1),
            goal: (w - 2, h - 2),
            door: (w / 2, h / 2),
            key: (1, h / 2 + 2),
            has_key: false,
            steps: 0,
            done: false,
            obs_buf: Array1::zeros(w * h * 3),
        }
    }

    pub fn reset(&mut self) -> Array1<f64> {
        self.agent = (1, 1);
        self.has_key = false;
        self.steps = 0;
        self.done = false;
        self.render_obs()
    }

    pub fn reset_with_goal(&mut self, goal: (usize, usize)) -> Array1<f64> {
        self.goal = goal;
        self.reset()
    }

    pub fn step(&mut self, action: usize) -> (f64, Array1<f64>, bool) {
        if self.done { return (0.0, self.render_obs(), true); }
        self.steps += 1;

        let (dx, dy) = match action {
            0 => (0, -1),  // up
            1 => (0, 1),   // down
            2 => (-1, 0),  // left
            3 => (1, 0),   // right
            _ => (0, 0),
        };
        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;

        // Move if within bounds and not through a locked door
        if nx >= 0 && nx < self.w as isize && ny >= 0 && ny < self.h as isize {
            let (nxu, nyu) = (nx as usize, ny as usize);
            // Door blocks unless player has key
            if (nxu, nyu) == self.door && !self.has_key {
                // Can't pass through locked door
            } else {
                self.agent = (nxu, nyu);
            }
        }

        // Pick up key
        if self.agent == self.key {
            self.has_key = true;
        }

        // Open door (agent on door with key)
        if self.agent == self.door && self.has_key {
            self.door = (0, 0); // door removed
        }

        let done = self.steps >= self.max_steps || self.agent == self.goal;
        let reward = if self.agent == self.goal { 10.0 } else { -0.1 };

        (reward, self.render_obs(), done)
    }

    /// RGB observation: 7x7x3 flattened to 147D
    fn render_obs(&self) -> Array1<f64> {
        let mut obs = Array1::zeros(self.w * self.h * 3);
        for y in 0..self.h {
            for x in 0..self.w {
                let idx = (y * self.w + x) * 3;
                // Floor = brown [0.4, 0.2, 0.0]
                obs[idx] = 0.4;
                obs[idx+1] = 0.2;
                obs[idx+2] = 0.0;

                // Walls on border
                if x == 0 || y == 0 || x == self.w-1 || y == self.h-1 {
                    obs[idx] = 0.0; obs[idx+1] = 0.0; obs[idx+2] = 0.0;
                }
            }
        }
        // Agent = red
        let ai = (self.agent.1 * self.w + self.agent.0) * 3;
        obs[ai] = 1.0; obs[ai+1] = 0.0; obs[ai+2] = 0.0;

        // Key = blue
        let ki = (self.key.1 * self.w + self.key.0) * 3;
        if self.key != (0,0) {
            obs[ki] = 0.0; obs[ki+1] = 0.0; obs[ki+2] = 1.0;
        }

        // Door = green
        let di = (self.door.1 * self.w + self.door.0) * 3;
        if self.door != (0,0) {
            obs[di] = 0.0; obs[di+1] = 1.0; obs[di+2] = 0.0;
        }

        // Goal = yellow
        let gi = (self.goal.1 * self.w + self.goal.0) * 3;
        obs[gi] = 1.0; obs[gi+1] = 1.0; obs[gi+2] = 0.0;

        obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obs_shape() {
        let env = MiniGridEnv::new();
        let obs = env.render_obs();
        assert_eq!(obs.len(), 147);
    }

    #[test]
    fn test_basic_navigation() {
        let mut env = MiniGridEnv::new();
        let mut done = false;
        let mut steps = 0;
        while !done && steps < 100 {
            let (_, _, d) = env.step(1); // down
            done = d;
            steps += 1;
        }
        // Should not reach goal without key
        assert!(env.agent != env.goal || env.has_key);
    }
}
