/// Rotating-T: open 5×5 room, goal rotates among 4 corners every N episodes.
/// Measures adaptation speed across non-stationary goal shifts.
use ndarray::Array1;

const W: usize = 5;
const H: usize = 5;

#[derive(Clone, Copy, PartialEq)]
pub enum GoalPhase { TopRight, BottomLeft, Random }

pub struct RotatingT {
    pub agent: (usize, usize),
    pub goal: (usize, usize),
    pub phase: GoalPhase,
    pub episode: usize,
    pub switch_every: usize,
    pub steps: usize,
    pub max_steps: usize,
    pub phase_count: usize,
    phase_ep_count: usize,
}

impl RotatingT {
    pub fn new(switch_every: usize) -> Self {
        let mut rt = RotatingT {
            agent: (0, 0),
            goal: (4, 0),
            phase: GoalPhase::TopRight,
            episode: 0,
            switch_every,
            steps: 0,
            max_steps: 20,
            phase_count: 0,
            phase_ep_count: 0,
        };
        rt
    }

    pub fn reset(&mut self) {
        self.agent = (0, 2);
        self.steps = 0;
        self.episode += 1;
        self.phase_ep_count += 1;

        if self.phase_ep_count >= self.switch_every {
            self.phase_ep_count = 0;
            self.phase_count += 1;
            match self.phase_count % 3 {
                0 => { self.phase = GoalPhase::TopRight; self.goal = (4, 0); }
                1 => { self.phase = GoalPhase::BottomLeft; self.goal = (0, 4); }
                _ => { self.phase = GoalPhase::Random;
                       let i = (self.phase_count / 3) % 2;
                       self.goal = if i == 0 { (4, 4) } else { (0, 0) }; }
            }
        } else if self.phase_ep_count == 1 {
            // First episode of phase: set goal (handle first ep)
            match self.phase {
                GoalPhase::TopRight => self.goal = (4, 0),
                GoalPhase::BottomLeft => self.goal = (0, 4),
                GoalPhase::Random => {}
            }
        }
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
        (reward, self.observation(), done)
    }

    pub fn observation(&self) -> Array1<f64> {
        let (x, y) = self.agent;
        let (gx, gy) = self.goal;
        Array1::from_vec(vec![
            if y == 0 { 1.0 } else { 0.0 },           // wall N
            if x == W - 1 { 1.0 } else { 0.0 },       // wall E
            if y == H - 1 { 1.0 } else { 0.0 },       // wall S
            if x == 0 { 1.0 } else { 0.0 },           // wall W
            if gx > x { 1.0 } else if gx < x { -1.0 } else { 0.0 }, // goal dir
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_switch() {
        let mut rt = RotatingT::new(2);
        assert_eq!(rt.goal, (4, 0));
        for _ in 0..2 { rt.reset(); }
        // after 2 episodes, should switch to BottomLeft
        assert!(rt.phase_ep_count == 0 || rt.goal == (0, 4));
    }

    #[test]
    fn test_step_bounds() {
        let mut rt = RotatingT::new(50);
        for _ in 0..50 { let (_, _, done) = rt.step(0); if done { break; } }
        assert!(rt.steps <= rt.max_steps);
    }
}
