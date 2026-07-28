/// ════════════════════════════════════════════════════════════════════════════
///  environment — Trait Environment + implémentation GridWorld 5×5
///
///  Interface unifiée pour tous les environnements (GridWorld, Minigrid, Habitat).
///  step(action) → (obs, reward, done).
///
///  Utilisation : Box<dyn Environment> dans TsoEngine (même pattern qu'Encoder).
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;

/// Résultat d'un step.
#[derive(Clone, Debug)]
pub struct StepResult {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub done: bool,
}

/// Interface universelle pour un environnement.
pub trait Environment: Send {
    /// Reset l'environnement, retourne l'observation initiale.
    fn reset(&mut self) -> Vec<f64>;

    /// Exécute une action, retourne (obs, reward, done).
    fn step(&mut self, action: usize) -> StepResult;

    /// Nombre d'actions possibles.
    fn action_space(&self) -> usize;

    /// Dimension de l'observation.
    fn observation_dim(&self) -> usize;
}

// ═══════════════════════════════════════════════════════════════════════════
//  GridWorld 5×5 (identique phase1b)
// ═══════════════════════════════════════════════════════════════════════════

const W: usize = 5;
const H: usize = 5;
const NA: usize = 4;
const PDIM: usize = 6;
const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

pub struct GridEnv {
    pub agent: (usize, usize),
    pub step_count: usize,
    pub done: bool,
}

impl GridEnv {
    pub fn new() -> Self {
        GridEnv { agent: (2, 2), step_count: 0, done: false }
    }

    fn perceive(&self) -> Vec<f64> {
        let (x, y) = self.agent;
        let ix = x as isize;
        let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0;
            let mut cx = ix + dx;
            let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < W as isize && cy < H as isize {
                d += 1;
                cx += dx;
                cy += dy;
            }
            d as f64 / (W.max(H) as f64)
        };
        let mut ws = 0.0;
        for &(wx, wy) in &WATER {
            let d = (((ix - wx as isize).abs().pow(2) + (iy - wy as isize).abs().pow(2)) as f64)
                .sqrt();
            if d <= 2.0 {
                ws = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }
        vec![ray(0, -1), ray(0, 1), ray(-1, 0), ray(1, 0), 0.0, ws]
    }
}

impl Environment for GridEnv {
    fn reset(&mut self) -> Vec<f64> {
        use rand::Rng;
        loop {
            let x = rand::thread_rng().r#gen_range(0..W);
            let y = rand::thread_rng().r#gen_range(0..H);
            if !WATER.contains(&(x, y)) {
                self.agent = (x, y);
                break;
            }
        }
        self.step_count = 0;
        self.done = false;
        self.perceive()
    }

    fn step(&mut self, action: usize) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.perceive(),
                reward: 0.0,
                done: true,
            };
        }
        self.step_count += 1;
        let (dx, dy) = match action {
            0 => (0, -1),
            1 => (0, 1),
            2 => (-1, 0),
            3 => (1, 0),
            _ => (0, 0),
        };
        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;
        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            self.done = self.step_count >= MAXS;
            return StepResult {
                observation: self.perceive(),
                reward: -0.5,
                done: self.done,
            };
        }
        self.agent = (nx as usize, ny as usize);
        if WATER.contains(&self.agent) {
            self.done = true;
            StepResult { observation: self.perceive(), reward: 10.0, done: true }
        } else if self.step_count >= MAXS {
            self.done = true;
            StepResult { observation: self.perceive(), reward: -1.0, done: true }
        } else {
            StepResult { observation: self.perceive(), reward: -0.02, done: false }
        }
    }

    fn action_space(&self) -> usize {
        NA
    }

    fn observation_dim(&self) -> usize {
        PDIM
    }
}
