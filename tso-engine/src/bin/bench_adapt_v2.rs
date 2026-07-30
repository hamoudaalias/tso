//! bench_adapt_v2: but tournant agressif — switch tous les 10, 8 buts.
//! Usage: cargo run --release --bin bench_adapt_v2

use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use ndarray::Array1;

const W: usize = 5;
const H: usize = 5;
const SWITCH: usize = 10;
const MAX_STEPS: usize = 10;
const N_EP: usize = 200;
const N_PHASES: usize = N_EP / SWITCH; // 20

/// 8 goals, 4 aliasing pairs
const GOALS: [(usize, usize); 8] = [
    (4, 0), (0, 4),  // pair 1
    (4, 4), (0, 0),  // pair 2
    (2, 4), (4, 2),  // pair 3
    (0, 2), (2, 0),  // pair 4
];

struct FastEnv {
    agent: (usize, usize),
    goal_idx: usize,
    ep_in_phase: usize,
    phase: usize,
}

impl FastEnv {
    fn new() -> Self {
        FastEnv { agent: (0, 2), goal_idx: 0, ep_in_phase: 0, phase: 0 }
    }

    fn reset(&mut self) -> Array1<f64> {
        self.agent = (0, 2);
        if self.ep_in_phase >= SWITCH {
            self.ep_in_phase = 0;
            self.goal_idx = (self.goal_idx + 1) % GOALS.len();
            self.phase += 1;
        }
        self.ep_in_phase += 1;
        self.observation()
    }

    fn step(&mut self, action: usize) -> (f64, Array1<f64>, bool) {
        let (x, y) = self.agent;
        let (nx, ny) = match action {
            0 if y > 0 => (x, y - 1),
            1 if x < W - 1 => (x + 1, y),
            2 if y < H - 1 => (x, y + 1),
            3 if x > 0 => (x - 1, y),
            _ => (x, y),
        };
        self.agent = (nx, ny);
        let goal = GOALS[self.goal_idx];
        let done = self.ep_in_phase > MAX_STEPS || (nx, ny) == goal;
        let reward = if (nx, ny) == goal { 10.0 } else { -0.1 };
        (reward, self.observation(), done)
    }

    fn observation(&self) -> Array1<f64> {
        let (x, y) = self.agent;
        Array1::from_vec(vec![
            if y == 0 { 1.0 } else { 0.0 },
            if x == W - 1 { 1.0 } else { 0.0 },
            if y == H - 1 { 1.0 } else { 0.0 },
            if x == 0 { 1.0 } else { 0.0 },
        ])
    }

    fn current_goal(&self) -> (usize, usize) { GOALS[self.goal_idx] }
}

fn main() {
    let n_seeds = 30;

    println!("# Adaptation but tournant agressif\n");
    println!("Switch tous les {SWITCH}, 8 buts (4 paires), {N_PHASES} phases\n");

    for &(name, config_fn) in &[
        ("Linear AC", run_linear as fn() -> Vec<f64>),
        ("TSO attracteur", run_tso_attractor as fn() -> Vec<f64>),
        ("TSO full", run_tso_full as fn() -> Vec<f64>),
        ("TSO + épisodique", run_tso_episodic as fn() -> Vec<f64>),
    ] {
        // Aggregate phase metrics
        let mut phase_means: Vec<Vec<f64>> = vec![vec![]; N_PHASES];
        let mut intra_slopes: Vec<Vec<f64>> = vec![vec![]; N_PHASES];

        for _ in 0..n_seeds {
            let curve = config_fn();
            for ph in 0..N_PHASES {
                let start = ph * SWITCH;
                let end = start + SWITCH;
                let phase_rewards: Vec<f64> = (start..end.min(N_EP)).map(|e| curve[e]).collect();
                let pm = mean(&phase_rewards);
                phase_means[ph].push(pm);

                // Slope: mean of last 5 - mean of first 5
                if phase_rewards.len() >= 10 {
                    let first5 = mean(&phase_rewards[0..5]);
                    let last5 = mean(&phase_rewards[5..10]);
                    intra_slopes[ph].push(last5 - first5);
                }
            }
        }

        println!("## {name}\n");
        println!("| Phase | Goal | Reward moyen | Pente intra | Drop au switch |");
        println!("|-------|------|-------------|-------------|----------------|");

        for ph in 0..N_PHASES.min(12) { // first 12 phases
            let goal_idx = ph % GOALS.len();
            let goal = GOALS[goal_idx];
            let pm = mean(&phase_means[ph]);
            let slope = if intra_slopes[ph].is_empty() { 0.0 } else { mean(&intra_slopes[ph]) };
            println!("| {} | ({},{}) | {:.2} | {:.2} |", ph + 1, goal.0, goal.1, pm, slope);
        }
        println!();
    }
}

fn run_linear() -> Vec<f64> {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 0);
    let mut env = FastEnv::new();
    let mut curve = vec![0.0; N_EP];
    for ep in 0..N_EP {
        let mut obs = env.reset();
        let mut prev_r = 0.0;
        loop {
            let logits = cb.forward_logits(&obs);
            let action = if rand::random::<f64>() < cb.epsilon {
                rand::random::<usize>() % 4
            } else {
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap()
            };
            let (reward, next_obs, done) = env.step(action);
            cb.reinforce_td(prev_r, 0.99);
            cb.decay_trace(0.99, 0.98);
            cb.mark(&obs, action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        curve[ep] = prev_r;
    }
    curve
}

macro_rules! make_tso_runner {
    ($name:ident, $config:expr) => {
        fn $name() -> Vec<f64> {
            let mut eng = TsoEngine::new(4, 4);
            $config(&mut eng);
            let mut env = FastEnv::new();
            let mut curve = vec![0.0; N_EP];
            for ep in 0..N_EP {
                env.reset();
                let mut obs = env.observation();
                let mut prev_r = 0.0;
                loop {
                    let action = eng.step(&obs, prev_r, None, &[]);
                    let (reward, next_obs, done) = env.step(action);
                    obs = next_obs; prev_r = reward;
                    if done { break; }
                }
                curve[ep] = prev_r;
            }
            curve
        }
    };
}

fn cfg_attractor(e: &mut TsoEngine) {
    e.cogs.attractor = true; e.cogs.hypothalamus = false;
    e.cogs.episodic_curiosity = false; e.cogs.attention = false;
    e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
}

fn cfg_full(e: &mut TsoEngine) {
    e.cogs.attractor = true; e.cogs.hypothalamus = true;
    e.cogs.episodic_curiosity = true; e.cogs.attention = true;
    e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
}

fn cfg_episodic(e: &mut TsoEngine) {
    e.cogs.attractor = true; e.cogs.hypothalamus = false;
    e.cogs.episodic_curiosity = true; e.cogs.attention = false;
    e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
}

make_tso_runner!(run_tso_attractor, cfg_attractor);
make_tso_runner!(run_tso_full, cfg_full);
make_tso_runner!(run_tso_episodic, cfg_episodic);

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}
