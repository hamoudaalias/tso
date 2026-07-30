//! bench_adapt_minigrid: but tournant sur MiniGrid 7×7 (147D).
//! Switch tous les N épisodes, 6 goals distincts.
//! Usage: cargo run --release --bin bench_adapt_minigrid

use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use ndarray::Array1;

const SWITCH: usize = 20;
const N_EP: usize = 240; // 12 phases × 20 épisodes
const W: usize = 7;
const H: usize = 7;

/// 6 goals distincts (visuellement différents sur 147D)
const GOALS: [(usize, usize); 6] = [
    (5, 5), (1, 5), (5, 1), (1, 1), (3, 5), (5, 3),
];

struct RotatingMiniGrid {
    env: MiniGridEnv,
    goal_idx: usize,
    ep_in_phase: usize,
}

impl RotatingMiniGrid {
    fn new() -> Self {
        RotatingMiniGrid { env: MiniGridEnv::new(), goal_idx: 0, ep_in_phase: 0 }
    }

    fn reset(&mut self) -> Array1<f64> {
        if self.ep_in_phase >= SWITCH {
            self.ep_in_phase = 0;
            self.goal_idx = (self.goal_idx + 1) % GOALS.len();
        }
        self.ep_in_phase += 1;
        self.env.reset_with_goal(GOALS[self.goal_idx])
    }

    fn step(&mut self, action: usize) -> (f64, Array1<f64>, bool) {
        self.env.step(action)
    }
}

fn main() {
    let n_seeds = 20;
    let dim = W * H * 3;
    let na = 7;

    for &(name, config_fn) in &[
        ("Linear AC", run_linear as fn(usize, usize) -> Vec<f64>),
        ("TSO attracteur", run_tso_attractor as fn(usize, usize) -> Vec<f64>),
        ("TSO full", run_tso_full as fn(usize, usize) -> Vec<f64>),
    ] {
        let mut phase_success = vec![0; N_EP / SWITCH];
        let mut phase_avg = vec![0.0; N_EP / SWITCH];

        for _ in 0..n_seeds {
            let curve = config_fn(dim, na);
            for ph in 0..(N_EP / SWITCH) {
                let start = ph * SWITCH;
                let phase_r: Vec<f64> = (start..start + SWITCH).map(|e| curve[e]).collect();
                let pm = mean(&phase_r);
                phase_avg[ph] += pm / n_seeds as f64;
                // Success = last 5 episodes all have reward > 0
                if phase_r.len() >= 5 {
                    let last5 = &phase_r[phase_r.len() - 5..];
                    if last5.iter().all(|&r| r > 0.0) { phase_success[ph] += 1; }
                }
            }
        }

        println!("## {name}\n");
        println!("| Phase | Goal | Reward moyen | Succès (≥5/20 seeds) |");
        println!("|-------|------|-------------|---------------------|");
        for ph in 0..(N_EP / SWITCH) {
            let g = GOALS[ph % GOALS.len()];
            let ok = if phase_success[ph] >= 5 { "✅" } else { "❌" };
            println!("| {} | ({},{}) | {:.2} | {}/{} {}", ph + 1, g.0, g.1, phase_avg[ph],
                phase_success[ph], n_seeds, ok);
        }
        println!("\nTaux de succès global: {:.0}%\n",
            100.0 * phase_success.iter().filter(|&&s| s >= 5).count() as f64 / (N_EP / SWITCH) as f64);
    }
}

fn run_linear(dim: usize, na: usize) -> Vec<f64> {
    let mut cb = Cerebellum::new(dim, na, 0.01, 0.3, 0.1, 0);
    let mut rmg = RotatingMiniGrid::new();
    let mut curve = vec![0.0; N_EP];
    for ep in 0..N_EP {
        let mut obs = rmg.reset();
        let mut prev_r = 0.0;
        loop {
            let logits = cb.forward_logits(&obs);
            let action = if rand::random::<f64>() < cb.epsilon {
                rand::random::<usize>() % na
            } else {
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap()
            };
            let (reward, next_obs, done) = rmg.step(action);
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

fn run_tso(dim: usize, na: usize, cfg: fn(&mut TsoEngine)) -> Vec<f64> {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    cfg(&mut eng);
    let mut rmg = RotatingMiniGrid::new();
    let mut curve = vec![0.0; N_EP];
    for ep in 0..N_EP {
        let mut obs = rmg.reset();
        eng.end_episode();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rmg.step(action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        curve[ep] = prev_r;
    }
    curve
}

fn run_tso_attractor(dim: usize, na: usize) -> Vec<f64> {
    run_tso(dim, na, |e| {
        e.cogs.attractor = true; e.cogs.hypothalamus = false;
        e.cogs.episodic_curiosity = false; e.cogs.attention = false;
        e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
    })
}

fn run_tso_full(dim: usize, na: usize) -> Vec<f64> {
    run_tso(dim, na, |e| {
        e.cogs.attractor = true; e.cogs.hypothalamus = true;
        e.cogs.episodic_curiosity = true; e.cogs.attention = true;
        e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
    })
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}
