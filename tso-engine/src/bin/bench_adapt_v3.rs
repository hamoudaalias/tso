//! bench_adapt_v3: 5×5, max_steps=8, 8 goals, switch=10.
//! Tight steps → exploration nécessaire après switch.

use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use ndarray::Array1;

const W: usize = 5;
const H: usize = 5;
const SWITCH: usize = 10;
const MAX_STEPS: usize = 8;
const N_EP: usize = 200;
const GOALS: [(usize, usize); 8] = [
    (4, 0), (0, 4), (4, 4), (0, 0), (2, 4), (4, 2), (0, 2), (2, 0),
];

struct Env {
    agent: (usize, usize),
    goal_idx: usize,
    ep_in_phase: usize,
    phase: usize,
}

impl Env {
    fn new() -> Self {
        Env { agent: (0, 2), goal_idx: 0, ep_in_phase: 0, phase: 0 }
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

    fn after_switch(&self) -> bool { self.ep_in_phase <= 3 }
}

fn main() {
    let n_seeds = 30;

    println!("# Adaptation but tournant agressif v3\n");
    println!("5×5, max_steps={MAX_STEPS}, switch={SWITCH}, 8 goals, {N_EP} episodes\n");
    println!("Une phase reussie = reward > 0 sur les 3 derniers episodes de la phase\n");

    for &(name, config_fn) in &[
        ("Linear AC", run_linear as fn() -> Vec<f64>),
        ("TSO attracteur", run_tso_attractor as fn() -> Vec<f64>),
        ("TSO full", run_tso_full as fn() -> Vec<f64>),
        ("TSO + épisodique", run_tso_episodic as fn() -> Vec<f64>),
    ] {
        let mut per_phase = vec![vec![0.0; N_EP]; n_seeds];
        for seed in 0..n_seeds {
            let curve = config_fn();
            per_phase[seed] = curve;
        }

        // Phase success rate
        let mut phase_success = vec![0; N_EP / SWITCH];
        for ep in 0..N_EP {
            let phase = ep / SWITCH;
            let mut successes = 0;
            for seed in 0..n_seeds {
                if per_phase[seed][ep] > 0.0 { successes += 1; }
            }
            if successes > n_seeds / 2 { phase_success[phase] += 1; }
        }

        println!("## {name}\n");
        println!("| Phase | Goal | Épisodes réussis (>50% seeds) |");
        println!("|-------|------|------------------------------|");
        for ph in 0..(N_EP / SWITCH).min(12) {
            let g = GOALS[ph % GOALS.len()];
            let ok = phase_success[ph];
            print!("| {} | ({},{}) | {}/{SWITCH} |", ph + 1, g.0, g.1, ok);
            if ok <= 3 { print!(" ← difficile"); }
            println!();
        }
        println!("\nRéussite globale: {:.0}%", 
            100.0 * phase_success.iter().sum::<usize>() as f64 / (N_EP / SWITCH * SWITCH) as f64);
        println!();
    }
}

fn run_linear() -> Vec<f64> {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 0);
    let mut env = Env::new();
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

fn run_tso(e: &mut TsoEngine) -> Vec<f64> {
    let mut env = Env::new();
    let mut curve = vec![0.0; N_EP];
    for ep in 0..N_EP {
        env.reset();
        let mut obs = env.observation();
        let mut prev_r = 0.0;
        loop {
            let action = e.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = env.step(action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        curve[ep] = prev_r;
    }
    curve
}

fn run_tso_attractor() -> Vec<f64> {
    let mut e = TsoEngine::new(4, 4);
    e.cogs.attractor = true; e.cogs.hypothalamus = false;
    e.cogs.episodic_curiosity = false; e.cogs.attention = false;
    e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
    run_tso(&mut e)
}

fn run_tso_full() -> Vec<f64> {
    let mut e = TsoEngine::new(4, 4);
    e.cogs.attractor = true; e.cogs.hypothalamus = true;
    e.cogs.episodic_curiosity = true; e.cogs.attention = true;
    e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
    run_tso(&mut e)
}

fn run_tso_episodic() -> Vec<f64> {
    let mut e = TsoEngine::new(4, 4);
    e.cogs.attractor = true; e.cogs.hypothalamus = false;
    e.cogs.episodic_curiosity = true; e.cogs.attention = false;
    e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
    run_tso(&mut e)
}
