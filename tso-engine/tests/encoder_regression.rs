//! Test de régression : TSO avec Encoder trait sur GridWorld 5×5.
//!
//! Vérifie que l'intégration du trait Encoder (AttractorEncoder) ne casse
//! pas les performances historiques : le TSO avec δ-clip + signal stationnaire
//! doit atteindre >90% en exploitation pure.
//!
//! Réplique les conditions de multi_seed_bisect (config 1 / config 2).

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::encoder::AttractorEncoder;
use tso_engine::CognitiveConfig;

// ─── GridWorld 5×5 ─────────────────────────────────────────────────────

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

struct Env5 { agent: (usize, usize), step: usize, done: bool }
impl Env5 {
    fn new() -> Self { Env5 { agent: (2, 2), step: 0, done: false } }
    fn reset(&mut self, rng: &mut impl Rng) {
        loop {
            let x = rng.gen_range(0..W); let y = rng.gen_range(0..H);
            if !WATER.contains(&(x, y)) { self.agent = (x, y); break; }
        }
        self.step = 0; self.done = false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x, y) = self.agent; let ix = x as isize; let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0; let mut cx = ix + dx; let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < W as isize && cy < H as isize { d += 1; cx += dx; cy += dy; }
            d as f64 / (W.max(H) as f64)
        };
        let mut ws = 0.0;
        for &(wx, wy) in &WATER {
            let d = (((ix - wx as isize).abs().pow(2) + (iy - wy as isize).abs().pow(2)) as f64).sqrt();
            if d <= 2.0 { ws = (1.0 - d / 3.0).max(0.0); break; }
        }
        Array1::from_vec(vec![ray(0, -1), ray(0, 1), ray(-1, 0), ray(1, 0), 0.0, ws])
    }
    fn step_env(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.step += 1;
        let (dx, dy) = match action { 0 => (0, -1), 1 => (0, 1), 2 => (-1, 0), 3 => (1, 0), _ => (0, 0) };
        let nx = self.agent.0 as isize + dx; let ny = self.agent.1 as isize + dy;
        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            if self.step >= MAX_STEPS { self.done = true; } return -0.5;
        }
        self.agent = (nx as usize, ny as usize);
        if WATER.contains(&self.agent) { self.done = true; return 10.0; }
        if self.step >= MAX_STEPS { self.done = true; return -1.0; }
        -0.02
    }
}

fn compute_bfs() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W-1)+(H-1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W]; let mut q = VecDeque::new();
    for &(wx, wy) in &WATER { dist[wx][wy] = Some(0); q.push_back((wx, wy)); }
    while let Some((cx, cy)) = q.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx; let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() { dist[nx][ny] = Some(d+1); q.push_back((nx, ny)); }
            }
        }
    }
    for x in 0..W { for y in 0..H { pot[x][y] = match dist[x][y] { Some(d) => -2.5*d as f64/d_max, None => -2.5 }; } }
    pot
}

fn run_test(use_encoder: bool, seed: u64) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;
    engine.cogs = CognitiveConfig::default(); // attractor enabled, δ-clip default

    if use_encoder {
        engine.encoder = Some(Box::new(AttractorEncoder::new(PERCEPTION_DIM)));
    }

    let bfs = compute_bfs();

    const TRAIN: usize = 500;
    const TEST: usize = 100;

    for ep in 1..=TRAIN {
        let remain = (TRAIN - ep).max(0) as f64 / TRAIN as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;
        run_one_ep(&mut engine, &bfs, &mut rng);
    }

    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    let mut ok = 0usize;
    for _ in 0..TEST {
        let (_, s) = run_one_ep(&mut engine, &bfs, &mut rng);
        if s { ok += 1; }
    }
    ok as f64 / TEST as f64 * 100.0
}

fn run_one_ep(engine: &mut TsoEngine, bfs: &[Vec<f64>], rng: &mut impl Rng) -> (f64, bool) {
    let mut env = Env5::new();
    env.reset(rng);
    engine.end_episode();

    let mut total = 0.0;
    let mut succeeded = false;
    let p = env.perceive();
    let bv = Some(bfs[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a);
        total += r;
        if env.done {
            succeeded = r > 0.0;
            let pt = env.perceive();
            engine.step(&pt, r, Some(bfs[env.agent.0][env.agent.1]), &[]);
            break;
        }
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    (total, succeeded)
}

#[test]
fn test_encoder_5x5() {
    let score = run_test(true, 42);
    eprintln!("AttractorEncoder 5×5 (ε=0): {:.1}%", score);
    assert!(score >= 90.0, "Encoder regression: {:.1}% < 90%", score);
}

#[test]
fn test_fallback_5x5() {
    let score = run_test(false, 42);
    eprintln!("AttractorField direct 5×5 (ε=0): {:.1}%", score);
    assert!(score >= 90.0, "Fallback regression: {:.1}% < 90%", score);
}
