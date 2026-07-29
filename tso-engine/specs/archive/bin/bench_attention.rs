/// ════════════════════════════════════════════════════════════════════════════
///  bench_attention — Benchmark avec/sans attention spatiale (e04s02)
///
///  Compare les performances du TSO complet sur l'environnement 5×5 Salle vide
///  avec et sans attention spatiale, sur N seeds. Utilise CognitiveConfig pour
///  n'activer/désactiver que l'attention, tous les autres sous-systèmes actifs.
///
///  Résultats : taux de succès exploitation ε=0 pour chaque config.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;

// ─── Environnement 5×5 (même que phase1b) ─────────────────────────────

const W: usize = 5;
const H: usize = 5;
const PERCEPTION_DIM: usize = 6;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;

const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool }

impl GridEnv5x5 {
    fn new() -> Self { GridEnv5x5 { agent: (2, 2), step: 0, done: false } }
    fn reset(&mut self, rng: &mut impl Rng) {
        loop {
            let x = rng.gen_range(0..W); let y = rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x, y)) { self.agent = (x, y); break; }
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
        for &(wx, wy) in &WATER_POSITIONS {
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
        if WATER_POSITIONS.contains(&self.agent) { self.done = true; return 10.0; }
        if self.step >= MAX_STEPS { self.done = true; return -1.0; }
        -0.02
    }
}

fn compute_bfs_potential() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W - 1) + (H - 1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W]; let mut q = VecDeque::new();
    for &(wx, wy) in &WATER_POSITIONS { dist[wx][wy] = Some(0); q.push_back((wx, wy)); }
    while let Some((cx, cy)) = q.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx; let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() { dist[nx][ny] = Some(d + 1); q.push_back((nx, ny)); }
            }
        }
    }
    for x in 0..W { for y in 0..H { pot[x][y] = match dist[x][y] { Some(d) => -2.5 * d as f64 / d_max, None => -2.5 }; } }
    pot
}

fn run_engine(use_attention: bool, seed: u64) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    let mut cogs = CognitiveConfig::default();
    cogs.attention = use_attention;
    engine.cogs = cogs;

    let bfs_pot = compute_bfs_potential();

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;
        run_one_ep(&mut engine, &bfs_pot, &mut rng);
    }

    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    let mut successes = 0usize;
    for _ in 0..TEST_EPS {
        let (_, succeeded) = run_one_ep(&mut engine, &bfs_pot, &mut rng);
        if succeeded { successes += 1; }
    }

    successes as f64 / TEST_EPS as f64 * 100.0
}

fn run_one_ep(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], rng: &mut impl Rng) -> (f64, bool) {
    let mut env = GridEnv5x5::new();
    env.reset(rng);
    engine.end_episode();

    let mut total_reward = 0.0;
    let mut succeeded = false;

    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a);
        total_reward += r;
        if env.done { succeeded = r > 0.0; break; }
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    (total_reward, succeeded)
}

fn main() {
    const N_SEEDS: usize = 10;
    let seeds: Vec<u64> = (0..N_SEEDS as u64).collect();

    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  BENCH ATTENTION — 5×5 vide, {N_SEEDS} seeds, δ-clip actif             ║");
    eprintln!("║  TSO complet (CognitiveConfig defaults) avec/sans attention          ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    for &use_attn in &[false, true] {
        let label = if use_attn { "Avec attention" } else { "Sans attention" };
        let mut scores = Vec::with_capacity(N_SEEDS);
        let t0 = Instant::now();

        for &seed in &seeds {
            let score = run_engine(use_attn, seed);
            scores.push(score);
        }

        let elapsed = t0.elapsed();
        let mean = scores.iter().sum::<f64>() / N_SEEDS as f64;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / N_SEEDS as f64;
        let std = var.sqrt();

        eprintln!("{:<20}  μ={:>7.1}%  σ={:>5.2}%  [{:.1?}]",
            label, mean, std, elapsed);
    }

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Si les deux configs sont >90% : l'attention ne dégrade pas          ║");
    eprintln!("║  Si sans attention <90% : l'attention est nécessaire à la stabilité  ║");
    eprintln!("║  Si avec attention plus haut : gain attentionnel mesuré              ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
