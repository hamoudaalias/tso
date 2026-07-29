/// ════════════════════════════════════════════════════════════════════════════
///  multi_seed_bisect — Matrice 8 configs × N seeds sur 5×5
///
///  Bisse le cycle cognitif TSO pour isoler l'interférence avec l'apprentissage
///  du Cerebellum (cf. BUG-2025-08-03T120000).
///
///  Configs (ordre de bissection) :
///    0 : Cerebellum seul, signal propre (référence)
///    1 : +δ-clip
///    2 : +attractor (concepts)
///    3 : +graph/Φ
///    4 : +episodic/attention/curiosity
///    5 : +metabolic_cost
///    6 : +hypothalamus
///    7 : TSO complet (tous les sous-systèmes)
///
///  Résultat : moyenne ± écart-type sur N seeds, succès exploitation ε=0.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;

// ─── Environnement 5×5 (identique à phase1b_fix3) ─────────────────────────

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

fn compute_potential_map() -> Vec<Vec<f64>> {
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

// ─── Configuration de l'expérience ─────────────────────────────────────────

struct ExpConfig {
    label: &'static str,
    build_cogs: fn() -> CognitiveConfig,
}

fn run_seed(engine_cfg: &CognitiveConfig, seed: u64) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0; // pas de replay pour isoler le TD online
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;
    engine.cogs = engine_cfg.clone();

    let bfs_pot = compute_potential_map();

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;

    // Train
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;
        run_one_ep(&mut engine, &bfs_pot, &mut rng);
    }

    // Test ε=0
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    let mut successes = 0usize;
    for _ep in 0..TEST_EPS {
        let (_, succeeded) = run_one_ep(&mut engine, &bfs_pot, &mut rng);
        if succeeded { successes += 1; }
    }

    successes as f64 / TEST_EPS as f64 * 100.0
}

fn run_one_ep(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], rng: &mut impl Rng) -> (f64, bool) {
    let mut env = GridEnv5x5::new();
    env.reset(rng);
    engine.end_episode();

    // Freeze hypothalamus dans tous les cas (isole l'interférence non-homéostatique)
    engine.hypothalamus.energy = 1.0;
    engine.hypothalamus.hydration = 1.0;
    engine.hypothalamus.temperature = 0.5;
    engine.hypothalamus.sleep_debt = 0.0;

    let mut total_reward = 0.0;
    let mut succeeded = false;

    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a);
        total_reward += r;
        if env.done { succeeded = r > 0.0; break; }
        // Recongele l'hypothalamus chaque step
        engine.hypothalamus.energy = 1.0;
        engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5;
        engine.hypothalamus.sleep_debt = 0.0;
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    (total_reward, succeeded)
}

// ─── Config builders (ordre de bissection) ─────────────────────────────────

fn cfg_cerebellum_only() -> CognitiveConfig {
    CognitiveConfig {
        attractor: false,
        graph_phi: false,
        attention: false,
        episodic_curiosity: false,
        metabolic_cost: false,
        hypothalamus: false,
        delta_clip_max: 0.0, // pas de clip (comportement original Phase 1 #8)
        ..CognitiveConfig::default()
    }
}

fn cfg_delta_clip() -> CognitiveConfig {
    let mut c = cfg_cerebellum_only();
    c.delta_clip_max = 5.0; // clip de δ à 5.0
    c
}

fn cfg_attractor() -> CognitiveConfig {
    let mut c = cfg_delta_clip();
    c.attractor = true;
    c
}

fn cfg_graph_phi() -> CognitiveConfig {
    let mut c = cfg_attractor();
    c.graph_phi = true;
    c
}

fn cfg_episodic() -> CognitiveConfig {
    let mut c = cfg_graph_phi();
    c.episodic_curiosity = true;
    c.attention = true;
    c
}

fn cfg_metabolic() -> CognitiveConfig {
    let mut c = cfg_episodic();
    c.metabolic_cost = true;
    c
}

fn cfg_hypothalamus() -> CognitiveConfig {
    let mut c = cfg_metabolic();
    c.hypothalamus = true;
    c
}

fn cfg_tso_complete() -> CognitiveConfig {
    CognitiveConfig::default() // tout-à-true (comportement actuel)
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    const N_SEEDS: usize = 10;
    let seeds: Vec<u64> = (0..N_SEEDS as u64).collect();

    let configs: [ExpConfig; 8] = [
        ExpConfig { label: "0  Cerebellum seul (référence)",     build_cogs: cfg_cerebellum_only },
        ExpConfig { label: "1  +δ-clip (5.0)",                   build_cogs: cfg_delta_clip },
        ExpConfig { label: "2  +attractor (concepts)",            build_cogs: cfg_attractor },
        ExpConfig { label: "3  +graph/Φ",                         build_cogs: cfg_graph_phi },
        ExpConfig { label: "4  +episodic/attention/curiosity",    build_cogs: cfg_episodic },
        ExpConfig { label: "5  +metabolic_cost",                  build_cogs: cfg_metabolic },
        ExpConfig { label: "6  +hypothalamus",                    build_cogs: cfg_hypothalamus },
        ExpConfig { label: "7  TSO complet (tout-à-true)",        build_cogs: cfg_tso_complete },
    ];

    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  MATRICE DE BISSECTION — {N_SEEDS} seeds × 8 configs                    ║");
    eprintln!("║  Environnement : 5×5, ε=0 test, hypothalamus gelé                    ║");
    eprintln!("║  RL signal : perception brute + reward ext + γ·Φ_BFS (stationnaire)  ║");
    eprintln!("║  Pas de replay (isole le TD online)                                   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("{:<45}  {:>8}  {:>8}", "Config", "Moyenne", "Écart-type");
    eprintln!("{:-<45}  {:-<8}  {:-<8}", "", "", "");

    for cfg in &configs {
        let mut scores = Vec::with_capacity(N_SEEDS);
        let t0 = Instant::now();

        for &seed in &seeds {
            let score = run_seed(&(cfg.build_cogs)(), seed);
            scores.push(score);
        }

        let elapsed = t0.elapsed();
        let mean = scores.iter().sum::<f64>() / N_SEEDS as f64;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / N_SEEDS as f64;
        let std = var.sqrt();

        eprintln!("{:<45}  {:>7.1}%  {:>6.2}%  [{:.1?}]",
            cfg.label, mean, std, elapsed);
        if let Some(step) = cfg.label.split(' ').next() {
            if mean < 80.0 && step.parse::<usize>().unwrap_or(99) > 0 {
                eprintln!("  ⚠  CHUTE DÉTECTÉE à l'étape {} (moyenne={:.1}%)", step, mean);
            }
        }
    }

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  LÉGENDE                                                             ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Si config 0 < 80% : le TD online est instable même sans cycle       ║");
    eprintln!("║  Si config 1 > 80% : le δ-clip suffit à stabiliser                   ║");
    eprintln!("║  Si config 1 < 80% mais config k > 80% : le cycle cognitif aide      ║");
    eprintln!("║  Si config k < 80% et config k+1 chute : le sous-système k+1         ║");
    eprintln!("║    est le responsable principal de l'interférence                    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
