/// ════════════════════════════════════════════════════════════════════════════
///  Phase 1b — Expérience B : TsoEngine complet sur 5×5
///
///  Objectif : isoler si le cycle cognitif TSO complet (attracteur, working
///  memory, graph conflict Φ, well_being à 9 termes) dégrade l'apprentissage
///  RL indépendamment de l'aliasing.
///
///  Cadran :
///                  5×5 (pas d'aliasing)      7×7 muré (aliasing)
///  Cervelet seul    #8 = 98%  ✅ (fait)       (Exp A à faire)
///  TSO complet      Exp B → ?                 (plus tard)
///
///  B1 : hypothalamus gelé (drift_rate=0 → deficits ~0, well_being ≈
///       R_ext + r_curiosity + shaping − ΔΦ + chronic_tension + parsimony
///       − metabolic_penalty)
///  B2 : hypothalamus normal (homeostasis complète, 9 termes du well_being)
///
///  Si B1 > 90% → le cycle cognitif est bénin, le 7×7 est un problème
///                 d'aliasing pur.
///  Si B1 < 90% → le cycle cognitif casse le RL même en 5×5 → chercher
///                 quel terme (Φ, attracteur, etc.) déstabilise le critic.
///  Si B2 < B1 → les termes homéostatiques (gate_reward, consummatory,
///                 deficit_penalty) sont la source de non-stationnarité.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

// ─── Environnement 5×5 compatible TSO ─────────────────────────────────────
// Perception : [wall_N, wall_S, wall_W, wall_E, food_sensed, water_sensed]
// Actions    : 0=N, 1=S, 2=W, 3=E
const W: usize = 5;
const H: usize = 5;
const PERCEPTION_DIM: usize = 6; // 4 whiskers + food + water
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;

// Ressources : uniquement de l'eau (pas de nourriture)
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

struct GridEnv5x5 {
    agent: (usize, usize),
    step: usize,
    done: bool,
}

impl GridEnv5x5 {
    fn new() -> Self {
        GridEnv5x5 { agent: (2, 2), step: 0, done: false }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..W);
            let y = rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x, y)) && !FOOD_POSITIONS.contains(&(x, y)) {
                self.agent = (x, y);
                break;
            }
        }
        self.step = 0;
        self.done = false;
    }

    /// Perception TSO-compatible : [wall_N, wall_S, wall_W, wall_E, food_sensed, water_sensed]
    fn perceive(&self) -> Array1<f64> {
        let (x, y) = self.agent;
        let ix = x as isize;
        let iy = y as isize;

        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0usize;
            let mut cx = ix + dx;
            let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < W as isize && cy < H as isize {
                d += 1;
                cx += dx;
                cy += dy;
            }
            d as f64 / (W.max(H) as f64)
        };

        // Food & water proximity detection (dans les 2 cases)
        let mut food_sensed = 0.0;
        for &(fx, fy) in &FOOD_POSITIONS {
            let dx = (ix - fx as isize).abs() as f64;
            let dy = (iy - fy as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 2.0 {
                food_sensed = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }

        let mut water_sensed = 0.0;
        for &(wx, wy) in &WATER_POSITIONS {
            let dx = (ix - wx as isize).abs() as f64;
            let dy = (iy - wy as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 2.0 {
                water_sensed = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }

        Array1::from_vec(vec![
            ray(0, -1), // N
            ray(0, 1),  // S
            ray(-1, 0), // W
            ray(1, 0),  // E
            food_sensed,
            water_sensed,
        ])
    }

    fn step_env(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.step += 1;

        let (dx, dy) = match action {
            0 => (0isize, -1isize), // N
            1 => (0, 1),            // S
            2 => (-1, 0),           // W
            3 => (1, 0),            // E
            _ => (0, 0),
        };

        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;

        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            if self.step >= MAX_STEPS { self.done = true; }
            return -0.5;
        }

        self.agent = (nx as usize, ny as usize);

        // Eau → récompense + terminal
        if WATER_POSITIONS.contains(&self.agent) {
            self.done = true;
            return 10.0;
        }
        // Nourriture → pas de nourriture dans cette config
        if FOOD_POSITIONS.contains(&self.agent) {
            self.done = true;
            return 10.0;
        }

        if self.step >= MAX_STEPS {
            self.done = true;
            return -1.0;
        }

        -0.02
    }
}

// ─── Experiment B runner ────────────────────────────────────────────────────

struct ConfigB {
    label: &'static str,
    hidden_dim: usize,
    /// Si true, on gèle l'hypothalamus : drift_rate = 0, pas de dérive
    freeze_hypothalamus: bool,
    /// Si true, on réinitialise les deficits à 1.0 avant chaque step
    reset_homeostasis: bool,
}

fn run_experiment_b(cfg: &ConfigB) {
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, cfg.hidden_dim);

    // Config du cervelet
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = false;

    // Désactiver le sommeil (on veut un apprentissage continu)
    engine.sleep_every_n_episodes = 0;

    // Seuil d'élimination des concepts zombies généreux (500 steps = jamais en 5×5)
    // Mais on le garde actif pour éviter l'accumulation

    // BFS potential map pour initialiser les nouveaux concepts
    let bfs_pot = compute_potential_map();

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;

    let t0 = Instant::now();
    let mut train_rewards: Vec<f64> = Vec::with_capacity(TRAIN_EPS);
    let mut train_success: Vec<bool> = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;

        let (total, succeeded) = run_ep(ep as u64, &mut engine, &bfs_pot, cfg);
        train_rewards.push(total);
        train_success.push(succeeded);
    }

    let elapsed = t0.elapsed();
    let train_avg: f64 = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_200: f64 = train_rewards[TRAIN_EPS - 200..].iter().sum::<f64>() / 200.0;
    let train_success_rate = train_success.iter().filter(|&&s| s).count() as f64 / TRAIN_EPS as f64;

    // TEST — ε=0, noise_std=0
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    let mut test_rewards: Vec<f64> = Vec::with_capacity(TEST_EPS);
    let mut test_success: Vec<bool> = Vec::with_capacity(TEST_EPS);

    for ep in 0..TEST_EPS {
        let (total, succeeded) = run_ep(1000 + ep as u64, &mut engine, &bfs_pot, cfg);
        test_rewards.push(total);
        test_success.push(succeeded);
    }

    let test_avg: f64 = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s| s).count() as f64 / TEST_EPS as f64;
    let debug_sample: Vec<f64> = test_rewards.iter().take(10).copied().collect();

    // État final du TSO
    let concepts = engine.num_concepts();
    let edges = engine.graph_edges();
    let phi = engine.current_phi;
    let replay_len = engine.cerebellum.replay.len();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<66} ║", cfg.label);
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  hd={}  freeze_hypothal={}  reset_homeo={}",
        cfg.hidden_dim, cfg.freeze_hypothalamus, cfg.reset_homeostasis);
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last200={:>7.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_200,
        train_success_rate * 100.0);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  success={:.1}%",
        TEST_EPS, test_avg, test_success_rate * 100.0);
    eprintln!("║  Concepts={}  Edges={}  Φ={:.3}  Replay={}",
        concepts, edges, phi, replay_len);
    eprintln!("║  10 premiers tests: {:?}", debug_sample);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

/// Precompute BFS potential for concept initialization
/// Φ(s) = -2.5 × d_nearest_water / d_max
fn compute_potential_map() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W - 1) + (H - 1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];

    let mut dist = vec![vec![None::<usize>; H]; W];
    let mut queue = VecDeque::new();
    for &(wx, wy) in &WATER_POSITIONS {
        dist[wx][wy] = Some(0);
        queue.push_back((wx, wy));
    }
    while let Some((cx, cy)) = queue.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0isize,1),(0,-1),(1,0),(-1,0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() {
                    dist[nx][ny] = Some(d + 1);
                    queue.push_back((nx, ny));
                }
            }
        }
    }
    for x in 0..W {
        for y in 0..H {
            pot[x][y] = match dist[x][y] {
                Some(d) => -2.5 * d as f64 / d_max,
                None => -2.5,
            };
        }
    }
    pot
}

fn run_ep(_seed: u64, engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], cfg: &ConfigB) -> (f64, bool) {
    let mut env = GridEnv5x5::new();
    env.reset();
    engine.end_episode(); // reset les traces

    if cfg.freeze_hypothalamus {
        // Force la satiété : pas de déficit → gate_reward ≈ R_ext, consummatory ≈ 0
        engine.hypothalamus.energy = 1.0;
        engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5;
        engine.hypothalamus.sleep_debt = 0.0;
    }

    let mut total_reward = 0.0;
    let mut succeeded = false;

    let p = env.perceive();
    // BFS value for concept initialization
    let bfs_val = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bfs_val, &[]);

    while !env.done {
        let r = env.step_env(a);
        total_reward += r;
        if env.done {
            succeeded = r > 0.0;
            let pt = env.perceive();
            let bfs_val = Some(bfs_pot[env.agent.0][env.agent.1]);
            engine.step(&pt, r, bfs_val, &[]);
            break;
        }
        // Si freeze, reset homeostasis après chaque step (contre la dérive interne)
        if cfg.freeze_hypothalamus || cfg.reset_homeostasis {
            engine.hypothalamus.energy = 1.0;
            engine.hypothalamus.hydration = 1.0;
            engine.hypothalamus.temperature = 0.5;
            engine.hypothalamus.sleep_debt = 0.0;
        }
        let pt = env.perceive();
        let bfs_val = Some(bfs_pot[env.agent.0][env.agent.1]);
        a = engine.step(&pt, r, bfs_val, &[]);
    }
    engine.end_episode();

    if !cfg.freeze_hypothalamus && engine.cerebellum.replay.len() >= 64 {
        engine.cerebellum.replay_train(64, 0.95, 10);
    }

    (total_reward, succeeded)
}

fn main() {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PHASE 1b — Expérience B : TsoEngine complet sur 5×5                ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Questions :                                                          ║");
    eprintln!("║  B1 : Le cycle cognitif TSO (attracteur, graph, Φ, well_being 9       ║");
    eprintln!("║       termes) casse-t-il le RL même sans aliasing ?                   ║");
    eprintln!("║  B2 : Les termes homéostatiques (gate_reward, consummatory,           ║");
    eprintln!("║       deficit_penalty) sont-ils la source de non-stationnarité ?      ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Prédiction B1 : >90% → cycle cognitif bénin sans aliasing            ║");
    eprintln!("║  Prédiction B2 : < B1 → homéostasie déstabilise le critic             ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── B0 : TSO complet avec hypothalamus gelé (référence, doit ~98%) ──
    run_experiment_b(&ConfigB {
        label: "B0. hd=4, TSO gelé (référence)",
        hidden_dim: 4,
        freeze_hypothalamus: true,
        reset_homeostasis: false,
    });

    // ── B1 : TSO complet, hypothalamus qui dérive (homéostasie active) ──
    run_experiment_b(&ConfigB {
        label: "B1. hd=4, TSO dérive normale",
        hidden_dim: 4,
        freeze_hypothalamus: false,
        reset_homeostasis: false,
    });

    // ── B2 : TSO complet, reset homeostasis chaque step ──
    run_experiment_b(&ConfigB {
        label: "B2. hd=4, TSO + reset chaque step",
        hidden_dim: 4,
        freeze_hypothalamus: false,
        reset_homeostasis: true,
    });

    // ── B3a : hd=16, TSO gelé ──
    run_experiment_b(&ConfigB {
        label: "B3a. hd=16, TSO gelé",
        hidden_dim: 16,
        freeze_hypothalamus: true,
        reset_homeostasis: false,
    });

    // ── B3b : hd=16, TSO dérive normale ──
    run_experiment_b(&ConfigB {
        label: "B3b. hd=16, TSO dérive normale",
        hidden_dim: 16,
        freeze_hypothalamus: false,
        reset_homeostasis: false,
    });

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  FIN DE PHASE 1b                                                      ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Si B0/B1 > 90% : cycle cognitif bénin, le 7×7 est aliasing pur.      ║");
    eprintln!("║  Si B2a << B1   : homéostasie déstabilise → stocker R_ext dans replay ║");
    eprintln!("║  Si B0/B1 < 90% : attracteur/graph/Φ cassent le RL → diagnostic Φ    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
