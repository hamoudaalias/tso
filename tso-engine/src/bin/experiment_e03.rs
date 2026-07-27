/// ════════════════════════════════════════════════════════════════════════════
///  e03s04 : Expérience comparative — Cervelet seul vs TSO complet
///
///  But : valider que le well-being stationnaire (correctifs e03) résout le
///  problème #1 : TSO complet passe de ~20% → ~90% en exploitation pure.
///
///  Configurations :
///    A — use_stationary_reward = false  (original, well-being 9-termes)
///    B — use_stationary_reward = true   (correctifs e03 : reward stationnaire)
///
///  Résultats attendus (baseline Phase 1) :
///    - Config A (well-being non-stationnaire) : ~20%
///    - Config B (reward stationnaire)        : ~90%
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

// ─── Environnement 5×5 (identique à phase1b) ─────────────────────────────

const W: usize = 5;
const H: usize = 5;
const PERCEPTION_DIM: usize = 6; // 4 whiskers + food_sense + water_sense
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;

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

    fn perceive(&self) -> Array1<f64> {
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

        let mut fs = 0.0;
        for &(fx, fy) in &FOOD_POSITIONS {
            let d = (((ix - fx as isize).abs().pow(2) + (iy - fy as isize).abs().pow(2)) as f64).sqrt();
            if d <= 2.0 {
                fs = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }

        let mut ws = 0.0;
        for &(wx, wy) in &WATER_POSITIONS {
            let d = (((ix - wx as isize).abs().pow(2) + (iy - wy as isize).abs().pow(2)) as f64).sqrt();
            if d <= 2.0 {
                ws = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }

        Array1::from_vec(vec![ray(0, -1), ray(0, 1), ray(-1, 0), ray(1, 0), fs, ws])
    }

    fn step_env(&mut self, action: usize) -> f64 {
        if self.done {
            return 0.0;
        }
        self.step += 1;

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
            if self.step >= MAX_STEPS {
                self.done = true;
            }
            return -0.5;
        }

        self.agent = (nx as usize, ny as usize);

        if WATER_POSITIONS.contains(&self.agent) {
            self.done = true;
            return 10.0;
        }
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

// ─── BFS potential (shaping) ──────────────────────────────────────────────

fn compute_potential_map() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W - 1) + (H - 1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W];
    let mut q = VecDeque::new();

    for &(wx, wy) in &WATER_POSITIONS {
        dist[wx][wy] = Some(0);
        q.push_back((wx, wy));
    }

    while let Some((cx, cy)) = q.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() {
                    dist[nx][ny] = Some(d + 1);
                    q.push_back((nx, ny));
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

// ─── Configuration ────────────────────────────────────────────────────────

struct Config {
    label: &'static str,
    hidden_dim: usize,
    use_stationary_reward: bool,
    freeze_hypothalamus: bool,  // gèle énergie, hydratation, température, sleep_debt
}

fn run_experiment(cfg: &Config) {
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, cfg.hidden_dim);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = cfg.use_stationary_reward;

    let bfs_pot = compute_potential_map();

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;

    let t0 = Instant::now();
    let mut train_rewards = Vec::with_capacity(TRAIN_EPS);
    let mut train_success = Vec::with_capacity(TRAIN_EPS);

    // ── Entraînement ─────────────────────────────────────────────────
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;

        let (total, succeeded) = run_ep(ep as u64, &mut engine, &bfs_pot, cfg);
        train_rewards.push(total);
        train_success.push(succeeded);
    }

    let elapsed = t0.elapsed();
    let train_avg = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_200 = train_rewards[TRAIN_EPS - 200..].iter().sum::<f64>() / 200.0;
    let train_success_rate = train_success.iter().filter(|&&s| s).count() as f64 / TRAIN_EPS as f64;

    // ── Test ε=0 ─────────────────────────────────────────────────────
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    let mut test_rewards = Vec::with_capacity(TEST_EPS);
    let mut test_success = Vec::with_capacity(TEST_EPS);
    for ep in 0..TEST_EPS {
        let (total, succeeded) = run_ep(1000 + ep as u64, &mut engine, &bfs_pot, cfg);
        test_rewards.push(total);
        test_success.push(succeeded);
    }

    let test_avg = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s| s).count() as f64 / TEST_EPS as f64;
    let debug: Vec<f64> = test_rewards.iter().take(10).copied().collect();
    let c = engine.num_concepts();
    let e = engine.graph_edges();
    let p = engine.current_phi;
    let rl = engine.cerebellum.replay.len();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<66} ║", cfg.label);
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    if cfg.use_stationary_reward {
        eprintln!("║  ✔ Correctifs e03 ACTIFS  (reward stationnaire + γ·Φ_BFS)          ║");
    } else {
        eprintln!("║  ✘ Well-being original  (9-termes non-stationnaire)               ║");
    }
    eprintln!("║  hd={}  stationary={}", cfg.hidden_dim, cfg.use_stationary_reward);
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last200={:>7.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_200, train_success_rate * 100.0);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  success={:.1}%",
        TEST_EPS, test_avg, test_success_rate * 100.0);
    eprintln!("║  C={} E={} Φ={:.3} Replay={} Debug={:?}", c, e, p, rl, debug);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

fn run_ep(_seed: u64, engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], cfg: &Config) -> (f64, bool) {
    let mut env = GridEnv5x5::new();
    env.reset();
    engine.end_episode();

    if cfg.freeze_hypothalamus {
        engine.hypothalamus.energy = 1.0;
        engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5;
        engine.hypothalamus.sleep_debt = 0.0;
    }

    let mut total_reward = 0.0;
    let mut succeeded = false;

    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a);
        total_reward += r;
        if env.done {
            succeeded = r > 0.0;
            let pt = env.perceive();
            engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
            break;
        }
        if cfg.freeze_hypothalamus {
            engine.hypothalamus.energy = 1.0;
            engine.hypothalamus.hydration = 1.0;
            engine.hypothalamus.temperature = 0.5;
            engine.hypothalamus.sleep_debt = 0.0;
        }
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();

    if engine.cerebellum.replay.len() >= 64 {
        engine.cerebellum.replay_train(64, 0.95, 10);
    }
    (total_reward, succeeded)
}

fn main() {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  E03S04 — Expérience comparative : TSO avec/sans reward stationnaire ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Config A : use_stationary_reward = false  (well-being original)      ║");
    eprintln!("║  Config B : use_stationary_reward = true   (reward stationnaire)      ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Prédiction : Config A ~20%, Config B ~90%                           ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // Config A — well-being non-stationnaire (original)
    run_experiment(&Config {
        label: "A. TSO original (well-being non-stationnaire)",
        hidden_dim: 4,
        use_stationary_reward: false,
        freeze_hypothalamus: false,
    });

    // Config B — reward stationnaire seul (sans freeze hypothalamus)
    run_experiment(&Config {
        label: "B. TSO + reward stationnaire (sans freeze hypothalamus)",
        hidden_dim: 4,
        use_stationary_reward: true,
        freeze_hypothalamus: false,
    });

    // Config C — reward stationnaire + hypothalamus gelé
    run_experiment(&Config {
        label: "C. TSO + reward stationnaire + hypothalamus gelé",
        hidden_dim: 4,
        use_stationary_reward: true,
        freeze_hypothalamus: true,
    });

    // Config D — well-being original + hypothalamus gelé
    run_experiment(&Config {
        label: "D. TSO original + hypothalamus gelé",
        hidden_dim: 4,
        use_stationary_reward: false,
        freeze_hypothalamus: true,
    });

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSULTATS                                                           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  A = non-stationnaire, hypothalamus libre     → ~20% (baseline)       ║");
    eprintln!("║  B = stationnaire, hypothalamus libre          → isole le RL signal    ║");
    eprintln!("║  C = stationnaire + hypothalamus gelé          → combiné optimal       ║");
    eprintln!("║  D = non-stationnaire + hypothalamus gelé      → isole l'homéostasie   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
