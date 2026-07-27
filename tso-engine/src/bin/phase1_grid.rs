/// ════════════════════════════════════════════════════════════════════════════
///  Phase 1 — Preuve de concept : cervelet + grid cells multi-modules + shaping BFS
///
///  Objectif : démontrer que le Cerebellum MLP peut apprendre une politique
///  exploitable (ε=0) sur un MDP simple avec :
///    - Code de grille multi-module INJECTIF (périodes [2,3,5])
///    - Shaping potential-based BFS (γ·Φ(s') − Φ(s), Ng et al. 1999)
///    - Replay buffer stockant R_ext + shaping (pas well_being)
///    - Hypothalamus gelé, pas de curiosité, pas de métabolique
///    - ε=0.1 entraînement, ε=0 test
///
///  Pièges évités :
///    ✓ grid cells forcées sur 5×5 (force_on contourne le seuil auto 36)
///    ✓ Code multi-module injectif → vérifié par test_injectivity()
///    ✓ Signal RL uniforme en ligne ET replay
///    ✓ Assertion gated.len() constant sur tous les steps
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::multi_grid_cells::MultiGridCell;

// ─── Paramètres ────────────────────────────────────────────────────────────
const GRID_W: usize = 5;
const GRID_H: usize = 5;
const PERCEPTION_DIM: usize = 4; // whiskers N,S,W,E seulement
const GRID_PERIODS: [usize; 3] = [2, 3, 5]; // 2×3×5 = 30 > 25 = injectif
const EXTRA_DIM: usize = GRID_PERIODS.len() * 4; // 12
const TOTAL_DIM: usize = PERCEPTION_DIM + EXTRA_DIM; // 16
const N_ACTIONS: usize = 4; // N,S,W,E
const MAX_STEPS: usize = 100;

// Positions de l'eau (toujours les mêmes)
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

// ─── BFS potential map ─────────────────────────────────────────────────────
// Potentiel Φ(s) = -2.5 × d(s) / d_max
// où d(s) = distance de Manhattan à la ressource la plus proche
// d_max = distance max sur la grille = (GRID_W-1) + (GRID_H-1) = 8
const D_MAX: f64 = ((GRID_W - 1) + (GRID_H - 1)) as f64;
const BFS_GAIN: f64 = 2.5;

/// Precompute BFS potential for each position on the grid.
/// Φ(s) = -BFS_GAIN × d_nearest_water / D_MAX
fn compute_bfs_potential() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;

    let mut dist = vec![vec![None::<usize>; GRID_H]; GRID_W];
    let mut queue = VecDeque::new();

    // Enqueue all water positions at distance 0
    for &(wx, wy) in &WATER_POSITIONS {
        dist[wx][wy] = Some(0);
        queue.push_back((wx, wy));
    }

    // BFS flood fill (Manhattan, all cells walkable)
    while let Some((cx, cy)) = queue.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0isize, 1isize), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < GRID_W as isize && ny < GRID_H as isize {
                let nx = nx as usize;
                let ny = ny as usize;
                if dist[nx][ny].is_none() {
                    dist[nx][ny] = Some(d + 1);
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    // Convert distances to potentials
    let mut potential = vec![vec![0.0; GRID_H]; GRID_W];
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            match dist[x][y] {
                Some(d) => potential[x][y] = -BFS_GAIN * d as f64 / D_MAX,
                None => potential[x][y] = -BFS_GAIN, // unreachable = minimum
            }
        }
    }
    potential
}

// ─── Environnement 5×5 ──────────────────────────────────────────────────────
struct GridEnv {
    agent: (usize, usize),
    step: usize,
    done: bool,
}

impl GridEnv {
    fn new() -> Self {
        GridEnv {
            agent: (2, 2), // start center
            step: 0,
            done: false,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        // Random start position (not on water)
        loop {
            let x = rng.gen_range(0..GRID_W);
            let y = rng.gen_range(0..GRID_H);
            if !WATER_POSITIONS.contains(&(x, y)) {
                self.agent = (x, y);
                break;
            }
        }
        self.step = 0;
        self.done = false;
    }

    /// Raw perception: [wall_N, wall_S, wall_W, wall_E]
    /// Returns (perception_vector, x, y) — x,y needed for grid cell encoding
    fn perception_raw(&self) -> (Vec<f64>, usize, usize) {
        let (x, y) = self.agent;
        let w = GRID_W as isize;
        let h = GRID_H as isize;
        let ix = x as isize;
        let iy = y as isize;

        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0usize;
            let mut cx = ix + dx;
            let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < w && cy < h {
                d += 1;
                cx += dx;
                cy += dy;
            }
            d as f64 / (GRID_W.max(GRID_H) as f64) // normalize in [0, 1]
        };

        let p = vec![
            ray(0, -1), // N
            ray(0, 1),  // S
            ray(-1, 0), // W
            ray(1, 0),  // E
        ];
        (p, x, y)
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

        // Walkable if inside grid
        if nx < 0 || ny < 0 || nx >= GRID_W as isize || ny >= GRID_H as isize {
            if self.step >= MAX_STEPS { self.done = true; }
            return -0.5; // wall bump
        }

        self.agent = (nx as usize, ny as usize);

        // Water reward
        if WATER_POSITIONS.contains(&self.agent) {
            self.done = true;
            return 10.0;
        }

        if self.step >= MAX_STEPS {
            self.done = true;
            return -1.0;
        }

        -0.02 // step cost
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────
struct Config {
    label: &'static str,
    hidden_dim: usize,
    replay_lr: f64,
    use_grid: bool,
    use_shaping: bool,
}

fn run_phase1(cfg: &Config) {
    let grid_cells = MultiGridCell::new(GRID_W, GRID_H, &GRID_PERIODS);
    let bfs_pot = compute_bfs_potential();

    // Vérification d'injectivité
    if cfg.use_grid {
        let injective = grid_cells.test_injectivity(GRID_W, GRID_H);
        if !injective {
            eprintln!("❌ PHASE 1 ABORT: code grid cells non injectif");
            return;
        }
    }

    let dim = if cfg.use_grid { TOTAL_DIM } else { PERCEPTION_DIM };

    let mut cerebellum = Cerebellum::new(dim, N_ACTIONS, 0.30, 0.1, 0.50, cfg.hidden_dim);
    cerebellum.epsilon = 0.1;
    cerebellum.noise_std = 0.1;
    cerebellum.replay_lr = cfg.replay_lr;
    cerebellum.replay_only = false;

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;
    let t0 = Instant::now();
    let mut train_rewards: Vec<f64> = Vec::with_capacity(TRAIN_EPS);
    let mut train_success: Vec<bool> = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        // Annealing: ε décroît linéairement
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cerebellum.epsilon = 0.8 * remain + 0.01;
        cerebellum.noise_std = 0.3 * remain + 0.01;

    let (total, succeeded) = run_episode(&mut cerebellum, &grid_cells, &bfs_pot, cfg);
        train_rewards.push(total);
        train_success.push(succeeded);
    }

    let elapsed = t0.elapsed();
    let train_avg: f64 = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_200: f64 = train_rewards[TRAIN_EPS - 200..].iter().sum::<f64>() / 200.0;
    let train_success_rate = train_success.iter().filter(|&&s| s).count() as f64 / TRAIN_EPS as f64;

    // TEST — ε=0, noise_std=0
    cerebellum.epsilon = 0.0;
    cerebellum.noise_std = 0.0;
    let mut test_rewards: Vec<f64> = Vec::with_capacity(TEST_EPS);
    let mut test_success: Vec<bool> = Vec::with_capacity(TEST_EPS);

    for _seed in 0..TEST_EPS {
        let (total, succeeded) = run_episode(&mut cerebellum, &grid_cells, &bfs_pot, cfg);
        test_rewards.push(total);
        test_success.push(succeeded);
    }

    let test_avg: f64 = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s| s).count() as f64 / TEST_EPS as f64;

    // Debug: 10 premiers tests
    let debug_sample: Vec<f64> = test_rewards.iter().take(10).copied().collect();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<66} ║", cfg.label);
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  hd={} grid={} shaping={} replay_lr={} dim={}",
        cfg.hidden_dim, cfg.use_grid, cfg.use_shaping, cfg.replay_lr, dim);
    eprintln!("║  Grid cells: {} modules × 4 = {} dim  (périodes {:?})",
        GRID_PERIODS.len(), EXTRA_DIM, GRID_PERIODS);
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last200={:>7.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_200,
        train_success_rate * 100.0);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  success={:.1}%  replay={}",
        TEST_EPS, test_avg, test_success_rate * 100.0, cerebellum.replay.len());
    eprintln!("║  10 premiers tests: {:?}", debug_sample);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

/// Exécute un épisode complet.
/// Retourne (total_reward, a_trouvé_l_eau).
fn run_episode(
    cerebellum: &mut Cerebellum,
    grid_cells: &MultiGridCell,
    bfs_pot: &[Vec<f64>],
    cfg: &Config,
) -> (f64, bool) {
    let mut env = GridEnv::new();
    env.reset();

    let mut total_reward = 0.0;
    let mut succeeded = false;
    cerebellum.reset_trace();

    // Premier step : obtenir la perception initiale
    let (p_raw, x, y) = env.perception_raw();
    let perception = if cfg.use_grid {
        grid_cells.augment(&p_raw, x, y)
    } else {
        p_raw
    };
    let decision_state = Array1::from_vec(perception.clone());
    let mut logits = cerebellum.forward_logits(&decision_state);

    // Action initiale (avec bruit d'exploration)
    let mut rng = rand::thread_rng();
    let exploring = cerebellum.noise_std > 0.0;
    let init_action = if exploring && rand::random::<f64>() < cerebellum.epsilon {
        rng.gen_range(0..N_ACTIONS)
    } else {
        if exploring {
            for l in logits.iter_mut() {
                *l += rng.gen_range(-cerebellum.noise_std..cerebellum.noise_std);
            }
        }
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0)
    };
    cerebellum.mark(&decision_state, init_action);

    // Clamp les weights initiaux pour éviter NaN

    let mut action = init_action;
    let mut prev_state = decision_state.clone();
    let mut prev_pot = bfs_pot[env.agent.0][env.agent.1];

    while !env.done {
        // Applique l'action dans l'environnement
        let reward = env.step_env(action);
        if env.done {
            succeeded = reward > 0.0;
            // Perception de l'état terminal (peu importe, on ne bouge plus)
            let (p_raw, x, y) = env.perception_raw();
            let perception = if cfg.use_grid {
                grid_cells.augment(&p_raw, x, y)
            } else {
                p_raw
            };
            let next_state = Array1::from_vec(perception.clone());

            // Shaping: γ·Φ(s') - Φ(s)
            let next_pot = bfs_pot[env.agent.0][env.agent.1]; // should be 0.0 if on water
            let shaping = if cfg.use_shaping {
                0.99 * next_pot - prev_pot
            } else {
                0.0
            };

            // Signal RL propre : R_ext + shaping
            let rl_signal = reward + shaping;

            // Critic update (TD)
            _ = cerebellum.forward_logits(&next_state);
            cerebellum.reinforce_td(rl_signal, 0.99);

            // Replay buffer: stocke la transition
            cerebellum.store_transition(&prev_state, action, rl_signal, &next_state, true);

            total_reward += reward;
            break;
        }

        // Perception de l'état suivant (avant action)
        let (p_raw, x, y) = env.perception_raw();
        let perception = if cfg.use_grid {
            grid_cells.augment(&p_raw, x, y)
        } else {
            p_raw
        };
        let next_state = Array1::from_vec(perception.clone());
        let next_pot = bfs_pot[x][y];

        // Shaping: γ·Φ(s') - Φ(s)
        let shaping = if cfg.use_shaping {
            0.99 * next_pot - prev_pot
        } else {
            0.0
        };

        // Signal RL propre
        let rl_signal = reward + shaping;

        // ASSERT: la dimension de l'état est constante
        assert_eq!(prev_state.len(), next_state.len(),
            "Dimension mismatch: prev={}, next={}", prev_state.len(), next_state.len());

        // Critic update
        _ = cerebellum.forward_logits(&next_state);
        cerebellum.reinforce_td(rl_signal, 0.99);
        cerebellum.decay_trace(0.99, 0.98);

        // Replay buffer: stocke la transition avec le signal RL propre
        cerebellum.store_transition(&prev_state, action, rl_signal, &next_state, false);

        // Sélection de la prochaine action
        let mut logits = cerebellum.forward_logits(&next_state);
        let exploring = cerebellum.noise_std > 0.0;
        action = if exploring && rand::random::<f64>() < cerebellum.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            if exploring {
                for l in logits.iter_mut() {
                    *l += rng.gen_range(-cerebellum.noise_std..cerebellum.noise_std);
                }
            }
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0)
        };
        cerebellum.mark(&next_state, action);

        prev_state = next_state;
        prev_pot = next_pot;
        total_reward += reward;
    }

    // Replay training après chaque épisode
    if cerebellum.replay.len() >= 64 {
        cerebellum.replay_train(64, 0.95, 10);
    }

    (total_reward, succeeded)
}

// ─── Main ───────────────────────────────────────────────────────────────────
fn main() {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PHASE 1 — Cerebellum MLP + Grid cells injectives + BFS shaping      ║");
    eprintln!("║  Grille 5×5, eau en (1,1),(3,3),(1,4)                               ║");
    eprintln!("║  Grid cells multi-module [2,3,5] → {} dim (injectif : 30 > 25)        ║", EXTRA_DIM);
    eprintln!("║  Shaping BFS potential-based (γ·Φ(s')−Φ(s), Ng et al. 1999)          ║");
    eprintln!("║  Signal RL = R_ext + shaping (partout : ligne ET replay)             ║");
    eprintln!("║  Pas d'hypothalamus, pas de curiosité, pas de métabolique            ║");
    eprintln!("║  Critère : succès en test ε=0 > 0%                                   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Series 1: Sans grid cells, sans shaping (baseline) ──
    run_phase1(&Config {
        label: "1. hd=4, base (sans grille, sans shaping)",
        hidden_dim: 4, replay_lr: 0.0, use_grid: false, use_shaping: false,
    });

    // ── Series 2: Grid cells seules ──
    run_phase1(&Config {
        label: "2. hd=4, grid (sans shaping)",
        hidden_dim: 4, replay_lr: 0.0, use_grid: true, use_shaping: false,
    });

    // ── Series 3: Shaping BFS seul ──
    run_phase1(&Config {
        label: "3. hd=4, shaping (sans grille)",
        hidden_dim: 4, replay_lr: 0.0, use_grid: false, use_shaping: true,
    });

    // ── Series 4: Grid cells + shaping BFS ──
    run_phase1(&Config {
        label: "4. hd=4, grid+shaping (sans replay)",
        hidden_dim: 4, replay_lr: 0.0, use_grid: true, use_shaping: true,
    });

    // ── Series 5: hd=16, grid+shaping ──
    run_phase1(&Config {
        label: "5. hd=16, grid+shaping (sans replay)",
        hidden_dim: 16, replay_lr: 0.0, use_grid: true, use_shaping: true,
    });

    // ── Series 6: hd=4, grid+shaping+replay ──
    run_phase1(&Config {
        label: "6. hd=4, grid+shaping+replay",
        hidden_dim: 4, replay_lr: 0.05, use_grid: true, use_shaping: true,
    });

    // ── Series 7: hd=16, grid+shaping+replay ──
    run_phase1(&Config {
        label: "7. hd=16, grid+shaping+replay",
        hidden_dim: 16, replay_lr: 0.05, use_grid: true, use_shaping: true,
    });

    // ── Series 8: Sans grid cell mais avec shaping + replay (isolation du shaping) ──
    run_phase1(&Config {
        label: "8. hd=4, shaping+replay (sans grille)",
        hidden_dim: 4, replay_lr: 0.05, use_grid: false, use_shaping: true,
    });

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  FIN DE PHASE 1                                                       ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Résultat attendu :                                                   ║");
    eprintln!("║  Si config #4 ou #7 >0% en ε=0 : c'était gradient + aliasing          ║");
    eprintln!("║  Si config #7 = 0% mais code injectif : bug câblage (trace la dim)   ║");
    eprintln!("║  Si config #4 >0% mais config #2 = 0% : shaping était nécessaire      ║");
    eprintln!("║  Si config #2 >0% mais config #1 = 0% : aliasing était le seul bloc  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
