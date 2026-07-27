/// ════════════════════════════════════════════════════════════════════════════
///  Expérience A — Cerebellum seul sur Terrarium 7×7
///
///  But : isoler si l'aliasing perceptuel du Terrarium 7×7 (murs internes,
///  49 positions) empêche le Cerebellum + shaping + replay de généraliser
///  en exploitation pure.
///
///  Cadran :
///                  5×5 (pas d'aliasing)      7×7 muré (aliasing)
///  Cervelet seul   98% (Phase 1 #8)          Exp A → ?
///
///  Si Exp A > 50% : l'aliasing n'est pas le problème majeur en 7×7.
///  Si Exp A < 30% : l'aliasing bloque, même avec shaping+replay.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::multi_grid_cells::MultiGridCell;

// ─── Terrarium 7×7 ─────────────────────────────────────────────────────────
// Perception : [wall_N, wall_S, wall_W, wall_E] = 4 moustaches
// Actions    : 0=N, 1=S, 2=W, 3=E
// Murs internes identiques au Terrarium original (terrarium.rs)
const W: usize = 7;
const H: usize = 7;
const PERCEPTION_DIM: usize = 4;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 200;

// Périodes multi-grid pour 7×7 : [3, 5, 7] → produit 105 > 49 = injectif
const GRID_PERIODS: [usize; 3] = [3, 5, 7];
const EXTRA_DIM: usize = GRID_PERIODS.len() * 4; // 12
const TOTAL_DIM: usize = PERCEPTION_DIM + EXTRA_DIM; // 16

// Eau uniquement, aux positions du Terrarium original
const WATER_POSITIONS: [(usize, usize); 3] = [(5, 1), (2, 5), (4, 2)];

// Murs du Terrarium original (terrarium.rs)
// For x=2..w-1: walls[x][3]=true; for y=1..h-1: walls[3][y]=true; walls[2][3]=false; walls[3][2]=false; walls[4][3]=false
fn is_walkable(x: isize, y: isize) -> bool {
    if x < 0 || y < 0 || x >= W as isize || y >= H as isize { return false; }
    let (x, y) = (x as usize, y as usize);
    // Borders are walls
    if x == 0 || x == W-1 || y == 0 || y == H-1 { return false; }
    // Internal walls
    if (2..=5).contains(&x) && y == 3 { return x == 2 || x == 4; } // passages at (2,3) and (4,3)
    if x == 3 && (1..=5).contains(&y) { return false; } // vertical wall at x=3 except passages
    true
}

/// Precompute BFS potential for Terrarium 7×7
fn compute_bfs_potential() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W-1)+(H-1)) as f64; // 12
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W];
    let mut q = VecDeque::new();
    for &(wx, wy) in &WATER_POSITIONS {
        if is_walkable(wx as isize, wy as isize) { dist[wx][wy] = Some(0); q.push_back((wx, wy)); }
    }
    while let Some((cx, cy)) = q.pop_front() {
        let d = dist[cx][cy].unwrap();
        for (dx, dy) in [(0,1),(0,-1),(1,0),(-1,0)] {
            let nx = cx as isize + dx; let ny = cy as isize + dy;
            if is_walkable(nx, ny) {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() { dist[nx][ny] = Some(d+1); q.push_back((nx, ny)); }
            }
        }
    }
    for x in 0..W {
        for y in 0..H {
            pot[x][y] = match dist[x][y] {
                Some(d) => -2.5 * d as f64 / d_max,
                None => -2.5, // unreachable
            };
        }
    }
    pot
}

struct TerEnv {
    agent: (usize, usize), step: usize, done: bool,
}

impl TerEnv {
    fn new() -> Self { TerEnv { agent: (1, 1), step: 0, done: false } }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(1..W-1); let y = rng.gen_range(1..H-1);
            if is_walkable(x as isize, y as isize) && !WATER_POSITIONS.contains(&(x, y)) {
                self.agent = (x, y); break;
            }
        }
        self.step = 0; self.done = false;
    }

    /// Raw 4-whisker perception
    fn perceive(&self) -> (Vec<f64>, usize, usize) {
        let (x, y) = self.agent; let ix = x as isize; let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0; let mut cx = ix+dx; let mut cy = iy+dy;
            while is_walkable(cx, cy) { d += 1; cx += dx; cy += dy; }
            d as f64 / (W.max(H) as f64)
        };
        (vec![ray(0,-1), ray(0,1), ray(-1,0), ray(1,0)], x, y)
    }

    fn step_env(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.step += 1;
        let (dx, dy) = match action { 0=>(0,-1), 1=>(0,1), 2=>(-1,0), 3=>(1,0), _=>(0,0) };
        let nx = self.agent.0 as isize + dx; let ny = self.agent.1 as isize + dy;

        if !is_walkable(nx, ny) {
            if self.step >= MAX_STEPS { self.done = true; }
            return -0.5;
        }
        self.agent = (nx as usize, ny as usize);

        if WATER_POSITIONS.contains(&self.agent) { self.done = true; return 10.0; }
        if self.step >= MAX_STEPS { self.done = true; return -1.0; }
        -0.02
    }
}

// ─── Runner ────────────────────────────────────────────────────────────────

struct Config {
    label: &'static str,
    hidden_dim: usize,
    replay_lr: f64,
    use_grid: bool,
    use_shaping: bool,
}

fn run_experiment_a(cfg: &Config) {
    let grid_cells = MultiGridCell::new(W, H, &GRID_PERIODS);
    if cfg.use_grid {
        let injective = grid_cells.test_injectivity(W, H);
        if !injective { eprintln!("❌ Exp A ABORT: code grid cells non injectif"); return; }
    }

    let bfs_pot = compute_bfs_potential();
    let dim = if cfg.use_grid { TOTAL_DIM } else { PERCEPTION_DIM };

    let mut cerebellum = Cerebellum::new(dim, N_ACTIONS, 0.30, 0.1, 0.50, cfg.hidden_dim);
    cerebellum.epsilon = 0.1; cerebellum.noise_std = 0.1;
    cerebellum.replay_lr = cfg.replay_lr;
    cerebellum.replay_only = false;

    const TRAIN_EPS: usize = 1000;
    const TEST_EPS: usize = 200;
    const GAMMA: f64 = 0.99;
    const D_MAX: f64 = ((W-1)+(H-1)) as f64;

    let t0 = Instant::now();
    let mut train_rewards = Vec::with_capacity(TRAIN_EPS);
    let mut train_success = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cerebellum.epsilon = 0.8 * remain + 0.01;
        cerebellum.noise_std = 0.3 * remain + 0.01;
        let (total, succeeded) = run_ep(&mut cerebellum, &grid_cells, &bfs_pot, cfg, D_MAX);
        train_rewards.push(total); train_success.push(succeeded);
    }

    let elapsed = t0.elapsed();
    let train_avg = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_500 = train_rewards[TRAIN_EPS-500..].iter().sum::<f64>() / 500.0;
    let train_success_rate = train_success.iter().filter(|&&s| s).count() as f64 / TRAIN_EPS as f64;

    // Test ε=0
    cerebellum.epsilon = 0.0; cerebellum.noise_std = 0.0;
    let mut test_rewards = Vec::with_capacity(TEST_EPS);
    let mut test_success = Vec::with_capacity(TEST_EPS);
    for ep in 0..TEST_EPS {
        let (total, succeeded) = run_ep(&mut cerebellum, &grid_cells, &bfs_pot, cfg, D_MAX);
        test_rewards.push(total); test_success.push(succeeded);
    }

    let test_avg = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s| s).count() as f64 / TEST_EPS as f64;
    let debug: Vec<f64> = test_rewards.iter().take(10).copied().collect();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<66} ║", cfg.label);
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  hd={} grid={} shaping={} replay_lr={} dim={}",
        cfg.hidden_dim, cfg.use_grid, cfg.use_shaping, cfg.replay_lr, dim);
    if cfg.use_grid {
        eprintln!("║  Grid cells: {} modules × 4 = {} dim (périodes {:?})  injectif: {}>{}",
            GRID_PERIODS.len(), EXTRA_DIM, GRID_PERIODS, GRID_PERIODS.iter().product::<usize>(), W*H);
    }
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last500={:>7.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_500, train_success_rate*100.0);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  success={:.1}%  replay={}",
        TEST_EPS, test_avg, test_success_rate*100.0, cerebellum.replay.len());
    eprintln!("║  10 premiers tests: {:?}", debug);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

fn run_ep(cerebellum: &mut Cerebellum, grid: &MultiGridCell, bfs_pot: &[Vec<f64>], cfg: &Config, d_max: f64) -> (f64, bool) {
    let mut env = TerEnv::new(); env.reset();
    cerebellum.reset_trace();

    let mut total = 0.0; let mut succeeded = false;

    // First step
    let (p_raw, x, y) = env.perceive();
    let perception = if cfg.use_grid { grid.augment(&p_raw, x, y) } else { p_raw };
    let state = Array1::from_vec(perception);
    let pot = bfs_pot[x][y];
    let mut logits = cerebellum.forward_logits(&state);

    let mut rng = rand::thread_rng();
    let exploring = cerebellum.noise_std > 0.0;
    let action = if exploring && rand::random::<f64>() < cerebellum.epsilon {
        rng.gen_range(0..N_ACTIONS)
    } else {
        if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cerebellum.noise_std..cerebellum.noise_std); } }
        logits.iter().enumerate().max_by(|(_,a),(_,b)|a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)|i).unwrap_or(0)
    };
    cerebellum.mark(&state, action);

    let mut prev_state = state.clone();
    let mut prev_pot = pot;
    let mut action = action;

    while !env.done {
        let reward = env.step_env(action);
        if env.done { succeeded = reward > 0.0; }
        let (p_raw, x, y) = env.perceive();

        // Compute RL signal: R_ext + γ·Φ_BFS(s') - Φ_BFS(s)
        let perception = if cfg.use_grid { grid.augment(&p_raw, x, y) } else { p_raw };
        let next_state = Array1::from_vec(perception);
        let next_pot = bfs_pot[x][y];
        let shaping = if cfg.use_shaping { 0.99 * next_pot - prev_pot } else { 0.0 };
        let rl_signal = reward + shaping;

        // Forward & criticize
        _ = cerebellum.forward_logits(&next_state);
        cerebellum.reinforce_td(rl_signal, 0.99);
        cerebellum.decay_trace(0.99, 0.98);

        // Replay with clean reward
        cerebellum.store_transition(&prev_state, action, rl_signal, &next_state, env.done);

        if env.done { total += reward; break; }

        // Next action
        let mut logits = cerebellum.forward_logits(&next_state);
        let exploring = cerebellum.noise_std > 0.0;
        action = if exploring && rand::random::<f64>() < cerebellum.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cerebellum.noise_std..cerebellum.noise_std); } }
            logits.iter().enumerate().max_by(|(_,a),(_,b)|a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)|i).unwrap_or(0)
        };
        cerebellum.mark(&next_state, action);

        prev_state = next_state; prev_pot = next_pot; total += reward;
    }

    if cerebellum.replay.len() >= 64 {
        cerebellum.replay_train(64, 0.95, 10);
    }
    (total, succeeded)
}

fn main() {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  EXPÉRIENCE A — Cerebellum seul sur Terrarium 7×7 muré               ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Objectif : l'aliasing 7×7 empêche-t-il l'apprentissage même avec     ║");
    eprintln!("║  shaping BFS + replay propre ?                                        ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Prédiction : base 4D s'effondre (<30%), grid+shaping+replay >50%    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // Série A1 : Sans grid cells
    run_experiment_a(&Config { label:"A1. hd=4, base (sans grille, sans shaping)",       hidden_dim:4, replay_lr:0.0, use_grid:false, use_shaping:false });
    run_experiment_a(&Config { label:"A2. hd=4, shaping seul",                            hidden_dim:4, replay_lr:0.0, use_grid:false, use_shaping:true });
    run_experiment_a(&Config { label:"A3. hd=4, shaping+replay",                          hidden_dim:4, replay_lr:0.05, use_grid:false, use_shaping:true });

    // Série A2 : Avec grid cells injectives
    run_experiment_a(&Config { label:"B1. hd=4, grid+shaping",                            hidden_dim:4, replay_lr:0.0, use_grid:true, use_shaping:true });
    run_experiment_a(&Config { label:"B2. hd=4, grid+shaping+replay",                     hidden_dim:4, replay_lr:0.05, use_grid:true, use_shaping:true });
    run_experiment_a(&Config { label:"B3. hd=16, grid+shaping+replay",                    hidden_dim:16, replay_lr:0.05, use_grid:true, use_shaping:true });

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSULTATS                                                           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Si A3 > 50% : l'aliasing 7×7 est gérable avec shaping+replay seul   ║");
    eprintln!("║  Si A3 < 30% : même shaping ne suffit pas → aliasing sévère           ║");
    eprintln!("║  Si B2 > A3 : les grid cells injectives aident au-delà du shaping     ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
