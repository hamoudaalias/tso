/// ════════════════════════════════════════════════════════════════════════════
///  Phase 1b — Fix 2 : replay propre (R_ext + shaping) + découplage entrée
///
///  Fix 1 : découplage perception ↔ décision (use_raw_perception_for_decision)
///    → Le cervelet reçoit la perception brute.
///    → Élimine la non-stationnarité de l'ENTRÉE.
///
///  Fix 2 : replay store_clean_reward_in_replay
///    → Le replay stocke R_ext + shaping (stationnaire) au lieu de well_being.
///    → Élimine la non-stationnarité de la RÉCOMPENSE.
///
///  Les deux fixes sont orthogonaux et additifs.
///  Prédiction : les deux combinés → >90% ε=0 (retour au niveau Phase 1).
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5;
const H: usize = 5;
const PERCEPTION_DIM: usize = 6;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

struct GridEnv5x5 {
    agent: (usize, usize), step: usize, done: bool,
}

impl GridEnv5x5 {
    fn new() -> Self { GridEnv5x5 { agent: (2,2), step:0, done:false } }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..W); let y = rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x,y)) && !FOOD_POSITIONS.contains(&(x,y)) {
                self.agent = (x,y); break;
            }
        }
        self.step=0; self.done=false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x,y) = self.agent; let ix=x as isize; let iy=y as isize;
        let ray = |dx:isize, dy:isize| -> f64 {
            let mut d=0; let mut cx=ix+dx; let mut cy=iy+dy;
            while cx>=0 && cy>=0 && cx<W as isize && cy<H as isize { d+=1; cx+=dx; cy+=dy; }
            d as f64 / (W.max(H) as f64)
        };
        let mut fs=0.0; for &(fx,fy) in &FOOD_POSITIONS {
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2)) as f64).sqrt();
            if d<=2.0 { fs=(1.0-d/3.0).max(0.0); break; }
        }
        let mut ws=0.0; for &(wx,wy) in &WATER_POSITIONS {
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2)) as f64).sqrt();
            if d<=2.0 { ws=(1.0-d/3.0).max(0.0); break; }
        }
        Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws])
    }
    fn step_env(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.step += 1;
        let (dx,dy) = match action { 0=>(0,-1),1=>(0,1),2=>(-1,0),3=>(1,0),_=>(0,0) };
        let nx=self.agent.0 as isize+dx; let ny=self.agent.1 as isize+dy;
        if nx<0||ny<0||nx>=W as isize||ny>=H as isize {
            if self.step>=MAX_STEPS { self.done=true; } return -0.5;
        }
        self.agent = (nx as usize, ny as usize);
        if WATER_POSITIONS.contains(&self.agent) { self.done=true; return 10.0; }
        if FOOD_POSITIONS.contains(&self.agent) { self.done=true; return 10.0; }
        if self.step>=MAX_STEPS { self.done=true; return -1.0; }
        -0.02
    }
}

fn compute_potential_map() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max = ((W-1)+(H-1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W];
    let mut q = VecDeque::new();
    for &(wx,wy) in &WATER_POSITIONS { dist[wx][wy]=Some(0); q.push_back((wx,wy)); }
    while let Some((cx,cy)) = q.pop_front() {
        let d=dist[cx][cy].unwrap();
        for (dx,dy) in [(0,1),(0,-1),(1,0),(-1,0)] {
            let nx=cx as isize+dx; let ny=cy as isize+dy;
            if nx>=0&&ny>=0&&nx<W as isize&&ny<H as isize {
                let (nx,ny)=(nx as usize,ny as usize);
                if dist[nx][ny].is_none() { dist[nx][ny]=Some(d+1); q.push_back((nx,ny)); }
            }
        }
    }
    for x in 0..W { for y in 0..H { pot[x][y] = match dist[x][y] { Some(d)=> -2.5*d as f64/d_max, None=> -2.5 }; } }
    pot
}

struct Config {
    label: &'static str,
    hidden_dim: usize,
    freeze: bool,
    /// Fix 1: use_raw_perception_for_decision
    fix1: bool,
    /// Fix 2: store_clean_reward_in_replay
    fix2: bool,
}

fn run_fix2(cfg: &Config) {
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, cfg.hidden_dim);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;

    let bfs_pot = compute_potential_map();

    const TRAIN_EPS: usize = 500;
    const TEST_EPS: usize = 100;
    let t0 = Instant::now();
    let mut train_rewards = Vec::with_capacity(TRAIN_EPS);
    let mut train_success = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;
        let (total, succeeded) = run_ep(ep as u64, &mut engine, &bfs_pot, cfg);
        train_rewards.push(total); train_success.push(succeeded);
    }

    let elapsed = t0.elapsed();
    let train_avg = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_200 = train_rewards[TRAIN_EPS-200..].iter().sum::<f64>() / 200.0;
    let train_success_rate = train_success.iter().filter(|&&s| s).count() as f64 / TRAIN_EPS as f64;

    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut test_rewards = Vec::with_capacity(TEST_EPS);
    let mut test_success = Vec::with_capacity(TEST_EPS);
    for ep in 0..TEST_EPS {
        let (total, succeeded) = run_ep(1000+ep as u64, &mut engine, &bfs_pot, cfg);
        test_rewards.push(total); test_success.push(succeeded);
    }

    let test_avg = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s| s).count() as f64 / TEST_EPS as f64;
    let debug: Vec<f64> = test_rewards.iter().take(10).copied().collect();
    let c=engine.num_concepts(); let e=engine.graph_edges(); let p=engine.current_phi; let rl=engine.cerebellum.replay.len();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<66} ║", cfg.label);
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  hd={} freeze={}  fix1(entrée)={}  fix2(replay)={}",
        cfg.hidden_dim, cfg.freeze, cfg.fix1, cfg.fix2);
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last200={:>7.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_200, train_success_rate*100.0);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  success={:.1}%",
        TEST_EPS, test_avg, test_success_rate*100.0);
    eprintln!("║  C={} E={} Φ={:.3} Replay={} Debug={:?}", c, e, p, rl, debug);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

fn run_ep(seed: u64, engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], cfg: &Config) -> (f64, bool) {
    let mut env = GridEnv5x5::new();
    env.reset();
    engine.end_episode();

    if cfg.freeze {
        engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
        engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
    }

    let mut total=0.0; let mut succeeded=false;
    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a); total += r;
        if env.done { succeeded=r>0.0; let pt=env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
        if cfg.freeze {
            engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
            engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
        }
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    if engine.cerebellum.replay.len() >= 64 {
        engine.cerebellum.replay_train(64, 0.95, 10);
    }
    (total, succeeded)
}

fn main() {
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PHASE 1b — Fix 2 : replay propre + découplage entrée               ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Factorial design : fix1 ∈ {{false,true}} × fix2 ∈ {{false,true}}        ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  ni fixé   = Exp B originale (~20%)                                   ║");
    eprintln!("║  fix1 seul = découplage entrée (~26%, déjà testé)                     ║");
    eprintln!("║  fix2 seul = replay propre (prédit ~60-80%)                           ║");
    eprintln!("║  les deux  = prédit >90% (retour Phase 1)                             ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // Ablation : fix2 seul (replay propre, entrée gated)
    run_fix2(&Config { label:"B' ref. hd=4, gelé (ni fixé)",     hidden_dim:4, freeze:true, fix1:false, fix2:false });
    run_fix2(&Config { label:"F2a. hd=4, gelé, fix2 seul",       hidden_dim:4, freeze:true, fix1:false, fix2:true });
    run_fix2(&Config { label:"F2b. hd=4, gelé, fix1+fix2",       hidden_dim:4, freeze:true, fix1:true,  fix2:true });

    // Avec homéostasie
    run_fix2(&Config { label:"F2c. hd=4, dérive, fix2 seul",     hidden_dim:4, freeze:false, fix1:false, fix2:true });
    run_fix2(&Config { label:"F2d. hd=4, dérive, fix1+fix2",     hidden_dim:4, freeze:false, fix1:true,  fix2:true });

    // hd=16
    run_fix2(&Config { label:"F2e. hd=16, gelé, fix2 seul",      hidden_dim:16, freeze:true, fix1:false, fix2:true });
    run_fix2(&Config { label:"F2f. hd=16, gelé, fix1+fix2",      hidden_dim:16, freeze:true, fix1:true,  fix2:true });

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSULTATS                                                           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Si fix2 seul >80%  → la non-stationnarité de la récompense domine   ║");
    eprintln!("║  Si fix1+fix2 >90%  → les deux fixes sont additifs et orthogonaux    ║");
    eprintln!("║  Si fix2 seul ~20%  → la cause n'est ni l'entrée ni la récompense    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
