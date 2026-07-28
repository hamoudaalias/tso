#![allow(dead_code, unused_assignments, unused_variables)]
/// ════════════════════════════════════════════════════════════════════════════
///  Matrice multi-seeds §8 — Phase 1 #8 vs S1 vs S1+replay_only vs S1+δ-clip
///
///  Chaque config ×10 seeds. Résultat : succès test ε=0 médian + IQR.
///  Tranche la question seed-luck vs systématique.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6; const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];
const TRAIN_EPS: usize = 500;
const TEST_EPS: usize = 100;
const N_SEEDS: usize = 10;

struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool }
impl GridEnv5x5 {
    fn new() -> Self { GridEnv5x5 { agent: (2,2), step:0, done:false } }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop { let x=rng.gen_range(0..W); let y=rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x,y))&&!FOOD_POSITIONS.contains(&(x,y)){self.agent=(x,y);break;} }
        self.step=0; self.done=false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x,y)=self.agent; let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws])
    }
    fn step_env(&mut self, action: usize) -> f64 {
        if self.done{return 0.0;} self.step+=1;
        let(dx,dy)=match action{0=>(0,-1),1=>(0,1),2=>(-1,0),3=>(1,0),_=>(0,0)};
        let nx=self.agent.0 as isize+dx; let ny=self.agent.1 as isize+dy;
        if nx<0||ny<0||nx>=W as isize||ny>=H as isize{
            if self.step>=MAX_STEPS{self.done=true;}return-0.5;}
        self.agent=(nx as usize,ny as usize);
        if WATER_POSITIONS.contains(&self.agent){self.done=true;return 10.0;}
        if FOOD_POSITIONS.contains(&self.agent){self.done=true;return 10.0;}
        if self.step>=MAX_STEPS{self.done=true;return-1.0;}
        -0.02
    }
}

fn compute_bfs_potential() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max=((W-1)+(H-1))as f64;let mut pot=vec![vec![0.0;H];W];
    let mut dist=vec![vec![None::<usize>;H];W];let mut q=VecDeque::new();
    for &(wx,wy)in&WATER_POSITIONS{dist[wx][wy]=Some(0);q.push_back((wx,wy));}
    while let Some((cx,cy))=q.pop_front(){let d=dist[cx][cy].unwrap();
        for(dx,dy)in[(0,1),(0,-1),(1,0),(-1,0)]{let nx=cx as isize+dx;let ny=cy as isize+dy;
            if nx>=0&&ny>=0&&nx<W as isize&&ny<H as isize{let(nx,ny)=(nx as usize,ny as usize);
                if dist[nx][ny].is_none(){dist[nx][ny]=Some(d+1);q.push_back((nx,ny));}}}}
    for x in 0..W{for y in 0..H{pot[x][y]=match dist[x][y]{Some(d)=>-2.5*d as f64/d_max,None=>-2.5};}}
    pot
}

fn pick_greedy(engine: &TsoEngine, p: &Array1<f64>) -> usize {
    let hd = 4; let dim = p.len();
    let mut h = vec![0.0; hd];
    for j in 0..hd {
        let mut s = 0.0;
        for i in 0..dim { s += engine.cerebellum.get_hidden_weight(j, i) * p[i]; }
        h[j] = s.tanh();
    }
    let mut logits = vec![0.0; N_ACTIONS];
    for a in 0..N_ACTIONS {
        for j in 0..hd { logits[a] += engine.cerebellum.get_out_weight(a, j) * h[j]; }
    }
    logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
}

fn pick_greedy_phase1(cbl: &Cerebellum, p: &Array1<f64>, dim: usize) -> usize {
    let hd = 4;
    let mut h = vec![0.0; hd];
    for j in 0..hd {
        let mut s = 0.0;
        for i in 0..dim { s += cbl.get_hidden_weight(j, i) * p[i]; }
        h[j] = s.tanh();
    }
    let mut logits = vec![0.0; N_ACTIONS];
    for a in 0..N_ACTIONS {
        for j in 0..hd { logits[a] += cbl.get_out_weight(a, j) * h[j]; }
    }
    logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
}

/// Test ε=0 propre : forward manuel, pas de step() → pas d'apprentissage
fn test_clean(engine: &TsoEngine) -> f64 {
    let mut successes = 0u64;
    for _ in 0..TEST_EPS {
        let mut env = GridEnv5x5::new(); env.reset();
        let _pos = env.agent;
        let mut total = 0.0; let mut ok = false;
        let mut a = pick_greedy(engine, &env.perceive());
        while !env.done {
            let r = env.step_env(a); total += r;
            if env.done { ok = r > 0.0; break; }
            a = pick_greedy(engine, &env.perceive());
        }
        if ok { successes += 1; }
    }
    successes as f64 / TEST_EPS as f64 * 100.0
}

fn test_clean_phase1(cbl: &Cerebellum, dim: usize) -> f64 {
    let mut successes = 0u64;
    for _ in 0..TEST_EPS {
        let mut env = GridEnv5x5::new(); env.reset();
        let mut a = pick_greedy_phase1(cbl, &env.perceive(), dim);
        let mut ok = false;
        while !env.done {
            let r = env.step_env(a);
            if env.done { ok = r > 0.0; break; }
            a = pick_greedy_phase1(cbl, &env.perceive(), dim);
        }
        if ok { successes += 1; }
    }
    successes as f64 / TEST_EPS as f64 * 100.0
}

// ── Phase 1 #8 ──
fn run_phase1_seed(bfs_pot: &[Vec<f64>]) -> (f64, f64) {
    let mut cbl = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05; cbl.replay_only = false;

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cbl.epsilon = 0.8*remain+0.01; cbl.noise_std = 0.3*remain+0.01;

        let mut env = GridEnv5x5::new(); env.reset();
        let mut total = 0.0; let mut ok = false;
        cbl.reset_trace();

        let p = env.perceive();
        let mut logits = cbl.forward_logits(&p);
        let mut rng = rand::thread_rng();
        let exploring = cbl.noise_std > 0.0;
        let act = if exploring && rand::random::<f64>() < cbl.epsilon { rng.gen_range(0..N_ACTIONS) } else {
            if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cbl.noise_std..cbl.noise_std); } }
            logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
        };
        cbl.mark(&p, act);
        let mut action = act;
        let mut prev_state = p;
        let mut prev_pot = bfs_pot[env.agent.0][env.agent.1];

        while !env.done {
            let reward = env.step_env(action); total += reward;
            if env.done { ok = reward > 0.0;
                let p = env.perceive();
                let next_pot = bfs_pot[env.agent.0][env.agent.1];
                let rl_signal = reward + 0.99*next_pot - prev_pot;
                cbl.forward_logits(&p); cbl.reinforce_td(rl_signal, 0.99);
                cbl.store_transition(&prev_state, action, rl_signal, &p, true);
                break;
            }
            let p = env.perceive();
            let next_pot = bfs_pot[env.agent.0][env.agent.1];
            let rl_signal = reward + 0.99*next_pot - prev_pot;
            cbl.forward_logits(&p); cbl.reinforce_td(rl_signal, 0.99);
            cbl.decay_trace(0.99, 0.98);
            cbl.store_transition(&prev_state, action, rl_signal, &p, false);
            let mut logits = cbl.forward_logits(&p);
            action = if exploring && rand::random::<f64>() < cbl.epsilon { rng.gen_range(0..N_ACTIONS) } else {
                if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cbl.noise_std..cbl.noise_std); } }
                logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
            };
            cbl.mark(&p, action);
            prev_state = p; prev_pot = next_pot;
        }
        if ep % 100 == 0 && cbl.replay.len() >= 64 { cbl.replay_train(64, 0.95, 10); }
    }
    // Replay final — s'assure que tout le buffer est entraîné
    if cbl.replay.len() >= 64 { cbl.replay_train(64, 0.95, cbl.replay.len()/64); }

    let test_sr = test_clean_phase1(&cbl, 6);
    (test_sr, 0.0)
}

// ── S1 ──
fn run_s1_seed(bfs_pot: &[Vec<f64>], replay_only: bool, delta_clip: f64) -> (f64, f64) {
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05; engine.cerebellum.replay_only = replay_only;
    engine.cerebellum.delta_clip = delta_clip;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8*remain+0.01;
        engine.cerebellum.noise_std = 0.3*remain+0.01;

        let mut env = GridEnv5x5::new(); env.reset();
        engine.end_episode();
        engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
        engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
        let p = env.perceive();
        let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
        let mut a = engine.step(&p, 0.0, bv, &[]);
        while !env.done {
            let r = env.step_env(a);
            if env.done { let pt=env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
            engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
            engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
            let pt=env.perceive(); a=engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
        }
        engine.end_episode();
        if ep % 100 == 0 && engine.cerebellum.replay.len() >= 64 {
            engine.cerebellum.replay_train(64, 0.95, 10);
        }
    }
    if engine.cerebellum.replay.len() >= 64 {
        engine.cerebellum.replay_train(64, 0.95, engine.cerebellum.replay.len()/64);
    }

    let test_sr = test_clean(&engine);
    (test_sr, 0.0)
}

fn stats(name: &str, results: &[f64]) {
    let mut sorted = results.to_vec();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let med = sorted[N_SEEDS/2];
    let q1 = sorted[N_SEEDS/4];
    let q3 = sorted[3*N_SEEDS/4];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let min = sorted[0];
    let max = sorted[N_SEEDS-1];
    eprintln!("  {:<35}  mean={:5.0}%  med={:5.0}%  q1={:5.0}  q3={:5.0}  min={:5.0}  max={:5.0}  range={:5.0}",
        name, mean, med, q1, q3, min, max, max-min);
}

fn main() {
    let bfs_pot = compute_bfs_potential();
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  MATRICE MULTI-SEEDS — {N_SEEDS} seeds par config                    ║");
    eprintln!("║  Phase 1 #8 (dim=6) vs S1 vs S1+replay_only vs S1+δ-clip(1.0)     ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Phase 1 #8 ──
    eprintln!("── Phase 1 #8 (dim=6, hd=4, en ligne+replay) ──");
    let mut p1_results = vec![];
    for s in 0..N_SEEDS {
        let (test_sr, _) = run_phase1_seed(&bfs_pot);
        p1_results.push(test_sr);
        eprintln!("  seed {}: test succ={:.0}%", s, test_sr);
    }
    stats("Phase 1 #8", &p1_results);
    eprintln!();

    // ── S1 ──
    eprintln!("── S1 (TsoEngine, en ligne+replay) ──");
    let mut s1_results = vec![];
    for s in 0..N_SEEDS {
        let (test_sr, _) = run_s1_seed(&bfs_pot, false, 0.0);
        s1_results.push(test_sr);
        eprintln!("  seed {}: test succ={:.0}%", s, test_sr);
    }
    stats("S1", &s1_results);
    eprintln!();

    // ── S1 + replay_only ──
    eprintln!("── S1 + replay_only ──");
    let mut s1_ro_results = vec![];
    for s in 0..N_SEEDS {
        let (test_sr, _) = run_s1_seed(&bfs_pot, true, 0.0);
        s1_ro_results.push(test_sr);
        eprintln!("  seed {}: test succ={:.0}%", s, test_sr);
    }
    stats("S1+replay_only", &s1_ro_results);
    eprintln!();

    // ── S1 + δ-clip(1.0) ──
    eprintln!("── S1 + δ-clip(1.0) ──");
    let mut s1_clip_results = vec![];
    for s in 0..N_SEEDS {
        let (test_sr, _) = run_s1_seed(&bfs_pot, false, 1.0);
        s1_clip_results.push(test_sr);
        eprintln!("  seed {}: test succ={:.0}%", s, test_sr);
    }
    stats("S1+δ-clip(1.0)", &s1_clip_results);
    eprintln!();

    // ── Résumé ──
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSUMÉ — médiane [Q1,Q3] (min–max)                               ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    fn fmt_summary(results: &[f64]) -> String {
        let mut s = results.to_vec();
        s.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let med = s[N_SEEDS/2];
        let q1 = s[N_SEEDS/4];
        let q3 = s[3*N_SEEDS/4];
        let min = s[0];
        let max = s[N_SEEDS-1];
        format!("{:.0} [{:.0},{:.0}] ({:.0}–{:.0})", med, q1, q3, min, max)
    }
    eprintln!("║  Phase 1 #8         {:>30} ║", fmt_summary(&p1_results));
    eprintln!("║  S1 (online+replay) {:>30} ║", fmt_summary(&s1_results));
    eprintln!("║  S1+replay_only     {:>30} ║", fmt_summary(&s1_ro_results));
    eprintln!("║  S1+δ-clip(1.0)     {:>30} ║", fmt_summary(&s1_clip_results));
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
}
