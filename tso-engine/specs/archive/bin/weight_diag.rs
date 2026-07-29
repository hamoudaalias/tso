#![allow(dead_code, unused_assignments, unused_variables)]
/// ════════════════════════════════════════════════════════════════════════════
///  Diagnostic ultra-ciblé : les poids du cerebellum bougent-ils en S1 ?
///
///  On lit w2[0][0], w2[1][0], w2[2][0], w2[3][0] via get_out_weight
///  avant et après chaque épisode d'entraînement, et après le replay.
///
///  Si les poids changent entre avant/après l'épisode mais pas entre
///  phase1 et S1, le problème est ailleurs (replay qui écrase, ou
///  soft_normalize_row).
///  Si les poids ne changent pas du tout en S1, le problème est dans
///  reinforce_td ou l'ordre des appels.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

const N_EPISODES: usize = 10;

struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool, dim: usize }
impl GridEnv5x5 {
    fn new(dim: usize) -> Self { GridEnv5x5 { agent: (2,2), step:0, done:false, dim } }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..W); let y = rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x,y)) && !FOOD_POSITIONS.contains(&(x,y)) {
                self.agent=(x,y); break;
            }
        }
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
        match self.dim {
            4 => Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)]),
            6 => Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws]),
            _ => Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)]),
        }
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

fn dump_w2(cbl: &Cerebellum, label: &str) {
    let _w2_00 = cbl.get_out_weight(0, 0);
    let _w2_10 = cbl.get_out_weight(1, 0);
    let w2_20 = cbl.get_out_weight(2, 0);
    let w2_30 = cbl.get_out_weight(3, 0);
    let w2_03 = cbl.get_out_weight(0, 3);
    let w2_13 = cbl.get_out_weight(1, 3);
    let w2_23 = cbl.get_out_weight(2, 3);
    let w2_33 = cbl.get_out_weight(3, 3);
    let norm0 = (cbl.get_out_weight(0,0).powi(2) + cbl.get_out_weight(0,1).powi(2)
        + cbl.get_out_weight(0,2).powi(2) + cbl.get_out_weight(0,3).powi(2)).sqrt();
    let norm1 = (cbl.get_out_weight(1,0).powi(2) + cbl.get_out_weight(1,1).powi(2)
        + cbl.get_out_weight(1,2).powi(2) + cbl.get_out_weight(1,3).powi(2)).sqrt();
    eprintln!("  [{label}] w2[0][0..3] = {:.12} {:.12} {:.12} {:.12}  ||w2[a=0]||={:.12}",
        cbl.get_out_weight(0,0), cbl.get_out_weight(0,1), cbl.get_out_weight(0,2), cbl.get_out_weight(0,3), norm0);
    eprintln!("  [{label}] w2[1][0..3] = {:.12} {:.12} {:.12} {:.12}  ||w2[a=1]||={:.12}",
        cbl.get_out_weight(1,0), cbl.get_out_weight(1,1), cbl.get_out_weight(1,2), cbl.get_out_weight(1,3), norm1);
    eprintln!("  [{label}] w2[2][0]={:.12} w2[3][0]={:.12}", w2_20, w2_30);
    eprintln!("  [{label}] w2[0][3]={:.12} w2[1][3]={:.12} w2[2][3]={:.12} w2[3][3]={:.12}",
        w2_03, w2_13, w2_23, w2_33);
}

fn diagnose_phase1(bfs_pot: &[Vec<f64>]) {
    eprintln!("\n═══ PHASE 1 #8 (dim=6, référence) ═══\n");
    let mut cbl = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05;

    eprintln!("Poids avant entraînement:");
    dump_w2(&cbl, "phase1/init");

    for ep in 1..=N_EPISODES {
        let remain = (N_EPISODES - ep).max(0) as f64 / N_EPISODES as f64;
        cbl.epsilon = 0.8*remain+0.01; cbl.noise_std = 0.3*remain+0.01;

        let (total, ok) = {
            let mut env = GridEnv5x5::new(6); env.reset();
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
            if cbl.replay.len() >= 64 { cbl.replay_train(64, 0.95, 10); }
            (total, ok)
        };
        eprintln!("  ép {ep}/{N_EPISODES} reward={total:.2} ok={ok} replay={}",
            cbl.replay.len());
        dump_w2(&cbl, "phase1/after_ep");
    }

    // Test ε=0
    cbl.epsilon = 0.0; cbl.noise_std = 0.0;
    let mut successes = 0;
    for _ in 0..10 {
        let mut env = GridEnv5x5::new(6); env.reset();
        let mut total = 0.0;
        cbl.reset_trace();
        let p = env.perceive();
        let mut logits = cbl.forward_logits(&p);
        let act = logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
        cbl.mark(&p, act);
        let mut action = act;
        let mut prev_pot = bfs_pot[env.agent.0][env.agent.1];
        while !env.done {
            let reward = env.step_env(action); total += reward;
            if env.done {
                if reward > 0.0 { successes += 1; }
                break;
            }
            let p = env.perceive();
            let next_pot = bfs_pot[env.agent.0][env.agent.1];
            let _rl_signal = reward + 0.99*next_pot - prev_pot;
            logits = cbl.forward_logits(&p);
            // No update during test
            action = logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
            cbl.mark(&p, action);
            prev_pot = next_pot;
        }
    }
    eprintln!("  TEST ε=0: {successes}/10 succès");
    dump_w2(&cbl, "phase1/final");
}

fn diagnose_s1(bfs_pot: &[Vec<f64>]) {
    eprintln!("\n═══ S1 (TsoEngine complet) ═══\n");
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05; engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0; engine.use_stationary_reward = true;

    eprintln!("Poids avant entraînement:");
    dump_w2(&engine.cerebellum, "S1/init");

    for ep in 1..=N_EPISODES {
        let remain = (N_EPISODES - ep).max(0) as f64 / N_EPISODES as f64;
        engine.cerebellum.epsilon = 0.8*remain+0.01;
        engine.cerebellum.noise_std = 0.3*remain+0.01;

        let (total, ok) = {
            let mut env = GridEnv5x5::new(6); env.reset();
            engine.end_episode();
            engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
            engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
            let mut total=0.0; let mut ok=false;
            let p = env.perceive();
            let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
            let mut a = engine.step(&p, 0.0, bv, &[]);
            while !env.done {
                let r = env.step_env(a); total += r;
                if env.done { ok=r>0.0;
                    let pt=env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
                engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
                engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
                let pt=env.perceive(); a=engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
            }
            engine.end_episode();
            if engine.cerebellum.replay.len()>=64 { engine.cerebellum.replay_train(64, 0.95, 10); }
            (total, ok)
        };
        eprintln!("  ép {ep}/{N_EPISODES} reward={total:.2} ok={ok} replay={}",
            engine.cerebellum.replay.len());
        dump_w2(&engine.cerebellum, "S1/after_ep");
    }

    // Test ε=0
    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut successes = 0;
    for _ in 0..10 {
        let mut env = GridEnv5x5::new(6); env.reset();
        engine.end_episode();
        engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
        engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
        let mut total=0.0;
        let p = env.perceive();
        let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
        let mut a = engine.step(&p, 0.0, bv, &[]);
        while !env.done {
            let r = env.step_env(a); total += r;
            if env.done {
                if r > 0.0 { successes += 1; }
                break;
            }
            engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
            engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
            let pt=env.perceive();
            a=engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
        }
    }
    eprintln!("  TEST ε=0: {successes}/10 succès");
    dump_w2(&engine.cerebellum, "S1/final");
}

fn main() {
    let bfs_pot = compute_bfs_potential();
    diagnose_phase1(&bfs_pot);
    diagnose_s1(&bfs_pot);
}
