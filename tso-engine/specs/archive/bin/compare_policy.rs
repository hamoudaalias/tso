#![allow(dead_code, unused_assignments, unused_variables)]
/// ════════════════════════════════════════════════════════════════════════════
///  Comparaison : dump des poids du cerebellum après entraînement
///
///  Pour chaque configuration (Phase 1 #8, S1), on dump la distribution
///  des logits sur toutes les positions de la grille → politique apprise.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6;
const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

const TRAIN_EPS: usize = 500;
const TEST_EPS: usize = 100;

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

fn optimal_policy(bfs_pot: &[Vec<f64>]) -> [[usize; H]; W] {
    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; }
        let mut best_a = 0; let mut best_v = bfs_pot[x][y];
        for (a, (dx, dy)) in [(0,-1),(0,1),(-1,0),(1,0)].iter().enumerate() {
            let nx = x as isize + dx; let ny = y as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let v = bfs_pot[nx as usize][ny as usize];
                if v > best_v { best_v = v; best_a = a; }
            }
        }
        policy[x][y] = best_a;
    }}
    policy
}

fn print_policy(policy: &[[usize; H]; W], title: &str) {
    eprintln!("── {title} ──");
    for y in 0..H {
        eprint!("  Row {y} ");
        for x in 0..W {
            let c = match policy[x][y] { 0=>'N', 1=>'S', 2=>'W', 3=>'E', 4=>'░', _=>'?' };
            eprint!("  {c}");
        }
        eprintln!();
    }
    eprintln!();
}

fn policy_stats(policy: &[[usize; H]; W], opt: &[[usize; H]; W]) {
    let mut correct=0; let mut total=0; let mut ac=[0u64;4];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { continue; }
        total+=1; ac[policy[x][y]]+=1;
        if policy[x][y]==opt[x][y] { correct+=1; }
    }}
    eprintln!("  Correct={correct}/{total} ({:.0}%)  N={} S={} W={} E={}",
        correct as f64/total as f64*100.0, ac[0], ac[1], ac[2], ac[3]);
}

fn dump_policy_from_weights(dim: usize, hd: usize, w1: &[Vec<f64>], b1: &[f64], w2: &[Vec<f64>], b2: &[f64]) -> [[usize; H]; W] {
    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y]=4; continue; }
        let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let (fs, ws) = if dim>=6 {
            let _f=0.0; let _w=0.0; // food & water senses from proximity
            // We approximate: since water positions are fixed, we compute actual proximity
            let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
                let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
                if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
            let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
                let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
                if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
            (fs, ws)
        } else { (0.0, 0.0) };
        let p = match dim {
            4 => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)],
            6 => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws],
            _ => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)],
        };
        // Manual forward through RUST network: h_j = tanh(∑w1[j][i]*p[i] + b1[j])
        let mut h = vec![0.0; hd];
        for j in 0..hd {
            let mut s = b1[j];
            for i in 0..dim { s += w1[j][i] * p[i]; }
            h[j] = s.tanh();
        }
        // logits: out[a] = ∑w2[a][j]*h[j] + b2[a]
        let mut logits = vec![0.0; N_ACTIONS];
        for a in 0..N_ACTIONS {
            logits[a] = b2[a];
            for j in 0..hd { logits[a] += w2[a][j] * h[j]; }
        }
        policy[x][y] = logits.iter().enumerate()
            .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
    }}
    policy
}

fn extract_weights_phase1(cbl: &Cerebellum, dim: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let hd = 4;
    let mut w1 = vec![vec![0.0; dim]; hd];
    let mut w2 = vec![vec![0.0; hd]; N_ACTIONS];
    for j in 0..hd {
        for i in 0..dim { w1[j][i] = cbl.get_hidden_weight(j, i); }
    }
    for a in 0..N_ACTIONS {
        for j in 0..hd { w2[a][j] = cbl.get_out_weight(a, j); }
    }
    // Biases NOT accessible via public API → use 0
    let b1 = vec![0.0; hd];
    let b2 = vec![0.0; N_ACTIONS];
    (w1, b1, w2, b2)
}

fn extract_weights_tsos1(engine: &TsoEngine, dim: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let hd = 4;
    let mut w1 = vec![vec![0.0; dim]; hd];
    let mut w2 = vec![vec![0.0; hd]; N_ACTIONS];
    for j in 0..hd {
        for i in 0..dim { w1[j][i] = engine.cerebellum.get_hidden_weight(j, i); }
    }
    for a in 0..N_ACTIONS {
        for j in 0..hd { w2[a][j] = engine.cerebellum.get_out_weight(a, j); }
    }
    let b1 = vec![0.0; hd];
    let b2 = vec![0.0; N_ACTIONS];
    (w1, b1, w2, b2)
}

/// ═══ Phase 1 #8 ═══
fn run_phase1(bfs_pot: &[Vec<f64>]) {
    eprintln!("\n╔══ Phase 1 #8 — dim=4 hd=4 ══╗\n");
    let mut cbl = Cerebellum::new(4, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05; cbl.replay_only = false;

    let opt = optimal_policy(bfs_pot);
    print_policy(&opt, "OPTIMALE");

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cbl.epsilon = 0.8*remain+0.01; cbl.noise_std = 0.3*remain+0.01;
        let (total,ok) = run_ep_phase1(&mut cbl, 4, bfs_pot);
        tr.push(total); ts.push(ok);
        if ep%100==0 {
            let sr = ts[ep-100..].iter().filter(|&&s|s).count() as f64;
            eprintln!("  ép {ep}/{TRAIN_EPS} avg_last100={:.1} succ={:.0}% replay={}",
                tr[ep-100..].iter().sum::<f64>()/100.0, sr, cbl.replay.len());
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tr_avg = tr.iter().sum::<f64>()/TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>()/200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64/TRAIN_EPS as f64*100.0;
    eprintln!("  TRAIN avg={tr_avg:.1} last200={tr_l200:.1} succ={tr_sr:.0}% {elapsed:.1}s\n");

    cbl.epsilon = 0.0; cbl.noise_std = 0.0;
    let mut er: Vec<f64>=vec![]; let mut es: Vec<bool>=vec![];
    for _ in 0..TEST_EPS { let (t,s)=run_ep_phase1(&mut cbl,4,bfs_pot); er.push(t); es.push(s); }
    let ea = er.iter().sum::<f64>()/TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64/TEST_EPS as f64*100.0;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={esr:.0}%");

    let (w1,b1,w2,b2) = extract_weights_phase1(&cbl, 4);
    let pol = dump_policy_from_weights(4, 4, &w1, &b1, &w2, &b2);
    print_policy(&pol, "POLITIQUE Phase 1 #8");
    policy_stats(&pol, &opt);

    // Dump weights
    eprintln!("  w2:");
    for a in 0..N_ACTIONS {
        eprintln!("    a{a}: {:+.3} {:+.3} {:+.3} {:+.3}",
            w2[a][0], w2[a][1], w2[a][2], w2[a][3]);
    }
    // Hidden layer norms per action
    for a in 0..N_ACTIONS {
        let norm: f64 = w2[a].iter().map(|x| x*x).sum::<f64>().sqrt();
        eprintln!("    ||w2[a={a}]||={norm:.3}");
    }
}

fn run_ep_phase1(cbl: &mut Cerebellum, dim: usize, bfs_pot: &[Vec<f64>]) -> (f64, bool) {
    let mut env = GridEnv5x5::new(dim); env.reset();
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
}

/// ═══ Phase 1 #8 avec dim=6 ═══
fn run_phase1_dim6(bfs_pot: &[Vec<f64>]) {
    eprintln!("\n╔══ Phase 1 #8 — dim=6 hd=4 (contrôle entrée 6D) ══╗\n");
    let mut cbl = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05; cbl.replay_only = false;

    let opt = optimal_policy(bfs_pot);

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cbl.epsilon = 0.8*remain+0.01; cbl.noise_std = 0.3*remain+0.01;
        let (total,ok) = run_ep_phase1(&mut cbl, 6, bfs_pot);
        tr.push(total); ts.push(ok);
        if ep%100==0 {
            eprintln!("  ép {ep}/{TRAIN_EPS} avg_last100={:.1} succ={:.0}% replay={}",
                tr[ep-100..].iter().sum::<f64>()/100.0,
                ts[ep-100..].iter().filter(|&&s|s).count() as f64, cbl.replay.len());
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tr_avg = tr.iter().sum::<f64>()/TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>()/200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64/TRAIN_EPS as f64*100.0;
    eprintln!("  TRAIN avg={tr_avg:.1} last200={tr_l200:.1} succ={tr_sr:.0}% {elapsed:.1}s\n");

    cbl.epsilon = 0.0; cbl.noise_std = 0.0;
    let mut er: Vec<f64>=vec![]; let mut es: Vec<bool>=vec![];
    for _ in 0..TEST_EPS { let (t,s)=run_ep_phase1(&mut cbl,6,bfs_pot); er.push(t); es.push(s); }
    let ea = er.iter().sum::<f64>()/TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64/TEST_EPS as f64*100.0;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={esr:.0}%");

    let (w1,b1,w2,b2) = extract_weights_phase1(&cbl, 6);
    let pol = dump_policy_from_weights(6, 4, &w1, &b1, &w2, &b2);
    print_policy(&pol, "POLITIQUE Phase 1 #8 dim=6");
    policy_stats(&pol, &opt);

    eprintln!("  w2:");
    for a in 0..N_ACTIONS {
        eprintln!("    a{a}: {:+.3} {:+.3} {:+.3} {:+.3}",
            w2[a][0], w2[a][1], w2[a][2], w2[a][3]);
    }
    for a in 0..N_ACTIONS {
        let norm: f64 = w2[a].iter().map(|x| x*x).sum::<f64>().sqrt();
        eprintln!("    ||w2[a={a}]||={norm:.3}");
    }
}

/// ═══ S1 ═══
fn run_s1(bfs_pot: &[Vec<f64>]) {
    eprintln!("\n╔══ Phase 1c S1 — TsoEngine use_stationary_reward=true ══╗\n");
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05; engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0; engine.use_stationary_reward = true;

    let opt = optimal_policy(bfs_pot);

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8*remain+0.01;
        engine.cerebellum.noise_std = 0.3*remain+0.01;
        let (total,ok) = run_ep_s1(&mut engine, bfs_pot);
        tr.push(total); ts.push(ok);
        if ep%100==0 {
            eprintln!("  ép {ep}/{TRAIN_EPS} avg_last100={:.1} succ={:.0}% replay={} C={}",
                tr[ep-100..].iter().sum::<f64>()/100.0,
                ts[ep-100..].iter().filter(|&&s|s).count() as f64,
                engine.cerebellum.replay.len(), engine.num_concepts());
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tr_avg = tr.iter().sum::<f64>()/TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>()/200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64/TRAIN_EPS as f64*100.0;
    eprintln!("  TRAIN avg={tr_avg:.1} last200={tr_l200:.1} succ={tr_sr:.0}% {elapsed:.1}s\n");

    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut er: Vec<f64>=vec![]; let mut es: Vec<bool>=vec![];
    for _ in 0..TEST_EPS { let (t,s)=run_ep_s1(&mut engine, bfs_pot); er.push(t); es.push(s); }
    let ea = er.iter().sum::<f64>()/TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64/TEST_EPS as f64*100.0;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={esr:.0}%");
    eprintln!("  C={} E={} Φ={:.3} Replay={}",
        engine.num_concepts(), engine.graph.edges.len(), engine.current_phi, engine.cerebellum.replay.len());

    let (w1,b1,w2,b2) = extract_weights_tsos1(&engine, PERCEPTION_DIM);
    let pol = dump_policy_from_weights(PERCEPTION_DIM, 4, &w1, &b1, &w2, &b2);
    print_policy(&pol, "POLITIQUE S1");
    policy_stats(&pol, &opt);

    eprintln!("  w2:");
    for a in 0..N_ACTIONS {
        eprintln!("    a{a}: {:+.3} {:+.3} {:+.3} {:+.3}",
            w2[a][0], w2[a][1], w2[a][2], w2[a][3]);
    }
    for a in 0..N_ACTIONS {
        let norm: f64 = w2[a].iter().map(|x| x*x).sum::<f64>().sqrt();
        eprintln!("    ||w2[a={a}]||={norm:.3}");
    }
}

fn run_ep_s1(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>]) -> (f64, bool) {
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
        if env.done { ok=r>0.0; let pt=env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
        engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
        engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;
        let pt=env.perceive(); a=engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    if engine.cerebellum.replay.len()>=64 { engine.cerebellum.replay_train(64, 0.95, 10); }
    (total, ok)
}

fn main() {
    let bfs_pot = compute_bfs_potential();
    run_phase1(&bfs_pot);
    run_phase1_dim6(&bfs_pot);
    run_s1(&bfs_pot);
    eprintln!("\n═══════ FIN ═══════");
}
