/// ════════════════════════════════════════════════════════════════════════════
///  Comparaison : politique apprise en Phase 1 #8 vs S1
///
///  Phase 1 #8 = cerebellum MLP seul (shaping BFS, replay, pas de TSO)
///  S1 = TsoEngine complet avec use_stationary_reward=true
///
///  Si Phase 1 #8 produit une politique diverse (plusieurs actions) et
///  S1 une politique collapsée (toute la même action), la différence
///  est ce que le TSO ajoute — pas l'algorithme d'apprentissage.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;

const W: usize = 5; const H: usize = 5;
const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

const TRAIN_EPS: usize = 500;
const TEST_EPS: usize = 100;

// ─── Environnement 5×5 avec perception configurable ────────────────────────
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
    fn perceive(&self) -> Vec<f64> {
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
            4 => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)],
            6 => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws],
            _ => vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)],
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

/// Politique optimale (suivre gradient BFS)
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

fn print_policy_grid(policy: &[[usize; H]; W], title: &str) {
    eprintln!("── {} ──", title);
    for y in 0..H {
        eprint!("  Row {} ", y);
        for x in 0..W {
            let c = match policy[x][y] {
                0 => 'N', 1 => 'S', 2 => 'W', 3 => 'E',
                4 => '░',
                _ => '?',
            };
            eprint!("  {}", c);
        }
        eprintln!();
    }
    eprintln!();
}

/// ============================================================
///  Phase 1 #8 : Cerebellum seul
/// ============================================================
fn run_phase1(bfs_pot: &[Vec<f64>]) {
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PHASE 1 #8 — Cervelet seul, dim=4, hd=4, shaping BFS, replay     ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut cbl = Cerebellum::new(4, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05; cbl.replay_only = false;

    let opt = optimal_policy(bfs_pot);
    print_policy_grid(&opt, "POLITIQUE OPTIMALE");

    // ── Entraînement ──
    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cbl.epsilon = 0.8 * remain + 0.01; cbl.noise_std = 0.3 * remain + 0.01;

        let (total, ok) = run_ep_phase1(&mut cbl, bfs_pot);
        tr.push(total); ts.push(ok);

        if ep % 100 == 0 {
            eprintln!("  [phase1] ép={}/{} avg_last100={:.1} succ={:.0}% replay={}",
                ep, TRAIN_EPS,
                tr[ep-100..].iter().sum::<f64>() / 100.0,
                ts[ep-100..].iter().filter(|&&s|s).count() as f64 * 100.0,
                cbl.replay.len());
        }
    }
    let elapsed = t0.elapsed();
    let tr_avg = tr.iter().sum::<f64>() / TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>() / 200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64 / TRAIN_EPS as f64;
    eprintln!("  TRAIN {TRAIN_EPS}eps {elapsed:.1}s avg={tr_avg:.1} last200={tr_l200:.1} succ={:.0}%",
        tr_sr*100.0);

    // ── Test ──
    cbl.epsilon = 0.0; cbl.noise_std = 0.0;
    let mut er: Vec<f64> = vec![]; let mut es: Vec<bool> = vec![];
    for _ in 0..TEST_EPS {
        let (total, ok) = run_ep_phase1(&mut cbl, bfs_pot);
        er.push(total); es.push(ok);
    }
    let ea = er.iter().sum::<f64>() / TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64 / TEST_EPS as f64;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={:.0}%", esr*100.0);

    // ── Dump politique ──
    dump_policy_phase1(&cbl, bfs_pot, "POLITIQUE APPRISE Phase 1 #8");

    // Distribution actions
    let mut ac = [0u64; 4];
    // On dummy-dump vite fait
    eprintln!();
}

fn run_ep_phase1(cbl: &mut Cerebellum, bfs_pot: &[Vec<f64>]) -> (f64, bool) {
    let mut env = GridEnv5x5::new(4); env.reset();
    let mut total = 0.0; let mut ok = false;
    cbl.reset_trace();

    let p = Array1::from_vec(env.perceive());
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
        if env.done {
            ok = reward > 0.0;
            let p = Array1::from_vec(env.perceive());
            let next_pot = bfs_pot[env.agent.0][env.agent.1];
            let shaping = 0.99 * next_pot - prev_pot;
            let rl_signal = reward + shaping;
            cbl.forward_logits(&p);
            cbl.reinforce_td(rl_signal, 0.99);
            cbl.store_transition(&prev_state, action, rl_signal, &p, true);
            break;
        }
        let p = Array1::from_vec(env.perceive());
        let next_pot = bfs_pot[env.agent.0][env.agent.1];
        let shaping = 0.99 * next_pot - prev_pot;
        let rl_signal = reward + shaping;

        cbl.forward_logits(&p);
        cbl.reinforce_td(rl_signal, 0.99);
        cbl.decay_trace(0.99, 0.98);
        cbl.store_transition(&prev_state, action, rl_signal, &p, false);

        let mut logits = cbl.forward_logits(&p);
        let exploring = cbl.noise_std > 0.0;
        action = if exploring && rand::random::<f64>() < cbl.epsilon { rng.gen_range(0..N_ACTIONS) } else {
            if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cbl.noise_std..cbl.noise_std); } }
            logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
        };
        cbl.mark(&p, action);
        prev_state = p;
        prev_pot = next_pot;
    }
    if cbl.replay.len() >= 64 { cbl.replay_train(64, 0.95, 10); }
    (total, ok)
}

fn dump_policy_phase1(cbl: &Cerebellum, bfs_pot: &[Vec<f64>], title: &str) {
    // On crée un clone du cerebellum (sans traces) pour le dump
    let mut cbl2 = Cerebellum::new(4, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    // Copier les poids
    for j in 0..cbl2.hidden_dim {
        for i in 0..4 { cbl2.w1[j][i] = cbl.get_hidden_weight(j, i); }
    }
    for a in 0..N_ACTIONS {
        for j in 0..cbl2.hidden_dim { cbl2.w2[a][j] = cbl.get_out_weight(a, j); }
    }
    // Copier critic
    // Récupération via... pas d'accès direct. On fait sans critic pour le dump.

    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; }
        let env = GridEnv5x5::new(4);
        let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let p_vec = vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0)];
        let p = Array1::from_vec(p_vec);
        let logits = cbl2.forward_logits(&p);
        let best = logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
        policy[x][y] = best;
    }}

    print_policy_grid(&policy, title);

    let mut correct = 0; let mut total = 0;
    let mut ac = [0u64; 4];
    let opt = optimal_policy(bfs_pot);
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { continue; }
        total += 1;
        if policy[x][y] == opt[x][y] { correct += 1; }
        ac[policy[x][y]] += 1;
    }}
    eprintln!("  Précision vs opt: {correct}/{total} ({:.0}%)  Actions: N={} S={} W={} E={}",
        correct as f64 / total as f64 * 100.0, ac[0], ac[1], ac[2], ac[3]);
}

/// ============================================================
///  Phase 1c S1 : TsoEngine complet
/// ============================================================
fn run_s1(bfs_pot: &[Vec<f64>]) {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  S1 — TsoEngine complet, use_stationary_reward=true               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut engine = TsoEngine::with_hidden(6, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05; engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0; engine.use_stationary_reward = true;

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;

        let (total, ok) = run_ep_s1(&mut engine, bfs_pot);
        tr.push(total); ts.push(ok);

        if ep % 100 == 0 {
            eprintln!("  [S1] ép={}/{} avg_last100={:.1} succ={:.0}% replay={} C={}",
                ep, TRAIN_EPS,
                tr[ep-100..].iter().sum::<f64>() / 100.0,
                ts[ep-100..].iter().filter(|&&s|s).count() as f64 * 100.0,
                engine.cerebellum.replay.len(), engine.num_concepts());
        }
    }
    let elapsed = t0.elapsed();
    let tr_avg = tr.iter().sum::<f64>() / TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>() / 200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64 / TRAIN_EPS as f64;
    eprintln!("  TRAIN {TRAIN_EPS}eps {elapsed:.1}s avg={tr_avg:.1} last200={tr_l200:.1} succ={:.0}%",
        tr_sr*100.0);

    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut er: Vec<f64> = vec![]; let mut es: Vec<bool> = vec![];
    for _ in 0..TEST_EPS {
        let (total, ok) = run_ep_s1(&mut engine, bfs_pot);
        er.push(total); es.push(ok);
    }
    let ea = er.iter().sum::<f64>() / TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64 / TEST_EPS as f64;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={:.0}%", esr*100.0);
    eprintln!("  Final: C={} E={} Φ={:.3} Replay={}",
        engine.num_concepts(), engine.graph.edges.len(), engine.current_phi, engine.cerebellum.replay.len());

    dump_policy_s1(&engine, bfs_pot, "POLITIQUE APPRISE S1");
}

fn run_ep_s1(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>]) -> (f64, bool) {
    let mut env = GridEnv5x5::new(6); env.reset();
    engine.end_episode();
    engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
    engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;

    let mut total = 0.0; let mut ok = false;
    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a); total += r;
        if env.done { ok = r > 0.0;
            let pt = env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
        engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    if engine.cerebellum.replay.len() >= 64 { engine.cerebellum.replay_train(64, 0.95, 10); }
    (total, ok)
}

fn dump_policy_s1(engine: &TsoEngine, bfs_pot: &[Vec<f64>], title: &str) {
    // La seule façon de dump sans mut : créer un cervelet copie
    let mut cbl2 = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    for j in 0..cbl2.hidden_dim {
        for i in 0..6 { cbl2.w1[j][i] = engine.cerebellum.get_hidden_weight(j, i); }
    }
    for a in 0..N_ACTIONS {
        for j in 0..cbl2.hidden_dim { cbl2.w2[a][j] = engine.cerebellum.get_out_weight(a, j); }
    }

    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; }
        let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        let p_vec = vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws];
        let p = Array1::from_vec(p_vec);
        let logits = cbl2.forward_logits(&p);
        let best = logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
        policy[x][y] = best;
    }}

    print_policy_grid(&policy, title);

    let opt = optimal_policy(bfs_pot);
    let mut correct = 0; let mut total = 0;
    let mut ac = [0u64; 4];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { continue; }
        total += 1;
        if policy[x][y] == opt[x][y] { correct += 1; }
        ac[policy[x][y]] += 1;
    }}
    eprintln!("  Précision vs opt: {correct}/{total} ({:.0}%)  Actions: N={} S={} W={} E={}",
        correct as f64 / total as f64 * 100.0, ac[0], ac[1], ac[2], ac[3]);
}

/// ============================================================
///  Isolement : Phase 1 #8 avec dim=6 (même que S1)
/// ============================================================
fn run_phase1_dim6(bfs_pot: &[Vec<f64>]) {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  DIAG — Phase 1 #8 avec dim=6 (même entrée que S1)                ║");
    eprintln("║  Cervelet seul, mais entrée 6D (incl. food_sensed, water_sensed)   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut cbl = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    cbl.epsilon = 0.1; cbl.noise_std = 0.1; cbl.replay_lr = 0.05; cbl.replay_only = false;

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];
    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        cbl.epsilon = 0.8 * remain + 0.01; cbl.noise_std = 0.3 * remain + 0.01;
        let (total, ok) = run_ep_dim6(&mut cbl, bfs_pot);
        tr.push(total); ts.push(ok);
        if ep % 100 == 0 {
            eprintln!("  [diag/dim6] ép={}/{} avg_last100={:.1} succ={:.0}% replay={}",
                ep, TRAIN_EPS,
                tr[ep-100..].iter().sum::<f64>() / 100.0,
                ts[ep-100..].iter().filter(|&&s|s).count() as f64 * 100.0,
                cbl.replay.len());
        }
    }
    let elapsed = t0.elapsed();
    let tr_avg = tr.iter().sum::<f64>() / TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>() / 200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64 / TRAIN_EPS as f64;
    eprintln!("  TRAIN {TRAIN_EPS}eps {elapsed:.1}s avg={tr_avg:.1} last200={tr_l200:.1} succ={:.0}%", tr_sr*100.0);

    cbl.epsilon = 0.0; cbl.noise_std = 0.0;
    let mut er: Vec<f64> = vec![]; let mut es: Vec<bool> = vec![];
    for _ in 0..TEST_EPS {
        let (total, ok) = run_ep_dim6(&mut cbl, bfs_pot);
        er.push(total); es.push(ok);
    }
    let ea = er.iter().sum::<f64>() / TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64 / TEST_EPS as f64;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={:.0}%", esr*100.0);
    dump_policy_phase1_dim6(&cbl, bfs_pot, "POLITIQUE APPRISE DIAG dim=6");
}

fn run_ep_dim6(cbl: &mut Cerebellum, bfs_pot: &[Vec<f64>]) -> (f64, bool) {
    let mut env = GridEnv5x5::new(6); env.reset();
    let mut total = 0.0; let mut ok = false;
    cbl.reset_trace();

    let p = Array1::from_vec(env.perceive());
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
        if env.done {
            ok = reward > 0.0;
            let p = Array1::from_vec(env.perceive());
            let next_pot = bfs_pot[env.agent.0][env.agent.1];
            let shaping = 0.99 * next_pot - prev_pot;
            let rl_signal = reward + shaping;
            cbl.forward_logits(&p);
            cbl.reinforce_td(rl_signal, 0.99);
            cbl.store_transition(&prev_state, action, rl_signal, &p, true);
            break;
        }
        let p = Array1::from_vec(env.perceive());
        let next_pot = bfs_pot[env.agent.0][env.agent.1];
        let shaping = 0.99 * next_pot - prev_pot;
        let rl_signal = reward + shaping;

        cbl.forward_logits(&p);
        cbl.reinforce_td(rl_signal, 0.99);
        cbl.decay_trace(0.99, 0.98);
        cbl.store_transition(&prev_state, action, rl_signal, &p, false);

        let mut logits = cbl.forward_logits(&p);
        let exploring = cbl.noise_std > 0.0;
        action = if exploring && rand::random::<f64>() < cbl.epsilon { rng.gen_range(0..N_ACTIONS) } else {
            if exploring { for l in logits.iter_mut() { *l += rng.gen_range(-cbl.noise_std..cbl.noise_std); } }
            logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap()
        };
        cbl.mark(&p, action);
        prev_state = p;
        prev_pot = next_pot;
    }
    if cbl.replay.len() >= 64 { cbl.replay_train(64, 0.95, 10); }
    (total, ok)
}

fn dump_policy_phase1_dim6(cbl: &Cerebellum, bfs_pot: &[Vec<f64>], title: &str) {
    let mut cbl2 = Cerebellum::new(6, N_ACTIONS, 0.30, 0.1, 0.50, 4);
    for j in 0..cbl2.hidden_dim {
        for i in 0..6 { cbl2.w1[j][i] = cbl.get_hidden_weight(j, i); }
    }
    for a in 0..N_ACTIONS {
        for j in 0..cbl2.hidden_dim { cbl2.w2[a][j] = cbl.get_out_weight(a, j); }
    }

    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; }
        let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        let p_vec = vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws];
        let p = Array1::from_vec(p_vec);
        let logits = cbl2.forward_logits(&p);
        let best = logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
        policy[x][y] = best;
    }}

    print_policy_grid(&policy, title);

    let opt = optimal_policy(bfs_pot);
    let mut correct = 0; let mut total = 0;
    let mut ac = [0u64; 4];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { continue; }
        total += 1;
        if policy[x][y] == opt[x][y] { correct += 1; }
        ac[policy[x][y]] += 1;
    }}
    eprintln!("  Précision vs opt: {correct}/{total} ({:.0}%)  Actions: N={} S={} W={} E={}",
        correct as f64 / total as f64 * 100.0, ac[0], ac[1], ac[2], ac[3]);
}

// ─── Main ───────────────────────────────────────────────────────────────────
fn main() {
    let bfs_pot = compute_bfs_potential();

    // 1) Phase 1 #8 (référence, dim=4)
    run_phase1(&bfs_pot);

    // 2) Diagnostic : Phase 1 #8 avec dim=6 (comme S1)
    run_phase1_dim6(&bfs_pot);

    // 3) S1 réel
    run_s1(&bfs_pot);

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  COMPARAISON TERMINÉE                                                 ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
