/// ════════════════════════════════════════════════════════════════════════════
///  Vérification finale : replay_only=true en S1
///
///  Si replay_only=true empêche le collapse (remontée du test ε=0),
///  le coupable est l'interaction online+replay qui déstabilise l'actor.
///  Si replay_only ne change rien, le problème est ailleurs.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6; const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];
const TRAIN_EPS: usize = 500;
const TEST_EPS: usize = 100;

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

fn run_replay_only() {
    let bfs_pot = compute_bfs_potential();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  S1 avec replay_only=true → online TD DÉSACTIVÉ                    ║");
    eprintln!("║  L'apprentissage se fait uniquement via replay_train               ║");
    eprintln!("║                                                                     ║");
    eprintln!("║  Si TEST ≥ 80% → le online TD est le coupable du collapse           ║");
    eprintln!("║  Si TEST ~20% → le problème est ailleurs (replay mal formé)         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = true;  // ← LE FIX
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    let t0 = Instant::now();
    let mut tr: Vec<f64> = vec![]; let mut ts: Vec<bool> = vec![];

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8*remain+0.01;
        engine.cerebellum.noise_std = 0.3*remain+0.01;

        let mut env = GridEnv5x5::new(); env.reset();
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
        if engine.cerebellum.replay.len() >= 64 {
            engine.cerebellum.replay_train(64, 0.95, 10);
        }
        tr.push(total); ts.push(ok);

        if ep%100==0 {
            let sr = ts[ep-100..].iter().filter(|&&s|s).count() as f64;
            eprintln!("  [replay_only] ép {ep}/{TRAIN_EPS} avg_last100={:.1} succ={:.0}% replay={} C={}",
                tr[ep-100..].iter().sum::<f64>()/100.0, sr, engine.cerebellum.replay.len(), engine.num_concepts());
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let tr_avg = tr.iter().sum::<f64>()/TRAIN_EPS as f64;
    let tr_l200 = tr[TRAIN_EPS-200..].iter().sum::<f64>()/200.0;
    let tr_sr = ts.iter().filter(|&&s|s).count() as f64/TRAIN_EPS as f64*100.0;
    eprintln!("  TRAIN avg={tr_avg:.1} last200={tr_l200:.1} succ={tr_sr:.0}% {elapsed:.1}s");

    // TEST (strict ε=0, pas d'apprentissage)
    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut er: Vec<f64>=vec![]; let mut es: Vec<bool>=vec![];
    for _ in 0..TEST_EPS {
        // Version test qui N'APPELLE PAS engine.step() → pas de reinforce_td
        let mut env = GridEnv5x5::new(); env.reset();
        engine.end_episode();
        engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
        engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;

        let mut total = 0.0; let mut ok = false;
        let (action, _) = pick_greedy_action(&engine, &env.perceive());
        let mut a = action;
        while !env.done {
            let r = env.step_env(a); total += r;
            if env.done { ok=r>0.0; break; }
            // Au lieu de step() (qui entraîne), on fait juste forward_logits
            let p = env.perceive();
            let (action, _) = pick_greedy_action(&engine, &p);
            a = action;
        }
        er.push(total); es.push(ok);
    }
    let ea = er.iter().sum::<f64>()/TEST_EPS as f64;
    let esr = es.iter().filter(|&&s|s).count() as f64/TEST_EPS as f64*100.0;
    eprintln!("  TEST ε=0 avg={ea:.1} succ={esr:.0}%  replay={} C={}",
        engine.cerebellum.replay.len(), engine.num_concepts());
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSULTAT : replay_only={} esr={esr:.0}%/{:.0}%                   ║",
        engine.cerebellum.replay_only, tr_sr);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
}

/// Choisit l'action gloutonne (ε=0) sans passer par step(), pour éviter
/// les effets de bord de reinforce_td/replay pendant le test.
fn pick_greedy_action(engine: &TsoEngine, p: &Array1<f64>) -> (usize, Vec<f64>) {
    // On fait un forward manuel sur les poids du cerebellum.
    // (forward_logits n'est pas accessible sans &mut, mais on peut copier les poids)
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
    let best = logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)|i).unwrap();
    (best, logits)
}

fn main() {
    run_replay_only();
}
