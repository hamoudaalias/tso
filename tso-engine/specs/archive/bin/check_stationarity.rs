/// ════════════════════════════════════════════════════════════════════════════
///  Vérification empirique : le signal de reinforce_td en S1 est-il
///  réellement stationnaire (identique pour la même transition) ?
///
///  Dump le rl_signal, reward_ext, bfs_shaping, prev_bfs_value, bfs_value
///  et la position de l'agent pour chaque step, sur 3 épisodes S1.
///
///  Prédiction A1 (flag no-op) : rl_signal dérive entre épisodes
///  Prédiction A2 (stationnaire) : rl_signal identique pour (s,a,s') donné
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6; const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool, start_positions: Vec<(usize, usize)>, ep_idx: usize }
impl GridEnv5x5 {
    fn new() -> Self {
        GridEnv5x5 { agent: (2,2), step:0, done:false, start_positions: vec![(2,2),(2,2),(2,2)], ep_idx:0 }
    }
    fn reset(&mut self) {
        let pos = self.start_positions[self.ep_idx % self.start_positions.len()];
        self.agent = pos;
        self.ep_idx += 1;
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

fn run_diagnostic() {
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.0;  // pas d'exploration → actions déterministes
    engine.cerebellum.noise_std = 0.0;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;
    engine.debug_step_dump = true;

    let bfs_pot = compute_bfs_potential();

    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  DIAGNOSTIC : Stationnarité du rl_signal en S1                     ║");
    eprintln!("║  Même position de départ (2,2) sur 3 épisodes, ε=0, noise=0       ║");
    eprintln!("║  Si rl_signal est identique pour (s,a,s') donné → stationnaire    ║");
    eprintln!("║  Si rl_signal dérive → concept_values/shaping fuient                ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    for ep in 0..3 {
        eprintln!("═══════════════════════════════════════════════════════");
        eprintln!("  ÉPISODE {} (départ (2,2), ε=0)", ep);
        eprintln!("═══════════════════════════════════════════════════════");
        run_ep_stationary(&mut engine, &bfs_pot, ep);
        engine.end_episode();
        eprintln!();
    }

    // Après 3 épisodes, dump le contenu du replay buffer pour les 5 premières transitions
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Replay buffer : {} transitions stockées                         ║", engine.cerebellum.replay.len());
    eprintln!("║  (stations = reward stockée dans le replay)                        ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Concepts={} Edges={} Φ={:.3} total_steps={}",
        engine.attractor.prototypes.len(), engine.graph.edges.len(),
        engine.current_phi, engine.total_steps);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn run_ep_stationary(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], ep_id: usize) {
    let mut env = GridEnv5x5::new();
    env.ep_idx = ep_id;  // même position de départ
    env.reset();

    // Geler l'hypothalamus
    engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
    engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;

    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    eprintln!("  [step 0] position=({},{}): reward_ext=0.0 bfs_value={:.3}",
        env.agent.0, env.agent.1, bv.unwrap());
    let mut a = engine.step(&p, 0.0, bv, &[]);

    let mut step_no = 1;
    while !env.done && step_no < 30 {
        let r = env.step_env(a);

        if env.done {
            let pt = env.perceive();
            let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
            eprintln!("  [step {}] position=({},{}): reward_ext={:.3} bfs_value={:.3} TERMINAL",
                step_no, env.agent.0, env.agent.1, r, bv.unwrap());
            engine.step(&pt, r, bv, &[]);
            break;
        }

        engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;
        let pt = env.perceive();
        let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
        eprintln!("  [step {}] position=({},{}): reward_ext={:.3} bfs_value={:.3}",
            step_no, env.agent.0, env.agent.1, r, bv.unwrap());
        a = engine.step(&pt, r, bv, &[]);
        step_no += 1;
    }
}

fn main() {
    run_diagnostic();
}
