use ndarray::Array1;
use rand::Rng;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6; const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool }
impl GridEnv5x5 {
    fn new() -> Self { GridEnv5x5 { agent: (2,2), step:0, done:false } }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop { let x=rng.gen_range(0..W); let y=rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x,y))&&!FOOD_POSITIONS.contains(&(x,y)) { self.agent=(x,y); break; } }
        self.step=0; self.done=false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x,y)=self.agent;let ix=x as isize;let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H)as f64)};
        let mut fs=0.0;for&(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0;for&(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws])
    }
    fn step_env(&mut self, action: usize) -> f64 {
        if self.done{return 0.0;}self.step+=1;
        let(dx,dy)=match action{0=>(0,-1),1=>(0,1),2=>(-1,0),3=>(1,0),_=>(0,0)};
        let nx=self.agent.0 as isize+dx;let ny=self.agent.1 as isize+dy;
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
    for&(wx,wy)in&WATER_POSITIONS{dist[wx][wy]=Some(0);q.push_back((wx,wy));}
    while let Some((cx,cy))=q.pop_front(){let d=dist[cx][cy].unwrap();
        for(dx,dy)in[(0,1),(0,-1),(1,0),(-1,0)]{let nx=cx as isize+dx;let ny=cy as isize+dy;
            if nx>=0&&ny>=0&&nx<W as isize&&ny<H as isize{let(nx,ny)=(nx as usize,ny as usize);
                if dist[nx][ny].is_none(){dist[nx][ny]=Some(d+1);q.push_back((nx,ny));}}}}
    for x in 0..W{for y in 0..H{pot[x][y]=match dist[x][y]{Some(d)=>-2.5*d as f64/d_max,None=>-2.5};}}
    pot
}

fn main() {
    let bfs_pot = compute_bfs_potential();
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05; engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;
    engine.hypothalamus.energy=1.0; engine.hypothalamus.hydration=1.0;
    engine.hypothalamus.temperature=0.5; engine.hypothalamus.sleep_debt=0.0;

    eprintln!("=== INSPECTION DU SIGNAL RL (1er épisode) ===");
    
    let mut env = GridEnv5x5::new(); env.reset(); engine.end_episode();
    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);
    
    let mut step_count = 0;
    while !env.done && step_count < 20 {
        let r = env.step_env(a);
        let pt = env.perceive();
        let bv_next = Some(bfs_pot[env.agent.0][env.agent.1]);
        a = engine.step(&pt, r, bv_next, &[]);
        step_count += 1;
    }
    engine.end_episode();

    eprintln!();
    eprintln!("=== FIN INSPECTION ===");
}
