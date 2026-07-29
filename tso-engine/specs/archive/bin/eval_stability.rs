/// EVAL — Stabilité du bien-être sur 100 seeds
///
/// Compare moyenne ± écart-type du taux de succès ε=0 pour chaque
/// configuration de poids, sur 100 seeds aléatoires.
/// Sans δ-clip pour éviter le plafond 100%.

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PDIM: usize = 6; const NA: usize = 4; const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const SEEDS: usize = 20;        // réduction pour rapidité

struct Env { agent: (usize, usize), step: usize, done: bool }
impl Env {
    fn new() -> Self { Env { agent: (2,2), step:0, done:false } }
    fn reset(&mut self, r: &mut impl Rng) {
        loop { let x=r.gen_range(0..W); let y=r.gen_range(0..H);
            if !WATER.contains(&(x,y)) { self.agent=(x,y); break; } }
        self.step=0; self.done=false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x,y)=self.agent;let ix=x as isize;let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H)as f64)};
        let mut ws=0.0;for&(wx,wy)in&WATER{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),0.0,ws])
    }
    fn step_env(&mut self, a: usize) -> f64 {
        if self.done{return 0.0;}self.step+=1;
        let(dx,dy)=match a{0=>(0,-1),1=>(0,1),2=>(-1,0),3=>(1,0),_=>(0,0)};
        let nx=self.agent.0 as isize+dx;let ny=self.agent.1 as isize+dy;
        if nx<0||ny<0||nx>=W as isize||ny>=H as isize{
            if self.step>=MAXS{self.done=true;}return-0.5;}
        self.agent=(nx as usize,ny as usize);
        if WATER.contains(&self.agent){self.done=true;return 10.0;}
        if self.step>=MAXS{self.done=true;return-1.0;}
        -0.02
    }
}

fn bfs() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let dm=((W-1)+(H-1))as f64;let mut p=vec![vec![0.0;H];W];
    let mut d=vec![vec![None::<usize>;H];W];let mut q=VecDeque::new();
    for&(wx,wy)in&WATER{d[wx][wy]=Some(0);q.push_back((wx,wy));}
    while let Some((cx,cy))=q.pop_front(){let dd=d[cx][cy].unwrap();
        for(dx,dy)in[(0,1),(0,-1),(1,0),(-1,0)]{let nx=cx as isize+dx;let ny=cy as isize+dy;
            if nx>=0&&ny>=0&&nx<W as isize&&ny<H as isize{let(nx,ny)=(nx as usize,ny as usize);
                if d[nx][ny].is_none(){d[nx][ny]=Some(dd+1);q.push_back((nx,ny));}}}}
    for x in 0..W{for y in 0..H{p[x][y]=match d[x][y]{Some(dd)=>-2.5*dd as f64/dm,None=>-2.5};}}
    p
}

struct Cfg { label: &'static str, weights: [f64; 9] }

fn run_seed(cfg: &[f64; 9], seed: u64) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let mut engine = TsoEngine::with_hidden(PDIM, NA, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0; engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = false;
    engine.cogs.delta_clip_max = 0.0;
    engine.well_being_weights = *cfg;

    let bf = bfs();
    const TRAIN: usize = 200; const TEST: usize = 30;

    for ep in 1..=TRAIN {
        let r = (TRAIN-ep).max(0)as f64/TRAIN as f64;
        engine.cerebellum.epsilon = 0.8*r+0.01; engine.cerebellum.noise_std = 0.3*r+0.01;
        run_ep(&mut engine, &bf, &mut rng);
    }
    engine.cerebellum.epsilon = 0.0; engine.cerebellum.noise_std = 0.0;
    let mut ok = 0usize;
    for _ in 0..TEST { let (_, s) = run_ep(&mut engine, &bf, &mut rng); if s { ok += 1; } }
    ok as f64 / TEST as f64 * 100.0
}

fn run_ep(engine: &mut TsoEngine, bf: &[Vec<f64>], rng: &mut impl Rng) -> (f64, bool) {
    let mut env = Env::new(); env.reset(rng); engine.end_episode();
    let mut total = 0.0; let mut s = false;
    let p = env.perceive(); let bv = Some(bf[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);
    while !env.done {
        let r = env.step_env(a); total += r;
        if env.done { s = r > 0.0; let pt = env.perceive(); engine.step(&pt, r, Some(bf[env.agent.0][env.agent.1]), &[]); break; }
        let pt = env.perceive(); a = engine.step(&pt, r, Some(bf[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode(); (total, s)
}

fn eval(cfg: &Cfg) {
    let mut scores = Vec::with_capacity(SEEDS);
    let t0 = Instant::now();
    eprint!("  {}: ", cfg.label);
    for s in 0..SEEDS as u64 {
        scores.push(run_seed(&cfg.weights, s));
        eprint!("{:.0}% ", scores.last().unwrap());
        if (s+1) % 10 == 0 { eprintln!(); eprint!("     "); }
    }
    let elapsed = t0.elapsed();
    let mean = scores.iter().sum::<f64>() / SEEDS as f64;
    let var = scores.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / SEEDS as f64;
    let std = var.sqrt();
    let pct5 = percentile(&scores, 5.0);
    let pct95 = percentile(&scores, 95.0);
    println!("{:<40} μ={:>6.1}% σ={:>5.1}% [P5={:>4.0}% P95={:>4.0}%] [{:.1?}]",
        cfg.label, mean, std, pct5, pct95, elapsed);
}

fn percentile(scores: &[f64], p: f64) -> f64 {
    let mut s = scores.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (s.len() as f64 - 1.0)).round() as usize;
    s[idx.min(s.len()-1)]
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  EVAL — Stabilité bien-être sur {SEEDS} seeds (sans δ-clip)           ║");
    println!("║  200 train + 30 test ε=0, 5×5, TSO complet                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    let configs = [
        Cfg { label: "Référence (tout=1.0)",                 weights: [1.0; 9] },
        Cfg { label: "metabolic_penalty ×5",                  weights: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0] },
        Cfg { label: "parsimony ×2",                          weights: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0] },
        Cfg { label: "consummatory ×2",                       weights: [1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] },
        Cfg { label: "curiosity ×1 (réf)",                    weights: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] },
        Cfg { label: "gated_reward ×0",                       weights: [0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0] },
    ];

    for cfg in &configs {
        eval(cfg);
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  LÉGENDE                                                             ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  μ = moyenne sur {SEEDS} seeds, σ = écart-type                          ║");
    println!("║  P5/P95 = 5ᵉ et 95ᵉ percentiles                                      ║");
    println!("║  Si P95 - P5 < 30% : config stable (peu de variance seed)            ║");
    println!("║  Si P95 - P5 > 50% : config instable (haute variance seed)           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
}
