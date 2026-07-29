/// ablation_matrix — Matrice d'ablations 9 termes × 5 régimes
///
/// Sortie : CSV sur stdout, prêt pour gnuplot / pandas.
///
/// Termes (9) : gated_reward, consummatory, curiosity, shaping,
///              phi_delta, chronic_tension, deficit_penalty,
///              metabolic_penalty, parsimony
///
/// Régimes (5) : Neutre, Faim, Anxiété, Surprise, Métabolique
///
/// Chaque cellule = taux de succès ε=0 (moyenne sur 5 seeds).
///
/// Usage:
///   cargo run --release --bin ablation_matrix > ablation.csv
///   colonnes: terme,regime,succes_moyen,succes_std

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PDIM: usize = 6; const NA: usize = 4; const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const N_SEEDS: usize = 5;

struct Env { agent: (usize, usize), step: usize, done: bool }
impl Env {
    fn new() -> Self { Env { agent: (2,2), step:0, done:false } }
    fn reset(&mut self, r: &mut impl Rng) {
        loop { let x=r.r#gen_range(0..W); let y=r.r#gen_range(0..H);
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

fn run_seed(weights: &[f64; 9], seed: u64, regime: usize) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let mut engine = TsoEngine::with_hidden(PDIM, NA, 4);
    engine.cerebellum.epsilon = 0.1; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0; engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = false;
    engine.cogs.delta_clip_max = 0.0;
    engine.well_being_weights = *weights;

    // Configurer le régime homéostatique
    match regime {
        0 => { /* Neutre : valeurs par défaut */ }
        1 => { // Faim : énergie basse
            engine.hypothalamus.energy = 0.2;
            engine.hypothalamus.hydration = 0.3;
        }
        2 => { // Anxiété : Φ artificiel (concepts forcés)
            for _ in 0..3 {
                let p = Array1::from_vec(vec![0.1,0.2,0.3,0.4,0.0,0.5]);
                engine.step(&p, 0.0, None, &[]);
            }
        }
        3 => { /* Surprise : environnement nouveau (reset normal suffit) */ }
        4 => { // Métabolique : beaucoup de concepts pré-créés
            for _ in 0..10 {
                let p = Array1::from_vec(vec![
                    rng.r#gen::<f64>(), rng.r#gen::<f64>(),
                    rng.r#gen::<f64>(), rng.r#gen::<f64>(),
                    0.0, rng.r#gen::<f64>()]);
                engine.step(&p, 0.0, None, &[]);
            }
        }
        _ => {}
    }

    let bf = bfs();
    const TRAIN: usize = 150; const TEST: usize = 30;

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

fn main() {
    let term_labels = ["gated_reward","consummatory","curiosity","shaping",
                       "phi_delta","chronic_tension","deficit_penalty",
                       "metabolic_penalty","parsimony"];
    let regime_labels = ["Neutre","Faim","Anxiété","Surprise","Métabolique"];

    // header CSV
    println!("{},{}", "terme", (0..5).map(|i| format!("{}_{}", regime_labels[i], "succes")).collect::<Vec<_>>().join(","));

    let t0 = Instant::now();
    let mut total_runs = 0;

    for term_idx in 0..9 {
        let mut results = Vec::new();
        for regime in 0..5 {
            let mut scores = Vec::new();
            for seed in 0..N_SEEDS as u64 {
                let mut w = [1.0; 9];
                w[term_idx] = 0.0; // ablation du terme
                scores.push(run_seed(&w, seed + regime as u64 * 100, regime));
                total_runs += 1;
            }
            let mean = scores.iter().sum::<f64>() / N_SEEDS as f64;
            let var = scores.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / N_SEEDS as f64;
            results.push((mean, var.sqrt()));
        }
        println!("{},{}",
            term_labels[term_idx],
            results.iter().map(|(m,_s)| format!("{:.1}", m)).collect::<Vec<_>>().join(","));
        eprintln!("  {} done ({:.1?})", term_labels[term_idx], t0.elapsed());
    }
    eprintln!("Total: {total_runs} runs en {:.1?}", t0.elapsed());
}
