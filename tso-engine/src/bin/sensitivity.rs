/// ════════════════════════════════════════════════════════════════════════════
///  sensitivity — Analyse de sensibilité des 9 termes du bien-être
///
///  Balayage systématique : pour chaque terme, multiplier par wi ∈ {0, 0.5, 1, 2, 5}
///  et mesurer le taux de succès ε=0 sur 5×5 dans 4 régimes homéostatiques.
///
///  Termes : gated_reward, consummatory, curiosity, shaping, phi_delta,
///           chronic_tension, deficit_penalty, metabolic_penalty, parsimony
///
///  Régimes : Neutre, Faim, Anxiété, Métabolique
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;

// ─── Grid 5×5 ─────────────────────────────────────────────────────────

const W: usize = 5; const H: usize = 5;
const PDIM: usize = 6; const NA: usize = 4; const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

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

// ── 9 well-being terms: index the contribution components ─────────────
// We'll weight step's built-in total_reward externally via CognitiveConfig.
// For internal weight control, we monkey-patch at step level:
struct SensitivityConfig {
    pub n_gated_reward: f64,  // weight on external reward gating
    pub n_consummatory: f64,
    pub n_curiosity: f64,
    pub n_shaping: f64,
    pub n_phi_delta: f64,
    pub n_chronic_tension: f64,
    pub n_deficit: f64,
    pub n_metabolic: f64,
    pub n_parsimony: f64,
}

fn run_sweep(engine_cfg: &SensitivityConfig, regime: &str, seed: u64, n_seed: f64, is_last: bool) -> f64 {
    run_single(engine_cfg, seed, n_seed, false)
}

fn run_single(cfg: &SensitivityConfig, seed: u64, _n_seed: f64, _is_test: bool) -> f64 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let mut engine = TsoEngine::with_hidden(PDIM, NA, 4);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = false;
    // Désactive le δ-clip pour stresser le well-being non-stationnaire
    engine.cogs.delta_clip_max = 0.0;
    engine.well_being_weights = [
        cfg.n_gated_reward, cfg.n_consummatory, cfg.n_curiosity,
        cfg.n_shaping, cfg.n_phi_delta, cfg.n_chronic_tension,
        cfg.n_deficit, cfg.n_metabolic, cfg.n_parsimony,
    ];

    let bf = bfs();
    const TRAIN: usize = 300;
    const TEST: usize = 50;

    for ep in 1..=TRAIN {
        let r = (TRAIN-ep).max(0)as f64/TRAIN as f64;
        engine.cerebellum.epsilon = 0.8*r+0.01;
        engine.cerebellum.noise_std = 0.3*r+0.01;
        run_ep(&mut engine, &bf, &mut rng);
    }

    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    let mut ok = 0usize;
    for _ in 0..TEST {
        let (_, s) = run_ep(&mut engine, &bf, &mut rng);
        if s { ok += 1; }
    }
    ok as f64 / TEST as f64 * 100.0
}

fn run_ep(engine: &mut TsoEngine, bf: &[Vec<f64>], rng: &mut impl Rng) -> (f64, bool) {
    let mut env = Env::new(); env.reset(rng);
    engine.end_episode();
    let mut total = 0.0; let mut s = false;
    let p = env.perceive();
    let bv = Some(bf[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);
    while !env.done {
        let r = env.step_env(a); total += r;
        if env.done { s = r > 0.0; let pt = env.perceive(); engine.step(&pt, r, Some(bf[env.agent.0][env.agent.1]), &[]); break; }
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bf[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    (total, s)
}

fn label_term(i: usize) -> &'static str {
    match i {
        0 => "gated_reward",
        1 => "consummatory",
        2 => "curiosity",
        3 => "shaping",
        4 => "phi_delta",
        5 => "chronic_tension",
        6 => "deficit_penalty",
        7 => "metabolic_penalty",
        8 => "parsimony",
        _ => "??",
    }
}

fn main() {
    // 9 termes × 5 weights × 1 seed = 45 runs ≈ ~30s
    let weights = [0.0, 0.5, 1.0, 2.0, 5.0];
    let base_cfg = SensitivityConfig {
        n_gated_reward: 1.0, n_consummatory: 1.0, n_curiosity: 1.0,
        n_shaping: 1.0, n_phi_delta: 1.0, n_chronic_tension: 1.0,
        n_deficit: 1.0, n_metabolic: 1.0, n_parsimony: 1.0,
    };

    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  SENSITIVITY — 9 well-being terms × {w} weights                 ║", w = weights.len());
    eprintln!("║  Env: 5×5, TSO complet, δ-clip, 300 train, 50 test ε=0             ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Chaque ligne : un terme, un weight, un seed. Colonne = taux test   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    eprintln!("{:<18} {:>6} {:>10} {:>6}", "Terme", "Poids", "Succès", "Delta");
    eprintln!("{:-<18} {:-<6} {:-<10} {:-<6}", "", "", "", "");

    let ref_score = run_single(&base_cfg, 42, 1.0, false);
    eprintln!("{:<18} {:>6} {:>9.1}%", "Référence (tout=1)", "", ref_score);

    for term_idx in 0..9 {
        for &w in &weights {
            let mut cfg = SensitivityConfig {
                n_gated_reward: if term_idx==0 { w } else { 1.0 },
                n_consummatory: if term_idx==1 { w } else { 1.0 },
                n_curiosity: if term_idx==2 { w } else { 1.0 },
                n_shaping: if term_idx==3 { w } else { 1.0 },
                n_phi_delta: if term_idx==4 { w } else { 1.0 },
                n_chronic_tension: if term_idx==5 { w } else { 1.0 },
                n_deficit: if term_idx==6 { w } else { 1.0 },
                n_metabolic: if term_idx==7 { w } else { 1.0 },
                n_parsimony: if term_idx==8 { w } else { 1.0 },
            };

            let score = run_single(&cfg, 42 + term_idx as u64 + (w * 10.0) as u64, 1.0, false);
            let delta = score - ref_score;
            eprintln!("{:<18} {:>5.1}x {:>8.1}% {:>+6.1}",
                label_term(term_idx), w, score, delta);
        }
    }
}
