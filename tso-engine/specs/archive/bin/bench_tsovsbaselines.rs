/// BENCHMARK — TSO complet vs Q-learning tabulaire vs Actor-Critic nu
/// Environnement : Terrarium 7×7 (aliasing, récompenses rares)
/// Protocole : 100 train, 20 test ε=0, 10 seeds

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::terrarium::Terrarium;

const N_ACTIONS: usize = 4;
const N_SEEDS: usize = 10;
const TRAIN: usize = 100;
const TEST: usize = 20;
const HIDDEN: usize = 4;

// ── Q-learning tabulaire (état = position) ──
fn run_ql(seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_states = 49; // 7×7
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; N_ACTIONS]; n_states];

    for ep in 1..=TRAIN {
        let frac = ((TRAIN - ep) as f64 / TRAIN as f64).max(0.0);
        let eps = 0.01 + 0.79 * frac;

        let mut env = Terrarium::new(seed);
        env.reset();
        loop {
            let (x, y) = env.agent;
            let s = x * 7 + y;
            let a = if rng.r#gen::<f64>() < eps { rng.gen_range(0..N_ACTIONS) }
                    else { q[s].iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap() };
            let r = env.step(a);
            if env.done {
                q[s][a] += 0.1 * (r - q[s][a]);
                break;
            }
            let (nx, ny) = env.agent;
            let sn = nx * 7 + ny;
            let qn = q[sn].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            q[s][a] += 0.1 * (r + 0.99 * qn - q[s][a]);
        }
    }
    // Test ε=0
    let mut ok = 0;
    for _ in 0..TEST {
        let mut env = Terrarium::new(seed);
        env.reset();
        loop {
            let (x, y) = env.agent;
            let s = x * 7 + y;
            let a = q[s].iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap();
            let r = env.step(a);
            if env.done { if r > 0.0 { ok += 1; } break; }
        }
    }
    ok as f64 / TEST as f64 * 100.0
}

// ── Actor-Critic nu (Cerebellum linéaire, perception brute) ──
fn run_ac(seed: u64) -> f64 {
    let _seed = seed;
    let pdim = 6; // whiskers(4) + food_sensed + water_sensed
    let mut engine = TsoEngine::with_hidden(pdim, N_ACTIONS, 0);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.use_stationary_reward = false;
    engine.cogs.delta_clip_max = 5.0;
    // Désactiver tout le cognitive overhead
    engine.cogs.attractor = false;
    engine.cogs.graph_phi = false;
    engine.cogs.episodic_curiosity = false;
    engine.cogs.metabolic_cost = false;
    engine.cogs.hypothalamus = false;
    engine.cogs.attention = false;
    engine.curiosity_weight = 0.0;

    for ep in 1..=TRAIN {
        let frac = ((TRAIN - ep) as f64 / TRAIN as f64).max(0.0);
        engine.cerebellum.epsilon = 0.8 * frac + 0.01;
        engine.cerebellum.noise_std = 0.3 * frac + 0.01;

        let mut env = Terrarium::new(seed);
        env.reset();
        engine.end_episode();
        let p = env.perception(None);
        let mut a = engine.step(&p, 0.0, None, &[]);
        loop {
            let r = env.step(a);
            if env.done {
                let pt = env.perception(None);
                engine.step(&pt, r, None, &[]);
                break;
            }
            let pt = env.perception(None);
            a = engine.step(&pt, r, None, &[]);
        }
        engine.end_episode();
    }
    // Test ε=0
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    let mut ok = 0;
    for _ in 0..TEST {
        let mut env = Terrarium::new(seed);
        env.reset();
        engine.end_episode();
        let p = env.perception(None);
        let mut a = engine.step(&p, 0.0, None, &[]);
        loop {
            let r = env.step(a);
            if env.done {
                if r > 0.0 { ok += 1; }
                let pt = env.perception(None);
                engine.step(&pt, r, None, &[]);
                break;
            }
            let pt = env.perception(None);
            a = engine.step(&pt, r, None, &[]);
        }
    }
    ok as f64 / TEST as f64 * 100.0
}

// ── TSO complet (curiosity=0.5, attracteur, graphe, etc.) ──
fn run_tso(seed: u64) -> f64 {
    let _seed = seed;
    let pdim = 6;
    let mut engine = TsoEngine::with_hidden(pdim, N_ACTIONS, HIDDEN);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.use_stationary_reward = false;
    engine.cogs.delta_clip_max = 5.0;
    engine.curiosity_weight = 0.5;
    engine.cogs.episodic_curiosity = true;
    engine.cogs.hypothalamus = true;
    engine.cogs.metabolic_cost = true;
    engine.cogs.attention = true;
    engine.cogs.attractor = true;
    engine.cogs.graph_phi = true;

    for ep in 1..=TRAIN {
        let frac = ((TRAIN - ep) as f64 / TRAIN as f64).max(0.0);
        engine.cerebellum.epsilon = 0.8 * frac + 0.01;
        engine.cerebellum.noise_std = 0.3 * frac + 0.01;

        let mut env = Terrarium::new(seed);
        env.reset();
        engine.end_episode();
        let p = env.perception(None);
        let mut a = engine.step(&p, 0.0, None, &[]);
        loop {
            let r = env.step(a);
            if env.done {
                let pt = env.perception(None);
                engine.step(&pt, r, None, &[]);
                break;
            }
            let pt = env.perception(None);
            a = engine.step(&pt, r, None, &[]);
        }
        engine.end_episode();
    }
    // Test ε=0
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    let mut ok = 0;
    for _ in 0..TEST {
        let mut env = Terrarium::new(seed);
        env.reset();
        engine.end_episode();
        let p = env.perception(None);
        let mut a = engine.step(&p, 0.0, None, &[]);
        loop {
            let r = env.step(a);
            if env.done {
                if r > 0.0 { ok += 1; }
                let pt = env.perception(None);
                engine.step(&pt, r, None, &[]);
                break;
            }
            let pt = env.perception(None);
            a = engine.step(&pt, r, None, &[]);
        }
    }
    ok as f64 / TEST as f64 * 100.0
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK — TSO vs Q-learning vs Actor-Critic sur Terrarium 7×7    ║");
    println!("║  {} train, {} test ε=0, {} seeds per config                    ║", TRAIN, TEST, N_SEEDS);
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Raw scores per agent
    let t0 = Instant::now();

    // ── Q-learning tabulaire ──
    let mut qs = Vec::with_capacity(N_SEEDS);
    for s in 0..N_SEEDS { qs.push(run_ql(s as u64)); }
    let qm = qs.iter().sum::<f64>() / N_SEEDS as f64;
    let qv = qs.iter().map(|x| (x - qm).powi(2)).sum::<f64>() / N_SEEDS as f64;
    println!("Q-learning tabulaire  : {:>7.1}% (σ={:>5.1}%) [position, observabilité parfaite]", qm, qv.sqrt());
    eprintln!("  QL done [{:.1?}]", t0.elapsed());

    // ── Actor-Critic nu ──
    let mut acs = Vec::with_capacity(N_SEEDS);
    for s in 0..N_SEEDS { acs.push(run_ac(s as u64)); }
    let acm = acs.iter().sum::<f64>() / N_SEEDS as f64;
    let acv = acs.iter().map(|x| (x - acm).powi(2)).sum::<f64>() / N_SEEDS as f64;
    println!("Actor-Critic nu       : {:>7.1}% (σ={:>5.1}%) [Cerebellum linéaire, whisper seulement]", acm, acv.sqrt());
    eprintln!("  AC done [{:.1?}]", t0.elapsed());

    // ── TSO complet ──
    let mut ts = Vec::with_capacity(N_SEEDS);
    for s in 0..N_SEEDS {
        ts.push(run_tso(s as u64));
    }
    let tm = ts.iter().sum::<f64>() / N_SEEDS as f64;
    let tv = ts.iter().map(|x| (x - tm).powi(2)).sum::<f64>() / N_SEEDS as f64;
    println!("TSO complet (ig=0.5)   : {:>7.1}% (σ={:>5.1}%) [attracteur+graphe+curiosité+hypothalamus]", tm, tv.sqrt());
    eprintln!("  TSO done [{:.1?}]", t0.elapsed());

    println!();
    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Δ QL → TSO  : {:>+6.1}%", tm - qm);
    println!("  Δ AC → TSO  : {:>+6.1}%", tm - acm);

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  QL: état = position (x,y) — triche observabilité parfaite");
    println!("  AC : 6 whiskers (4 murs + food_sensed + water_sensed)");
    println!("  TSO: mêmes whiskers + attracteur + graphe Φ + curiosité info_gain");
    println!("─────────────────────────────────────────────────────────────────────");

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  RAPPORT pour paper.md                                                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Baseline QL (position): {:>5.1}% ± {:.1}%                          ║", qm, qv.sqrt());
    println!("║  Baseline AC (whiskers): {:>5.1}% ± {:.1}%                          ║", acm, acv.sqrt());
    println!("║  TSO complet            : {:>5.1}% ± {:.1}%                          ║", tm, tv.sqrt());
    println!("║  Delta TSO − QL         : {:>+5.1}%                                   ║", tm - qm);
    println!("║  Delta TSO − AC         : {:>+5.1}%                                   ║", tm - acm);
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
}
