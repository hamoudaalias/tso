//! bench_ablation: TSO variantes sur MiniGrid 7×7 avec stats.
//! 30 seeds, Cohen's d, IC 95%.
//! Usage: cargo run --release --bin bench_ablation

use tso_engine::baselines::multi_seed::{run_bench, SeedResults};
use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use ndarray::Array1;

fn main() {
    let n_seeds = 30;
    let n_ep = 100;
    let dim = 147;
    let n_actions = 7;

    println!("# Ablation TSO — MiniGrid 7×7 (147D)\n");
    println!("N seeds = {n_seeds}, {n_ep} episodes\n");
    println!("| Config | Mean | σ | IC 95% | Cohen d vs A0 | Cohen d vs A1 |");
    println!("|--------|------|---|--------|---------------|---------------|");

    // DQN baseline (30 seeds)
    let dqn = run_bench(n_seeds, || bench_linear(dim, n_actions));
    let ci = dqn.ci95();
    println!("| A0: Linear AC | {:.2} | {:.2} | [{:.2},{:.2}] | 0.00 | — |",
        dqn.mean, dqn.std, ci.0, ci.1);

    let attractor = run_bench(n_seeds, || bench_tso_attractor(dim, n_actions));
    let ci = attractor.ci95();
    println!("| A1: TSO attracteur | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} | 0.00 |",
        attractor.mean, attractor.std, ci.0, ci.1, attractor.cohens_d(&dqn));

    let vae = run_bench(n_seeds, || bench_tso_vae(dim, n_actions));
    let ci = vae.ci95();
    println!("| A2: TSO + VAE | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} | {:.2} |",
        vae.mean, vae.std, ci.0, ci.1, vae.cohens_d(&dqn), vae.cohens_d(&attractor));

    let full = run_bench(n_seeds, || bench_tso_full(dim, n_actions));
    let ci = full.ci95();
    println!("| A3: TSO full | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} | {:.2} |",
        full.mean, full.std, ci.0, ci.1, full.cohens_d(&dqn), full.cohens_d(&attractor));

    let gating = run_bench(n_seeds, || bench_tso_gating(dim, n_actions));
    let ci = gating.ci95();
    println!("| A4: TSO + Φ gating | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} | {:.2} |",
        gating.mean, gating.std, ci.0, ci.1, gating.cohens_d(&dqn), gating.cohens_d(&attractor));

    let no_attractor = run_bench(n_seeds, || bench_tso_no_attractor(dim, n_actions));
    let ci = no_attractor.ci95();
    println!("| A5: TSO sans attrac. | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} | {:.2} |",
        no_attractor.mean, no_attractor.std, ci.0, ci.1,
        no_attractor.cohens_d(&dqn), no_attractor.cohens_d(&attractor));
}

fn bench_linear(dim: usize, na: usize) -> f64 {
    let mut cb = Cerebellum::new(dim, na, 0.01, 0.3, 0.1, 0);
    let mut env = MiniGridEnv::new();
    let mut total = 0.0;
    for _ in 0..100 {
        let mut obs = env.reset();
        let mut prev_r = 0.0;
        loop {
            let logits = cb.forward_logits(&obs);
            let action = if rand::random::<f64>() < cb.epsilon {
                rand::random::<usize>() % na
            } else {
                logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap()
            };
            let (reward, next_obs, done) = env.step(action);
            cb.reinforce_td(prev_r, 0.99);
            cb.decay_trace(0.99, 0.98);
            cb.mark(&obs, action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
}

fn bench_tso_attractor(dim: usize, na: usize) -> f64 {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    eng.cogs.attractor = true;
    eng.cogs.hypothalamus = false; eng.cogs.episodic_curiosity = false;
    eng.cogs.attention = false; eng.cogs.graph_phi = false; eng.cogs.metabolic_cost = false;
    run_minigrid(&mut eng)
}

fn bench_tso_vae(dim: usize, na: usize) -> f64 {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    eng.cogs.attractor = true;
    eng.cogs.use_fpi = false;
    eng.cogs.hypothalamus = false; eng.cogs.episodic_curiosity = false;
    eng.cogs.attention = false; eng.cogs.graph_phi = false; eng.cogs.metabolic_cost = false;
    run_minigrid(&mut eng)
}

fn bench_tso_full(dim: usize, na: usize) -> f64 {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    eng.cogs.attractor = true; eng.cogs.hypothalamus = true;
    eng.cogs.episodic_curiosity = true; eng.cogs.attention = true;
    eng.cogs.graph_phi = true; eng.cogs.metabolic_cost = true;
    run_minigrid(&mut eng)
}

fn bench_tso_gating(dim: usize, na: usize) -> f64 {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    eng.cogs.attractor = true; eng.cogs.hypothalamus = true;
    eng.cogs.episodic_curiosity = true; eng.cogs.attention = true;
    eng.cogs.graph_phi = true; eng.cogs.metabolic_cost = true;
    eng.cogs.phi_gating = true; eng.cogs.phi_threshold = 0.5;
    run_minigrid(&mut eng)
}

fn bench_tso_no_attractor(dim: usize, na: usize) -> f64 {
    let mut eng = TsoEngine::with_hidden(dim, na, 0);
    eng.cogs.attractor = false;
    eng.cogs.hypothalamus = false; eng.cogs.episodic_curiosity = false;
    eng.cogs.attention = false; eng.cogs.graph_phi = false; eng.cogs.metabolic_cost = false;
    run_minigrid(&mut eng)
}

fn run_minigrid(eng: &mut TsoEngine) -> f64 {
    let mut env = MiniGridEnv::new();
    let mut total = 0.0;
    for _ in 0..100 {
        let mut obs = env.reset();
        eng.end_episode();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = env.step(action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
}
