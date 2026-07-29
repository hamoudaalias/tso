//! Benchmark: TSO vs MLP actor-critic.
//! MLP baseline = Cerebellum (hidden_dim=64) sans TsoEngine.
//! Usage: cargo run --release --bin bench_vs_mlp

use tso_engine::rotating_t::RotatingT;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::tso_engine::TsoEngine;

fn main() {
    let ep = 150;
    let sw = 50;
    let seeds = 30;

    println!("=== TSO vs MLP actor-critic ({seeds} seeds, {ep} episodes) ===");
    let mlp = bench_mlp(ep, sw, seeds);
    println!("MLP AC (hidden=64):        {:7.2} ± {:5.2}", mlp.0, mlp.1);
    let tso = bench_tso(ep, sw, seeds);
    println!("TSO-full:                  {:7.2} ± {:5.2}", tso.0, tso.1);
    println!("delta:                     {:7.2}", tso.0 - mlp.0);
}

fn bench_mlp(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 64);
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = rt.observation();
            let mut prev_r = 0.0;
            loop {
                let logits = cb.forward_logits(&obs);
                let action = if rand::random::<f64>() < cb.epsilon {
                    rand::random::<usize>() % 4
                } else {
                    logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i).unwrap()
                };
                let (reward, next_obs, done) = rt.step(action);
                cb.reinforce_td(prev_r, 0.99);
                cb.decay_trace(0.99, 0.98);
                cb.mark(&obs, action);
                obs = next_obs;
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn bench_tso(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(4, 4);
        engine.cogs.attractor = true; engine.cogs.hypothalamus = true;
        engine.cogs.episodic_curiosity = true; engine.cogs.attention = true;
        engine.cogs.graph_phi = true; engine.cogs.metabolic_cost = true;
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = rt.observation();
            let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, next_obs, done) = rt.step(action);
                obs = next_obs;
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn stats(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}
