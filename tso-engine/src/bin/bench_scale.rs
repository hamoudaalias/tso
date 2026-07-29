//! Benchmark TSO vs linear sur MiniGrid de tailles croissantes.
//! Teste 7×7 (147D), 13×13 (507D), 19×19 (1083D).

use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;

fn main() {
    let n_ep = 50;
    let seeds = 5;

    println!("=== Scale benchmark: TSO vs linear AC ===
");

    for &(w, h) in &[(7, 7), (13, 13), (19, 19)] {
        let dim = w * h * 3;
        println!("--- {w}×{h} ({dim}D) ---");
        let b = bench_linear(w, h, seeds, n_ep);
        println!("  linear-AC:              {:7.2} ± {:5.2}", b.0, b.1);
        let t = bench_tso(w, h, seeds, n_ep);
        println!("  TSO attractor:          {:7.2} ± {:5.2}", t.0, t.1);
        println!("  Δ TSO – linear:         {:+.2}", t.0 - b.0);
        println!();
    }
}

fn bench_linear(w: usize, h: usize, seeds: usize, n_ep: usize) -> (f64, f64) {
    let dim = w * h * 3;
    let n_actions = 7;
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(dim, n_actions, 0.01, 0.3, 0.1, 0);
        let mut env = MiniGridEnv::with_size(w, h);
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
            let mut prev_r = 0.0;
            loop {
                let logits = cb.forward_logits(&obs);
                let action = if rand::random::<f64>() < cb.epsilon {
                    rand::random::<usize>() % n_actions
                } else {
                    logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i).unwrap()
                };
                let (reward, next_obs, done) = env.step(action);
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

fn bench_tso(w: usize, h: usize, seeds: usize, n_ep: usize) -> (f64, f64) {
    let dim = w * h * 3;
    let n_actions = 7;
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::with_hidden(dim, n_actions, 0);
        engine.cogs.attractor = true;
        let mut env = MiniGridEnv::with_size(w, h);
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
            engine.end_episode();
            let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, next_obs, done) = env.step(action);
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
