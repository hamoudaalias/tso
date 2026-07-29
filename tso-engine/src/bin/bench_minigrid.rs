//! MiniGrid DoorKey: TSO + baselines on 147D visual observations.
//! Usage: cargo run --release --bin bench_minigrid -- [--seeds N]

use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::baselines::dqn::DqnAgent;

fn main() {
    let n_ep = 100;
    let seeds = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(10);
    let dim = 147;
    let n_actions = 7;

    println!("=== MiniGrid DoorKey (147D RGB, {seeds} seeds, {n_ep} episodes) ===");

    let b = bench_linear(seeds, n_ep);
    println!("linear-AC (raw 147D):        {:7.2} ± {:5.2}", b.0, b.1);

    let t = bench_tso(seeds, n_ep);
    println!("TSO attractor (raw 147D):    {:7.2} ± {:5.2}", t.0, t.1);

    
    

    
    

    println!();
    println!("Δ TSO – linear:             {:+.2}", t.0 - b.0);
    
    
}

fn bench_linear(seeds: usize, n_ep: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(147, 7, 0.01, 0.3, 0.1, 0);
        let mut env = MiniGridEnv::new();
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
            let mut prev_r = 0.0;
            loop {
                let logits = cb.forward_logits(&obs);
                let action = if rand::random::<f64>() < cb.epsilon {
                    rand::random::<usize>() % 7
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

fn bench_tso(seeds: usize, n_ep: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::with_hidden(147, 7, 0);
        engine.cogs.attractor = true;
        let mut env = MiniGridEnv::new();
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

fn _bench_tso_vae(seeds: usize, n_ep: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::with_hidden(147, 7, 0);
        engine.cogs.attractor = true;
        engine.cogs.use_fpi = false;
        let mut env = MiniGridEnv::new();
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

fn _bench_dqn(seeds: usize, n_ep: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut agent = DqnAgent::new(147, 7, 64, 0.001, 0.1);
        let mut env = MiniGridEnv::new();
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
            let mut prev_r = 0.0;
            loop {
                let action = agent.act(&obs);
                let (reward, next_obs, done) = env.step(action);
                agent.store(&obs, action, prev_r, &next_obs, done);
                agent.train(32);
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
