/// MiniGrid DoorKey: TSO + VAE vs linear AC on 147D visual observations.
use ndarray::Array1;
use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::encoder::VaeEncoder;

fn main() {
    let n_ep = 100;
    let seeds = 30;

    println!("=== MiniGrid DoorKey (147D RGB, {seeds} seeds) ===\n");

    // Linear AC on raw 147D
    let b = bench_linear(n_ep, seeds);
    println!("linear-AC (raw 147D):        {:7.2} ± {:5.2}", b.0, b.1);

    // TSO + VAE (147D → 16D latent)
    let t = bench_tso_vae(n_ep, seeds);
    println!("TSO + VAE (147D→16D):       {:7.2} ± {:5.2}", t.0, t.1);

    // TSO attractor on raw 147D
    let r = bench_tso_raw(n_ep, seeds);
    println!("TSO attractor (raw 147D):    {:7.2} ± {:5.2}", r.0, r.1);

    println!();
    println!("Δ TSO-VAE – linear:         {:7.2}", t.0 - b.0);
    println!("Δ TSO-VAE – TSO-raw:        {:7.2}", t.0 - r.0);
}

fn bench_linear(n_ep: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(147, 4, 0.01, 0.3, 0.1, 0);
        let mut env = MiniGridEnv::new();
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
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

fn bench_tso_vae(n_ep: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(147, 4);
        let vae_enc = VaeEncoder::new(147, 32, 16, 0.3);
        engine.belt.set_encoder(Box::new(vae_enc));
        engine.cogs.attractor = true;
        engine.cogs.episodic_curiosity = true;
        engine.cogs.hypothalamus = false;
        engine.cogs.graph_phi = false;
        let mut env = MiniGridEnv::new();
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
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

fn bench_tso_raw(n_ep: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(147, 4);
        engine.cogs.attractor = true;
        engine.cogs.episodic_curiosity = true;
        engine.cogs.hypothalamus = false;
        engine.cogs.graph_phi = false;
        let mut env = MiniGridEnv::new();
        let mut total = 0.0;
        for _ in 0..n_ep {
            let mut obs = env.reset();
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

fn stats(s: &[f64]) -> (f64, f64) {
    let m = s.iter().sum::<f64>() / s.len() as f64;
    let v = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / s.len() as f64;
    (m, v.sqrt())
}
