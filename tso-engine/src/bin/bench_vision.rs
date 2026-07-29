/// Vision GridWorld: TSO (VAE + attractor) vs linear AC on 25D grid obs.
/// GridWorld 5×5 open room, goal rotates every 50 episodes (4 positions).
/// Observation: 25D (grid one-hot encoding agent + goal + walls).
use ndarray::Array1;
use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::encoder::{VaeEncoder, Encoder};

fn main() {
    let ep = 150;
    let sw = 50;
    let seeds = 30;

    println!("=== Vision GridWorld (25D grid obs, VAE+attractor vs linear) ===\n");

    // Linear AC on raw 25D
    let b = bench_linear(ep, sw, seeds);
    println!("linear-AC (raw 25D):       {:7.2} ± {:5.2}", b.0, b.1);

    // TSO with VAE encoder (25D → 8D latent → attractor)
    let t = bench_tso_vae(ep, sw, seeds);
    println!("TSO + VAE (25D→8D):       {:7.2} ± {:5.2}", t.0, t.1);

    // TSO without VAE (attractor on raw 25D)
    let r = bench_tso_raw(ep, sw, seeds);
    println!("TSO attractor (raw 25D):   {:7.2} ± {:5.2}", r.0, r.1);

    println!();
    println!("Δ TSO-VAE – linear:       {:7.2}", t.0 - b.0);
    println!("Δ TSO-VAE – TSO-raw:      {:7.2}", t.0 - r.0);
}

/// 25D visual observation: walls + goal direction expanded
fn visual_obs(rt: &RotatingT) -> Array1<f64> {
    let (x, y) = rt.agent;
    let (gx, gy) = rt.goal;
    let mut o = Array1::zeros(25);
    // Agent position (+2 to distinguish from empty/wall/goal)
    o[y * 5 + x] += 2.0;
    // Goal indicator
    o[gy * 5 + gx] += 1.0;
    o
}

fn bench_linear(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(25, 4, 0.01, 0.3, 0.1, 0);
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = visual_obs(&rt);
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
                let (reward, _, done) = rt.step(action);
                cb.reinforce_td(prev_r, 0.99);
                cb.decay_trace(0.99, 0.98);
                cb.mark(&obs, action);
                obs = visual_obs(&rt);
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn bench_tso_vae(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(25, 4);
        // Replace attractor with VAE + smaller latent
        let vae_enc = VaeEncoder::new(25, 16, 8, 0.3);
        engine.belt.set_encoder(Box::new(vae_enc));
        engine.cogs.attractor = true;
        engine.cogs.episodic_curiosity = true;
        engine.cogs.hypothalamus = false;
        engine.cogs.graph_phi = false;
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = visual_obs(&rt);
            let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, _, done) = rt.step(action);
                obs = visual_obs(&rt);
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn bench_tso_raw(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(25, 4);
        engine.cogs.attractor = true;
        engine.cogs.episodic_curiosity = true;
        engine.cogs.hypothalamus = false;
        engine.cogs.graph_phi = false;
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = visual_obs(&rt);
            let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, _, done) = rt.step(action);
                obs = visual_obs(&rt);
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
