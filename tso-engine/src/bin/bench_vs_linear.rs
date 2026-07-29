/// Rotating-T: TSO vs pure linear actor-critic. 100 seeds.
use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;

fn main() {
    let ep = 150;
    let sw = 50;
    let seeds = 100;

    println!("=== Rotating-T: TSO vs true linear actor-critic ({seeds} seeds) ===\n");

    // Pure linear actor-critic (no engine)
    let b = bench_linear(ep, sw, seeds);
    println!("linear-AC:          {:7.2} ± {:5.2}", b.0, b.1);

    // TSO-full
    let f = bench_tso(ep, sw, seeds);
    println!("TSO-full:           {:7.2} ± {:5.2}", f.0, f.1);

    // TSO-all-off (engine with no subsystems)
    let o = bench_tso_off(ep, sw, seeds);
    println!("TSO-all-off:        {:7.2} ± {:5.2}", o.0, o.1);

    println!();
    println!("Δ TSO – linear-AC:  {:7.2}", f.0 - b.0);
    println!("Δ TSO – all-off:    {:7.2}", f.0 - o.0);
    println!("Δ all-off – linear: {:7.2}", o.0 - b.0);
}

fn bench_linear(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut cb = Cerebellum::new(5, 4, 0.01, 0.3, 0.1, 0);
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
        let mut engine = TsoEngine::new(5, 4);
        // full subsystems
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

fn bench_tso_off(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(5, 4);
        engine.cogs.attractor = false; engine.cogs.hypothalamus = false;
        engine.cogs.episodic_curiosity = false; engine.cogs.attention = false;
        engine.cogs.graph_phi = false; engine.cogs.metabolic_cost = false;
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

fn stats(s: &[f64]) -> (f64, f64) {
    let m = s.iter().sum::<f64>() / s.len() as f64;
    let v = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / s.len() as f64;
    (m, v.sqrt())
}
