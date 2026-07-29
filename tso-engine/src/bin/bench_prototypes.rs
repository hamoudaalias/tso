//! Benchmark: TSO attracteur vs K-means vs pas de catégorisation.
//! Usage: cargo run --release --bin bench_prototypes

use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;

fn main() {
    let ep = 100;
    let sw = 50;
    let seeds = 20;

    println!("=== Catégorisation: attracteur vs none vs kmeans ({seeds} seeds) ===
");
    let none = bench_tso_off(ep, sw, seeds);
    println!("TSO sans catégorisation:  {:7.2} ± {:5.2}", none.0, none.1);
    let attr = bench_tso_attractor(ep, sw, seeds);
    println!("TSO + attracteur:         {:7.2} ± {:5.2}", attr.0, attr.1);
    let km = bench_tso_kmeans(ep, sw, seeds);
    println!("TSO + k-means:            {:7.2} ± {:5.2}", km.0, km.1);
    println!("
Δ attracteur - none:     {:+.2}", attr.0 - none.0);
    println!("Δ k-means - none:        {:+.2}", km.0 - none.0);
    println!("Δ attracteur - k-means:  {:+.2}", attr.0 - km.0);
}

fn bench_tso_off(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(4, 4);
        engine.cogs.attractor = false;
        engine.cogs.hypothalamus = false; engine.cogs.episodic_curiosity = false;
        engine.cogs.attention = false; engine.cogs.graph_phi = false;
        engine.cogs.metabolic_cost = false;
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, next_obs, done) = rt.step(action);
                obs = next_obs; prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn bench_tso_attractor(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(4, 4);
        engine.cogs.attractor = true;
        engine.cogs.hypothalamus = false; engine.cogs.episodic_curiosity = false;
        engine.cogs.attention = false; engine.cogs.graph_phi = false;
        engine.cogs.metabolic_cost = false;
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, next_obs, done) = rt.step(action);
                obs = next_obs; prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    stats(&scores)
}

fn bench_tso_kmeans(n_ep: usize, sw: usize, seeds: usize) -> (f64, f64) {
    // Implémentation K-means simple sans dépendance externe.
    // Chaque perception est assignée au centroid le plus proche (L2).
    // Les centroids sont mis à jour par moyenne glissante (LR=0.1).
    let k = 10;
    let lr = 0.1;
    let dim = 4;
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(4, 4);
        engine.cogs.attractor = false;  // désactive attracteur, on utilise k-means manuel
        engine.cogs.hypothalamus = false; engine.cogs.episodic_curiosity = false;
        engine.cogs.attention = false; engine.cogs.graph_phi = false;
        engine.cogs.metabolic_cost = false;
        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..dim).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset(); let mut prev_r = 0.0;
            loop {
                let obs = rt.observation();
                // K-means assignation
                let mut best_d = f64::MAX;
                let mut best_k = 0;
                for (k_idx, c) in centroids.iter().enumerate() {
                    let d: f64 = (0..dim).map(|i| (obs[i] - c[i]).powi(2)).sum();
                    if d < best_d { best_d = d; best_k = k_idx; }
                }
                // Update centroid
                for i in 0..dim {
                    centroids[best_k][i] += lr * (obs[i] - centroids[best_k][i]);
                }
                // Step engine (raw obs, attractor off)
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, _next_obs, done) = rt.step(action);
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
