//! Benchmark complet: Φ Gating event-driven — reward, FLOPs, latence, threshold sweep
//! Usage: cargo run --release --bin bench_phi_gating_v2 [--seeds N] [threshold1 threshold2 ...]
use std::time::Instant;
use tso_engine::{minigrid_env::MiniGridEnv, TsoEngine};

const DIM: usize = 147;
const NA: usize = 7;

#[derive(Clone, Default)]
struct DetailedMetrics {
    reward: f64,
    skips: usize,
    resolve_calls: usize,
    resolve_iters: usize,
    graph_nodes: usize,
    graph_edges: usize,
    total_steps: usize,
    /// Wall-clock ms for the eval phase
    wall_ms: u64,
    /// Estimated Φ compute cost (edges per phi call × phi calls)
    phi_compute_cost: f64,
    /// Estimated resolve FLOPs (edges × iters × 3 ops per edge)
    resolve_cost: f64,
    /// Cerebellum FLOPs (dim × n_actions per forward)
    cerebellum_cost: f64,
    /// Total energy proxy = phi + resolve + cerebellum (lower is better)
    total_energy: f64,
}

fn warm_engine(eng: &mut TsoEngine, n_ep: usize) {
    let mut env = MiniGridEnv::new();
    for _ in 0..n_ep {
        let mut obs = env.reset();
        eng.end_episode();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (r, o, done) = env.step(action);
            obs = o;
            prev_r = r;
            if done {
                break;
            }
        }
    }
}

fn run_eval_full(eng: &mut TsoEngine, n_ep: usize) -> DetailedMetrics {
    let mut env = MiniGridEnv::new();
    let mut total = 0.0;
    let mut steps = 0;
    let start = Instant::now();
    for _ in 0..n_ep {
        let mut obs = env.reset();
        eng.end_episode();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (r, o, done) = env.step(action);
            obs = o;
            prev_r = r;
            steps += 1;
            if done {
                total += r;
                break;
            }
        }
    }
    let wall_us = start.elapsed().as_micros() as u64;
    let avg = total / n_ep as f64;

    // FLOP model
    let e = eng.graph.edges.len().max(1) as f64;
    let phi_cost = e * (eng.resolve_count as f64 + 1.0); // each resolve + each step's phi()
    let resolve_cost = e * eng.resolve_total_iters as f64 * 3.0; // 3 ops per edge per iter
    let cb_cost = DIM as f64 * NA as f64 * steps as f64; // cerebellum forward per step

    DetailedMetrics {
        reward: avg,
        skips: eng.gating_skip_count,
        resolve_calls: eng.resolve_count,
        resolve_iters: eng.resolve_total_iters,
        graph_nodes: eng.graph.nodes.len() as usize,
        graph_edges: eng.graph.edges.len() as usize,
        total_steps: steps,
        wall_ms: wall_us / 1000,
        phi_compute_cost: phi_cost,
        resolve_cost,
        cerebellum_cost: cb_cost,
        total_energy: phi_cost + resolve_cost + cb_cost,
    }
}

fn mean(v: &[f64]) -> f64 {
    let s: f64 = v.iter().sum();
    s / v.len() as f64
}
fn std_dev(v: &[f64], m: f64) -> f64 {
    let var: f64 = v.iter().map(|x| (x - m).powi(2)).sum();
    (var / v.len() as f64).sqrt()
}

fn benchmark_gating(seed: usize) -> (DetailedMetrics, DetailedMetrics) {
    // Passive gating
    let passive = {
        let mut eng = TsoEngine::with_hidden(DIM, NA, seed);
        eng.cogs.phi_gating = false;
        eng.cogs.graph_phi = true;
        warm_engine(&mut eng, 30);
        eng.gating_skip_count = 0;
        eng.resolve_count = 0;
        eng.resolve_total_iters = 0;
        run_eval_full(&mut eng, 70)
    };

    // Active gating (threshold 0.5)
    let active = {
        let mut eng = TsoEngine::with_hidden(DIM, NA, seed);
        eng.cogs.phi_gating = false;
        eng.cogs.graph_phi = true;
        warm_engine(&mut eng, 30);
        eng.cogs.phi_gating = true;
        eng.cogs.phi_threshold = 0.5;
        eng.gating_skip_count = 0;
        eng.resolve_count = 0;
        eng.resolve_total_iters = 0;
        run_eval_full(&mut eng, 70)
    };

    (passive, active)
}

fn benchmark_gating_with_threshold(seed: usize, threshold: f64) -> DetailedMetrics {
    let mut eng = TsoEngine::with_hidden(DIM, NA, seed);
    eng.cogs.phi_gating = false;
    eng.cogs.graph_phi = true;
    warm_engine(&mut eng, 30);
    eng.cogs.phi_gating = true;
    eng.cogs.phi_threshold = threshold;
    eng.gating_skip_count = 0;
    eng.resolve_count = 0;
    eng.resolve_total_iters = 0;
    run_eval_full(&mut eng, 70)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let thresholds: Vec<f64> = args[2..]
        .iter()
        .filter_map(|a| a.parse().ok())
        .collect();
    let thresholds = if thresholds.is_empty() {
        vec![0.1, 0.3, 0.5, 0.7, 1.0]
    } else {
        thresholds
    };

    // Passive baseline (no gating)
    let mut passive_seeds: Vec<DetailedMetrics> = Vec::new();
    for seed in 0..n_seeds {
        let mut eng = TsoEngine::with_hidden(DIM, NA, seed);
        eng.cogs.phi_gating = false;
        eng.cogs.graph_phi = true;
        warm_engine(&mut eng, 30);
        eng.gating_skip_count = 0;
        eng.resolve_count = 0;
        eng.resolve_total_iters = 0;
        passive_seeds.push(run_eval_full(&mut eng, 70));
    }

    // Threshold sweep
    let mut sweep_results: Vec<(f64, Vec<DetailedMetrics>)> = Vec::new();
    for &thresh in &thresholds {
        let mut metrics = Vec::new();
        for seed in 0..n_seeds {
            metrics.push(benchmark_gating_with_threshold(seed, thresh));
        }
        sweep_results.push((thresh, metrics));
    }

    // ── Print results ──
    println!("=== Φ Gating Event-Driven Benchmark ===");
    println!("Config: MiniGrid DoorKey 7×7 ({}D), {} seeds, 30 warm + 70 eval", DIM, n_seeds);
    println!();

    // Passive summary
    let p_rewards: Vec<f64> = passive_seeds.iter().map(|m| m.reward).collect();
    let p_mean = mean(&p_rewards);
    let p_std = std_dev(&p_rewards, p_mean);
    let p_wall: Vec<f64> = passive_seeds.iter().map(|m| m.wall_ms as f64).collect();
    let p_wall_mean = mean(&p_wall);
    let p_nodes_mean: f64 = passive_seeds.iter().map(|m| m.graph_nodes as f64).sum::<f64>() / n_seeds as f64;
    let p_edges_mean: f64 = passive_seeds.iter().map(|m| m.graph_edges as f64).sum::<f64>() / n_seeds as f64;
    let p_steps_mean: f64 = passive_seeds.iter().map(|m| m.total_steps as f64).sum::<f64>() / n_seeds as f64;

    println!("--- Passive baseline (no gating) ---");
    println!("Reward: {:.4} ± {:.4}", p_mean, p_std);
    println!("Wall-clock: {:.1} ms ({:.1} ms / 1k steps)", p_wall_mean, p_wall_mean / p_steps_mean.max(1.0) * 1000.0);
    println!("Graph nodes: {:.1}, edges: {:.1}", p_nodes_mean, p_edges_mean);
    println!();

    // Threshold sweep table
    println!("--- Threshold sweep ---");
    println!("{:<10}  {:>10}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Threshold", "Reward", "Skip ratio", "Wall (ms)", "ms/1k steps",
        "Nodes", "Edges", "Φ chk cost");
    println!("{:-<10}  {:->10}  {:->10}  {:->12}  {:->12}  {:->12}  {:->12}  {:->12}",
        "", "", "", "", "", "", "", "");

    // Passive row
    println!("{:<10}  {:>10.4}  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12}",
        "∞ (off)", p_mean, "—", p_wall_mean,
        p_wall_mean / p_steps_mean.max(1.0) * 1000.0,
        p_nodes_mean, p_edges_mean, "—");

    for (thresh, metrics) in &sweep_results {
        let rewards: Vec<f64> = metrics.iter().map(|m| m.reward).collect();
        let r_mean = mean(&rewards);
        let skip_rat: Vec<f64> = metrics.iter().map(|m| m.skips as f64 / m.total_steps.max(1) as f64).collect();
        let sr_mean = mean(&skip_rat);
        let walls: Vec<f64> = metrics.iter().map(|m| m.wall_ms as f64).collect();
        let w_mean = mean(&walls);
        let steps: Vec<f64> = metrics.iter().map(|m| m.total_steps as f64).collect();
        let s_mean = mean(&steps);
        let nodes: Vec<f64> = metrics.iter().map(|m| m.graph_nodes as f64).collect();
        let n_mean = mean(&nodes);
        let edges: Vec<f64> = metrics.iter().map(|m| m.graph_edges as f64).collect();
        let e_mean = mean(&edges);
        let phi_costs: Vec<f64> = metrics.iter().map(|m| m.phi_compute_cost).collect();
        let pc_mean = mean(&phi_costs);

        println!("{:<10.1}  {:>10.4}  {:>9.1}%  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.0}",
            thresh, r_mean, sr_mean, w_mean,
            w_mean / s_mean.max(1.0) * 1000.0,
            n_mean, e_mean, pc_mean);
    }
    println!();

    // Summary
    println!("--- Key Findings ---");
    let best = sweep_results.iter().max_by(|a, b| {
        let ra: f64 = a.1.iter().map(|m| m.reward).sum();
        let rb: f64 = b.1.iter().map(|m| m.reward).sum();
        ra.partial_cmp(&rb).unwrap()
    });
    let best_wall = sweep_results.iter().min_by(|a, b| {
        let wa: f64 = a.1.iter().map(|m| m.wall_ms as f64).sum();
        let wb: f64 = b.1.iter().map(|m| m.wall_ms as f64).sum();
        wa.partial_cmp(&wb).unwrap()
    });
    if let Some((t, _)) = best {
        println!("Best reward at threshold = {:.1}", t);
    }
    if let Some((t, _)) = best_wall {
        println!("Lowest latency at threshold = {:.1}", t);
    }
    let (_, active_metrics) = &sweep_results[sweep_results.len() / 2];
    let a_wall_mean: f64 = active_metrics.iter().map(|m| m.wall_ms as f64).sum::<f64>() / n_seeds as f64;
    let wall_save = (1.0 - a_wall_mean / p_wall_mean.max(1.0)) * 100.0;
    println!("Wall-clock savings at threshold=0.5: {:.1}% vs passive", wall_save);
}
