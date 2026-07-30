//! TSO Demo — quick-start in 30 seconds.
//!
//! Usage:
//!   cargo run --release --bin demo                     # quick: 3 seeds, 10 episodes
//!   cargo run --release --bin demo -- --seeds 5 --episodes 20    # custom
//!   cargo run --release --bin demo -- --attractor --graph_phi --threshold 0.3
//!   cargo run --release --bin demo -- --json                     # JSON output
//!
//! All CognitiveConfig flags can be toggled via --<flag> (true) or --no-<flag> (false).

use std::collections::HashMap;
use tso_engine::minigrid_env::MiniGridEnv;
use tso_engine::tso_engine::{CognitiveConfig, TsoEngine};

struct Args {
    seeds: usize,
    episodes: usize,
    json: bool,
    config: CognitiveConfig,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut seeds = 3usize;
    let mut episodes = 10usize;
    let mut json = false;
    let mut config = CognitiveConfig::default();

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--seeds" | "-s" => { i += 1; seeds = raw[i].parse().unwrap_or(3); }
            "--episodes" | "-e" => { i += 1; episodes = raw[i].parse().unwrap_or(10); }
            "--json" | "-j" => json = true,
            "--attractor" => config.attractor = true,
            "--no-attractor" => config.attractor = false,
            "--graph_phi" => config.graph_phi = true,
            "--no-graph_phi" => config.graph_phi = false,
            "--hypothalamus" => config.hypothalamus = true,
            "--no-hypothalamus" => config.hypothalamus = false,
            "--episodic" => config.episodic_curiosity = true,
            "--no-episodic" => config.episodic_curiosity = false,
            "--attention" => config.attention = true,
            "--no-attention" => config.attention = false,
            "--threshold" | "-t" => { i += 1; config.phi_threshold = raw[i].parse().unwrap_or(0.5); }
            "--use_fpi" => config.use_fpi = true,
            "--no-use_fpi" => config.use_fpi = false,
            "--help" | "-h" => {
                eprintln!("Usage: cargo run --release --bin demo -- [OPTIONS]");
                eprintln!("  --seeds N, -s N           Seeds (default 3)");
                eprintln!("  --episodes N, -e N       Episodes per seed (default 10)");
                eprintln!("  --json, -j               JSON output");
                eprintln!("  --attractor / --no-attractor");
                eprintln!("  --graph_phi / --no-graph_phi");
                eprintln!("  --hypothalamus / --no-hypothalamus");
                eprintln!("  --episodic / --no-episodic");
                eprintln!("  --attention / --no-attention");
                eprintln!("  --threshold N, -t N      Phi threshold");
                eprintln!("  --use_fpi / --no-use_fpi");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    Args { seeds, episodes, json, config }
}

fn run_tso_benchmark(cfg: &CognitiveConfig, seeds: usize, n_ep: usize) -> (Vec<f64>, f64, f64) {
    let mut scores = Vec::with_capacity(seeds);
    for _ in 0..seeds {
        let mut engine = TsoEngine::with_hidden(147, 7, 0);
        engine.cogs = cfg.clone();
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
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let var = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    (scores, mean, var.sqrt())
}

fn main() {
    let args = parse_args();
    let label = format!("TSO (seeds={}, ep={})", args.seeds, args.episodes);

    if !args.json {
        println!("=== {} ===", label);
        println!("Config: attractor={}, graph_phi={}, hypothalamus={}, episodic={}, attention={}, use_fpi={}, threshold={}",
            args.config.attractor, args.config.graph_phi, args.config.hypothalamus,
            args.config.episodic_curiosity, args.config.attention, args.config.use_fpi,
            args.config.phi_threshold);
    }

    let (scores, mean, std) = run_tso_benchmark(&args.config, args.seeds, args.episodes);

    if args.json {
        let mut out = HashMap::new();
        out.insert("config", serde_json::json!({
            "attractor": args.config.attractor,
            "graph_phi": args.config.graph_phi,
            "hypothalamus": args.config.hypothalamus,
            "episodic": args.config.episodic_curiosity,
            "attention": args.config.attention,
            "use_fpi": args.config.use_fpi,
            "phi_threshold": args.config.phi_threshold,
        }));
        out.insert("results", serde_json::json!({
            "seeds": args.seeds,
            "episodes": args.episodes,
            "mean": mean,
            "std": std,
            "scores": scores,
        }));
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!("Mean: {:.4} ± {:.4}", mean, std);
        println!("Scores: {:?}", scores.iter().map(|s| format!("{:.4}", s)).collect::<Vec<_>>());
    }
}
