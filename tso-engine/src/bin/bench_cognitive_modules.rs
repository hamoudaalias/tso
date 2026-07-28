/// Benchmark : activation des modules cognitifs "dormants" sur Zigzag 10×10
///
/// Teste l'effet de :
///   (a) Baseline (attractor only) — comme avant
///   (b) + episodic_curiosity      — curiosité intrinsèque via prédiction épisodique
///   (c) + graph_phi               — résolution de conflit Φ
///   (d) + episodic + graph_phi    — les deux combinés
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;

fn run_trial(config_name: &str, episodic: bool, graph_phi: bool,
    seeds: usize, episodes_per_seed: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = TsoEngine::with_hidden(5, 4, 16);
        engine.cogs = CognitiveConfig {
            attractor: true,
            graph_phi,
            attention: false,
            episodic_curiosity: episodic,
            metabolic_cost: false,
            hypothalamus: false,
            delta_clip_max: 5.0,
            ..CognitiveConfig::default()
        };
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.use_stationary_reward = true;
        engine.sleep_every_n_episodes = 0;
        // Activation des graphes = toute la mécanique Φ
        if graph_phi {
            engine.graph = tso_engine::core::Graph::with_params(0.7, 0.1);
        }

        let t0 = Instant::now();
        let mut successes = 0;
        for ep_i in 1..=episodes_per_seed {
            let mut obs = env.reset();
            engine.end_episode();
            loop {
                let action = engine.step(&obs, 0.0, None, &[]);
                let (rew, next_obs) = env.step_env(action);
                if env.done {
                    if rew > 0.0 { successes += 1; }
                    engine.end_episode();
                    break;
                }
                obs = next_obs;
            }
            let frac = (ep_i as f64 / (episodes_per_seed as f64 * 0.5)).min(1.0);
            engine.cerebellum.epsilon = 0.8 * (1.0 - frac * 0.9875);
        }
        let rate = successes as f64 / episodes_per_seed as f64 * 100.0;
        rates.push(rate);
        eprintln!("  {:26} seed {}/{}... {:.1}% [{:.1?}]",
            config_name, seed + 1, seeds, rate, t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  COGNITIVE MODULES — modules dormants sur Zigzag 10×10              ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 8;
    let episodes = 200;

    let configs = [
        ("Baseline",               false, false),
        ("+ épisodique",           true,  false),
        ("+ graphe Φ",             false, true),
        ("+ épisodique + Φ",       true,  true),
    ];

    let mut results: Vec<(&str, f64, f64)> = Vec::new();
    for (name, ep, gr) in &configs {
        eprintln!("\n  ── {} ──", name);
        let (m, s) = run_trial(name, *ep, *gr, seeds, episodes);
        results.push((*name, m, s));
    }

    eprintln!("\n  {}", "=".repeat(60));
    eprintln!("  RÉSULTATS ({} seeds × {} episodes) :", seeds, episodes);
    eprintln!("  {}", "=".repeat(60));
    let baseline = results[0].1;
    for (name, m, s) in &results {
        let gain = m - baseline;
        eprintln!("  {:26} μ={:5.1}% σ={:.2}%  {:+.1}%", name, m, s, gain);
    }

    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    eprintln!("\n  Meilleure config : {} → μ={:.1}%", best.0, best.1);
}
