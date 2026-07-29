/// RL Scaling — benchmark des leviers sur Zigzag 10×10
///
/// Mesure l'effet de :
///   1. Replay epochs (1→3→5)              — échantillonnage plus efficace
///   2. Hidden dim (16→32→64)               — capacité du mapping
///   3. Replay learning rate (0.05→0.1)     — vitesse d'apprentissage
///   4. Combinaison : tous les leviers à la fois
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;

fn run_trial(config_name: &str, hidden_dim: usize, replay_epochs: usize, replay_lr: f64,
    seeds: usize, episodes_per_seed: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = TsoEngine::with_hidden(5, 4, hidden_dim);
        engine.cogs = CognitiveConfig {
            attractor: true,
            graph_phi: false,
            attention: false,
            episodic_curiosity: false,
            metabolic_cost: false,
            hypothalamus: false,
            delta_clip_max: 5.0,
            ..CognitiveConfig::default()
        };
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = replay_lr;
        engine.cerebellum.replay_only = true;
        engine.use_stationary_reward = true;
        engine.sleep_every_n_episodes = 0;

        let t0 = Instant::now();
        let mut successes = 0;
        for ep_i in 1..=episodes_per_seed {
            let mut obs = env.reset();
            engine.end_episode();
            let mut step_no = 0;
            loop {
                let action = engine.step(&obs, 0.0, None, &[]);
                let (rew, next_obs) = env.step_env(action);
                if env.done {
                    if rew > 0.0 { successes += 1; }
                    engine.end_episode();
                    break;
                }
                obs = next_obs;
                step_no += 1;
            }
            // Replay training multi-epoch après chaque épisode
            for _ in 0..replay_epochs {
                engine.cerebellum.replay_train(64, 0.99, 1);
            }
            // Annealing ε
            let frac = (ep_i as f64 / (episodes_per_seed as f64 * 0.5)).min(1.0);
            engine.cerebellum.epsilon = 0.8 * (1.0 - frac * 0.9875);
        }
        let rate = successes as f64 / episodes_per_seed as f64 * 100.0;
        rates.push(rate);
        eprintln!("  {:20} seed {}/{}... {:.1}% [{:.1?}]",
            config_name, seed + 1, seeds, rate, t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RL SCALING — Leviers d'amélioration du cervelet sur Zigzag 10×10   ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 8;
    let episodes = 200;

    // Configs à tester (hidden_dim, replay_epochs, replay_lr)
    let configs = [
        ("Baseline   16/1/0.05", 16, 1, 0.05),
        ("HD=32      32/1/0.05", 32, 1, 0.05),
        ("HD=64      64/1/0.05", 64, 1, 0.05),
        ("Epochs=3   16/3/0.05", 16, 3, 0.05),
        ("Epochs=5   16/5/0.05", 16, 5, 0.05),
        ("LR=0.1     16/1/0.10", 16, 1, 0.10),
        ("Tout max   64/5/0.10", 64, 5, 0.10),
    ];

    let mut results: Vec<(&str, f64, f64)> = Vec::new();
    for (name, hd, ep, lr) in &configs {
        eprintln!("\n  ── {} ──", name);
        let (m, s) = run_trial(name, *hd, *ep, *lr, seeds, episodes);
        results.push((*name, m, s));
    }

    eprintln!("\n  {}", "=".repeat(65));
    eprintln!("  RÉSULTATS ({} seeds × {} episodes = {} total) :", seeds, episodes, seeds * episodes);
    eprintln!("  {}", "=".repeat(65));
    let baseline = results[0].1;
    for (name, m, s) in &results {
        let gain = m - baseline;
        eprintln!("  {:22} μ={:5.1}% σ={:.2}%  {:+.1}%", name, m, s, gain);
    }
    eprintln!();

    // Best config
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    eprintln!("  Meilleure config : {} → μ={:.1}%", best.0, best.1);
}
