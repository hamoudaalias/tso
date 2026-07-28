/// Curiosity par comptage de visites — spike sur Zigzag 10×10
///
/// Principe : chaque état perçu a un compteur N(s). Bonus = β / sqrt(N(s) + 1).
/// L'agent est récompensé pour explorer des états nouveaux — utile pour les
/// environnements à reward sparse (Minigrid, Sokoban).
///
/// Implémentation minimale : HashMap<état_hash, compteur> dans le benchmark,
/// pas de modification du moteur TSO (spike).
use std::collections::HashMap;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;

/// Hash simple pour un vecteur f64 (arrondi à 2 décimales)
fn hash_state(s: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let rounded: Vec<i64> = s.iter().map(|x| (x * 100.0).round() as i64).collect();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    rounded.hash(&mut h);
    h.finish()
}

fn run_trial(config_name: &str, curiosity_bonus: f64,
    seeds: usize, episodes_per_seed: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = TsoEngine::with_hidden(5, 4, 16);
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
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.use_stationary_reward = true;
        engine.sleep_every_n_episodes = 0;

        let mut visit_counts: HashMap<u64, usize> = HashMap::new();
        // Désactiver le shaping BFS dans step_env — on veut du sparse pur
        // (on garde step_flat -0.01 + goal 20)

        let t0 = Instant::now();
        let mut successes = 0;
        for ep_i in 1..=episodes_per_seed {
            let mut obs = env.reset();
            engine.end_episode();
            loop {
                // Bonus de curiosité par comptage de visites
                let state_sig = hash_state(obs.as_slice().unwrap_or(&[]));
                let count = visit_counts.get(&state_sig).copied().unwrap_or(0);
                let bonus = curiosity_bonus / ((count as f64 + 1.0).sqrt());
                visit_counts.insert(state_sig, count + 1);

                let action = engine.step(&obs, bonus, None, &[]);
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
        eprintln!("  {:22} seed {}/{}... {:.1}% [{:.1?}]",
            config_name, seed + 1, seeds, rate, t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  SPICE : Count-based curiosity sur Zigzag 10×10                     ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 8;
    let episodes = 200;

    let configs = [
        ("Baseline (β=0)",        0.0),
        ("Curiosité β=0.1",       0.1),
        ("Curiosité β=0.25",      0.25),
        ("Curiosité β=0.5",       0.5),
        ("Curiosité β=1.0",       1.0),
    ];

    let mut results: Vec<(&str, f64, f64)> = Vec::new();
    for (name, beta) in &configs {
        eprintln!("\n  ── {} ──", name);
        let (m, s) = run_trial(name, *beta, seeds, episodes);
        results.push((*name, m, s));
    }

    eprintln!("\n  {}", "=".repeat(55));
    eprintln!("  RÉSULTATS ({} seeds × {} episodes) :", seeds, episodes);
    eprintln!("  {}", "=".repeat(55));
    let baseline = results[0].1;
    for (name, m, s) in &results {
        let gain = m - baseline;
        eprintln!("  {:22} μ={:5.1}% σ={:.2}%  {:+.1}%", name, m, s, gain);
    }

    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    eprintln!("\n  Meilleure config : {} → μ={:.1}%", best.0, best.1);
}
