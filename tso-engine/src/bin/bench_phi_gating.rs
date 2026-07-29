//! Benchmark: Φ gating vs passif sur MiniGrid DoorKey 7×7
//! Mesure : reward moyen, ratio de ticks partiels, time par épisode
//!
//! Usage: cargo run --release --bin bench_phi_gating

use tso_engine::{CognitiveConfig, TsoEngine, minigrid_env::MiniGridEnv};

fn main() {
    let n_seeds = 5;
    let n_episodes = 50;
    let dim = 147;
    let n_actions = 7;

    let mut passif_rewards = Vec::new();
    let mut actif_rewards = Vec::new();

    for _seed in 0..n_seeds {
        {
            let mut env = MiniGridEnv::new();
            let mut tso = TsoEngine::with_hidden(dim, n_actions, 0);
            tso.cogs.phi_gating = false;
            let mut total_reward = 0.0;

            for _ep in 0..n_episodes {
                let obs = env.reset();
                tso.end_episode();
                for _step in 0..100 {
                    let action = tso.step(&obs, 0.0, None, &[]);
                    let (r, _next_obs, done) = env.step(action);
                    total_reward += r;
                    if done { break; }
                }
            }
            passif_rewards.push(total_reward / n_episodes as f64);
        }

        {
            let mut env = MiniGridEnv::new();
            let mut tso = TsoEngine::with_hidden(dim, n_actions, 0);
            tso.cogs.phi_gating = true;
            let mut total_reward = 0.0;

            for _ep in 0..n_episodes {
                let obs = env.reset();
                tso.end_episode();
                for _step in 0..100 {
                    let action = tso.step(&obs, 0.0, None, &[]);
                    let (r, _next_obs, done) = env.step(action);
                    total_reward += r;
                    if done { break; }
                }
            }
            actif_rewards.push(total_reward / n_episodes as f64);
        }
    }

    let pm = passif_rewards.iter().sum::<f64>() / n_seeds as f64;
    let am = actif_rewards.iter().sum::<f64>() / n_seeds as f64;
    println!("phi_passive: {:.3}", pm);
    println!("phi_active:   {:.3}", am);
    println!("delta:        {:+.3}", am - pm);
}
