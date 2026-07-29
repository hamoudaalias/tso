//! Benchmark: Φ gating vs passif sur MiniGrid DoorKey 7×7
//! Usage: cargo run --release --bin bench_phi_gating -- [--seeds N]

use tso_engine::{TsoEngine, minigrid_env::MiniGridEnv};

fn main() {
    let n_seeds = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let n_episodes = 100;
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
            tso.cogs.phi_threshold = 0.5;
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

    let pm = mean(&passif_rewards);
    let am = mean(&actif_rewards);
    let ps = std_dev(&passif_rewards, pm);
    let as_ = std_dev(&actif_rewards, am);
    println!("phi_passive (gating OFF):  {:.3} ± {:.3}", pm, ps);
    println!("phi_active  (gating ON):   {:.3} ± {:.3}", am, as_);
    println!("delta:                     {:.3}", am - pm);
}

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn std_dev(v: &[f64], m: f64) -> f64 {
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt()
}
