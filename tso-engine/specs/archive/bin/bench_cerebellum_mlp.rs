/// Benchmark : Cerebellum linéaire vs MLP (16, 32, 64) sur Zigzag 10×10.
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;
use std::time::Instant;

fn run_trial(hidden: usize, seeds: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = TsoEngine::with_hidden(5, 4, hidden);
        engine.cogs = CognitiveConfig { attention: true, ..CognitiveConfig::default() };
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.cerebellum.lr = 0.1;
        engine.use_stationary_reward = true;
        engine.sleep_every_n_episodes = 0;

        let t0 = Instant::now();
        let mut successes = 0;
        let ep = 200;
        for ep_i in 1..=ep {
            let mut obs = env.reset(); engine.end_episode();
            loop {
                let action = engine.step(&obs, 0.0, None, &[]);
                let (rew, next_obs) = env.step_env(action);
                if env.done { if rew > 0.0 { successes += 1; } engine.end_episode(); break; }
                obs = next_obs;
            }
            if ep_i < 100 {
                let frac = ep_i as f64 / 100.0;
                engine.cerebellum.epsilon = 0.8 * (1.0 - frac);
            } else { engine.cerebellum.epsilon = 0.01; }
        }
        rates.push(successes as f64 / ep as f64 * 100.0);
        eprintln!("  hidden={:3} seed {}/{}... {:.1}% [{:.1?}]",
            hidden, seed + 1, seeds, rates[seed], t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Zigzag 10×10 — Cerebellum linéaire vs MLP (replay + δ-clip)        ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 5;
    for &hd in &[0, 8, 16, 32] {
        let label = if hd == 0 { "linéaire" } else { &format!("MLP {hd}") };
        let (m, s) = run_trial(hd, seeds);
        eprintln!("  {label:>12}  μ={:6.1}%  σ={:.2}%", m, s);
    }
}
