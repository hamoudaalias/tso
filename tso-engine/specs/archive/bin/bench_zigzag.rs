/// Benchmark attention sur Zigzag 10×10 — compare TSO avec/sans attention.
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;
use std::time::Instant;

fn run_trial(attention_on: bool, seeds: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = TsoEngine::with_hidden(5, 4, 16);
        engine.cogs = CognitiveConfig {
            attention: attention_on,
            ..CognitiveConfig::default()
        };
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
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
        eprintln!("  {} attention seed {}/{}... {:.1}% [{:.1?}]",
            if attention_on { "Avec" } else { "Sans" },
            seed + 1, seeds, rates[seed], t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  BENCH ZIGZAG 10×10 — Attention spatiale (δ-clip + replay)          ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 10;
    let (m_sans, s_sans) = run_trial(false, seeds);
    let (m_avec, s_avec) = run_trial(true, seeds);

    eprintln!("\n  {}", "=".repeat(50));
    eprintln!("  Résultats ({} seeds) :", seeds);
    eprintln!("  Sans attention        μ={:6.1}%  σ={:.2}%", m_sans, s_sans);
    eprintln!("  Avec attention        μ={:6.1}%  σ={:.2}%", m_avec, s_avec);
    let gain = m_avec - m_sans;
    eprintln!("  Gain attentionnel     {:+.1}%", gain);
    if gain > 2.0 { eprintln!("  ✅ L'attention améliore significativement"); }
    else if gain < -2.0 { eprintln!("  ❌ L'attention dégrade"); }
    else { eprintln!("  ⏸️  Pas de différence significative"); }
}
