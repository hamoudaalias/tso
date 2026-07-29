/// Benchmark : GridCells activées vs désactivées sur Zigzag 10×10.
///
/// GridCells ajoute cell_id ∈ [0,1] à la perception, résolvant l'aliasing.
/// On construit la perception 5D → on ajoute cell_id en 6e dimension si actif.
use tso_engine::tso_engine::TsoEngine;
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;
use tso_engine::grid_cells::GridCells;
use ndarray::Array1;
use std::time::Instant;

fn run_trial(gridcells_on: bool, seeds: usize) -> (f64, f64) {
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut cells = GridCells::new(10, 10);
        let dim = if gridcells_on { 6 } else { 5 };
        let mut engine = TsoEngine::with_hidden(dim, 4, 16);
        engine.cogs = CognitiveConfig { attention: true, ..CognitiveConfig::default() };
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
            let raw_obs = env.reset(); engine.end_episode();
            let mut obs = if gridcells_on {
                let cell_val = cells.cell_id(env.agent.0, env.agent.1);
                let mut v = raw_obs.to_vec();
                v.push(cell_val);
                Array1::from_vec(v)
            } else { raw_obs };
            loop {
                let action = engine.step(&obs, 0.0, None, &[]);
                let (rew, next_raw) = env.step_env(action);
                if env.done {
                    if rew > 0.0 { successes += 1; }
                    engine.end_episode();
                    break;
                }
                obs = if gridcells_on {
                    let cell_val = cells.cell_id(env.agent.0, env.agent.1);
                    let mut v = next_raw.to_vec();
                    v.push(cell_val);
                    Array1::from_vec(v)
                } else { next_raw };
            }
            if ep_i < 100 {
                let frac = ep_i as f64 / 100.0;
                engine.cerebellum.epsilon = 0.8 * (1.0 - frac);
            } else { engine.cerebellum.epsilon = 0.01; }
        }
        rates.push(successes as f64 / ep as f64 * 100.0);
        let label = if gridcells_on { "GridCells ON" } else { "GridCells OFF" };
        eprintln!("  {label:13} seed {}/{}... {:.1}% [{:.1?}]",
            seed + 1, seeds, rates[seed], t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Zigzag 10×10 — GridCells ON vs OFF (δ-clip + replay)               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 5;
    let (m_off, s_off) = run_trial(false, seeds);
    let (m_on, s_on) = run_trial(true, seeds);

    eprintln!("\n  {}", "=".repeat(50));
    eprintln!("  GridCells OFF    μ={:6.1}%  σ={:.2}%", m_off, s_off);
    eprintln!("  GridCells ON     μ={:6.1}%  σ={:.2}%", m_on, s_on);
    let gain = m_on - m_off;
    eprintln!("  Gain              {:+.1}%", gain);
    if gain > 5.0 { eprintln!("  ✅ GridCells résout l'aliasing significativement"); }
    else if gain > 2.0 { eprintln!("  👍 Gain modéré"); }
    else { eprintln!("  ⏸️  Aliasing non dominant ou GridCells inefficace"); }
}
