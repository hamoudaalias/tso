use tso_engine::tso_engine::TsoEngine;
use tso_engine::grid_world::GridWorld;

fn bfs_val(env: &GridWorld) -> Option<f64> {
    env.bfs_at_current_pos().map(|d| (20.0 - 0.5 * d as f64).max(0.0))
}

fn run_ep(engine: &mut TsoEngine, env: &mut GridWorld, noise: f64) -> f64 {
    engine.cerebellum.noise_std = noise;
    env.reset();
    let mut total = 0.0;
    let mut p = env.perception();
    let mut a = engine.step(&p, 0.0, bfs_val(env), &env.bfs_gradient());
    while !env.done {
        let r = env.step(a);
        total += r;
        if env.done {
            engine.step(&env.perception(), r, bfs_val(env), &env.bfs_gradient());
            break;
        }
        p = env.perception();
        a = engine.step(&p, r, bfs_val(env), &env.bfs_gradient());
    }
    engine.end_episode();
    total
}

fn main() {
    let n_episodes = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let path = std::env::args().nth(2)
        .unwrap_or_else(|| "tso_model.bin".to_string());

    eprintln!("Pretraining TSO Engine on Zigzag for {} episodes...", n_episodes);
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    let mut env = GridWorld::corridor();

    for ep in 1..=n_episodes {
        run_ep(&mut engine, &mut env, 0.1);
        if ep % 50 == 0 {
            let mut test_env = GridWorld::corridor();
            let rate = (0..100).filter(|_| run_ep(&mut engine, &mut test_env, 0.0) > 0.0).count();
            eprintln!("ep={:4}  test={}%  conc={}  edges={}  Φ={:.3}  osc={}",
                ep, rate, engine.num_concepts(), engine.graph_edges(), engine.current_phi,
                engine.oscillation_breaks);
        }
    }

    let mut test_env = GridWorld::corridor();
    let final_rate = (0..200).filter(|_| run_ep(&mut engine, &mut test_env, 0.0) > 0.0).count();
    eprintln!("Final test: {}% ({} / 200)", final_rate as f64 / 2.0, final_rate);

    engine.save(&path).expect("Failed to save model");
    eprintln!("Model saved to {}", path);
}
