/// test_env_trait — Vérifie que le trait Environment fonctionne avec TsoEngine
///
/// 1. Crée un GridEnv via le trait
/// 2. Boucle 10 épisodes avec TsoEngine
/// 3. Mesure le succès

use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::environment::{Environment, GridEnv};

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  TEST — Environment trait + TsoEngine                              ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let mut engine = TsoEngine::with_hidden(6, 4, 4);
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    let mut env = GridEnv::new();
    let t0 = Instant::now();

    let mut total_r = 0.0;
    let mut steps = 0usize;
    let mut episodes = 0usize;

    for ep in 0..10 {
        let mut obs = Environment::reset(&mut env);
        engine.end_episode();

        let mut ep_r = 0.0;
        loop {
            let action = engine.step(
                &ndarray::Array1::from_vec(obs.clone()),
                0.0, None, &[],
            );
            let r = env.step(action);
            ep_r += r.reward;
            steps += 1;

            if r.done {
                engine.end_episode();
                if r.reward > 0.0 { total_r += 1.0; }
                break;
            }
            obs = r.observation;
        }
        episodes += 1;
    }

    let elapsed = t0.elapsed();
    let success = total_r / episodes as f64 * 100.0;
    eprintln!("Épisodes: {episodes}, steps: {steps}, succès: {success:.0}%, temps: {:.1?}", elapsed);
    eprintln!("✅ Environment trait fonctionne avec TsoEngine");
}
