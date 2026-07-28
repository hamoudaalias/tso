/// TSO sur Minigrid via PyO3 — Évaluation réelle
///
/// Boucle TSO complète sur un environnement Minigrid (EmptyEnv-8×8)
/// via le trait Environment + tso_env MinigridEnv.
///
/// Métriques : succès ε=0, steps/episode, latence step().
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::environment::{Environment, StepResult};
use tso_engine::CognitiveConfig;
use ndarray::Array1;
use pyo3::prelude::*;

/// Wrapper PyO3 pour Minigrid implémentant le trait Environment
struct MinigridTrait {
    inner: Py<PyAny>,
    act_space: usize,
    obs_dim: usize,
}

impl MinigridTrait {
    fn new(env_name: &str, width: usize) -> Self {
        Python::with_gil(|py| {
            let minigrid = py.import("minigrid").unwrap();
            let cls = minigrid.getattr(env_name).unwrap();
            let env = cls.call1((width,)).unwrap();
            let act_space: usize = env.getattr("action_space").unwrap()
                .getattr("n").unwrap().extract().unwrap();
            let obs_space = env.getattr("observation_space").unwrap();
            let shape: Vec<usize> = obs_space.getattr("shape").unwrap().extract().unwrap();
            let obs_dim: usize = shape.iter().product();
            MinigridTrait { inner: env.into(), act_space, obs_dim }
        })
    }
}

impl Environment for MinigridTrait {
    fn reset(&mut self) -> Array1<f64> {
        Python::with_gil(|py| {
            let env = self.inner.bind(py);
            let result = env.call_method1("reset", ((),)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                Array1::from_vec(arr.iter().flat_map(|r| r.iter().flat_map(|c| c.iter())).copied().collect())
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() { Array1::from_vec(v) }
            else { Array1::zeros(self.obs_dim) }
        })
    }

    fn step(&mut self, action: usize) -> StepResult {
        Python::with_gil(|py| {
            let env = self.inner.bind(py);
            let result = env.call_method1("step", (action as i64,)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            let obs = if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                Array1::from_vec(arr.iter().flat_map(|r| r.iter().flat_map(|c| c.iter())).copied().collect())
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() { Array1::from_vec(v) }
            else { Array1::zeros(self.obs_dim) };
            let reward = result.get_item(1).unwrap().extract::<f64>().unwrap_or(0.0);
            let done = result.get_item(2).unwrap().extract::<bool>().unwrap_or(false);
            StepResult { observation: obs, reward, done }
        })
    }

    fn action_space(&self) -> usize { self.act_space }
    fn observation_dim(&self) -> usize { self.obs_dim }
}

fn main() {
    pyo3::prepare_freethreaded_python();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  TSO × Minigrid — EmptyEnv-8×8 via PyO3                            ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  TSO complet, δ-clip=5.0, use_stationary_reward, AttractorEncoder  ║");
    eprintln!("║  Pas de BFS (environnement visuel, pas de moustaches)              ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let dim = 147; // 7×7×3
    let n_actions = 7;
    let mut engine = TsoEngine::with_hidden(dim, n_actions, 16);
    engine.cerebellum.epsilon = 0.2;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    let mut env = MinigridTrait::new("EmptyEnv", 8);
    let act = env.action_space();
    let obs_dim = env.observation_dim();
    eprintln!("Environment: EmptyEnv-8×8, actions={act}, obs_dim={obs_dim}");
    eprintln!();

    const EPISODES: usize = 50;
    let t0 = Instant::now();
    let mut successes = 0usize;
    let mut total_steps = 0usize;
    let mut step_times = Vec::new();

    for ep in 1..=EPISODES {
        let mut obs = env.reset();
        engine.end_episode();
        let mut ep_steps = 0usize;
        let mut ep_reward = 0.0;

        loop {
            let t_step = Instant::now();
            let action = engine.step(&obs, 0.0, None, &[]);
            let r = env.step(action);
            step_times.push(t_step.elapsed());
            ep_reward += r.reward;
            ep_steps += 1;

            if r.done {
                if r.reward > 0.0 { successes += 1; }
                engine.end_episode();
                break;
            }
            obs = r.observation;
        }
        total_steps += ep_steps;

        if ep % 10 == 0 || ep == 1 {
            let rate = successes as f64 / ep as f64 * 100.0;
            eprintln!("  ep={ep:3} succès={rate:5.1}% steps={ep_steps:3} reward={ep_reward:+.1} (total={successes}/{ep})");
        }
    }

    let elapsed = t0.elapsed();
    let success_rate = successes as f64 / EPISODES as f64 * 100.0;
    let avg_step = step_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / step_times.len() as f64;

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSULTATS                                                           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Succès ε-greedy (0.2): {success_rate:.1}% ({successes}/{EPISODES})    ");
    eprintln!("║  Steps total: {total_steps}  temps: {elapsed:.1?}");
    eprintln!("║  Step moyen: {:.1?}  (dont PyO3 overhead)", avg_step);
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
