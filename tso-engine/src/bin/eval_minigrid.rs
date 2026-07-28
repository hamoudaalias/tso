/// TSO × Minigrid via PyO3 — Évaluation réelle

#[cfg(not(feature = "interop"))]
fn main() { eprintln!("interop feature required: cargo run --features interop --bin eval_minigrid"); }

#[cfg(feature = "interop")]
mod impl_ {
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::environment::{Environment, StepResult};
use ndarray::Array1;
use pyo3::prelude::*;

struct MinigridTrait {
    inner: Py<PyAny>,
    act_space: usize, obs_dim: usize,
}

impl Environment for MinigridTrait {
    fn reset(&mut self) -> Array1<f64> {
        Python::attach(|py| {
            let e = self.inner.bind(py);
            let r = e.call_method0("reset").unwrap();
            let o = r.get_item(0).unwrap();
            if let Ok(a) = o.extract::<Vec<Vec<Vec<f64>>>>() { Array1::from_vec(a.iter().flat_map(|r| r.iter().flat_map(|c| c.iter())).copied().collect()) }
            else if let Ok(v) = o.extract::<Vec<f64>>() { Array1::from_vec(v) } else { Array1::zeros(self.obs_dim) }
        })
    }
    fn step(&mut self, action: usize) -> StepResult {
        Python::attach(|py| {
            let e = self.inner.bind(py);
            let r = e.call_method1("step", (action as i64,)).unwrap();
            let o = r.get_item(0).unwrap();
            let obs = if let Ok(a) = o.extract::<Vec<Vec<Vec<f64>>>>() { Array1::from_vec(a.iter().flat_map(|r| r.iter().flat_map(|c| c.iter())).copied().collect()) }
            else if let Ok(v) = o.extract::<Vec<f64>>() { Array1::from_vec(v) } else { Array1::zeros(self.obs_dim) };
            let reward = r.get_item(1).unwrap().extract::<f64>().unwrap_or(0.0);
            let done = r.get_item(2).unwrap().extract::<bool>().unwrap_or(false);
            StepResult { observation: obs, reward, done }
        })
    }
    fn action_space(&self) -> usize { self.act_space }
    fn observation_dim(&self) -> usize { self.obs_dim }
}

pub fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  TSO × Minigrid — EmptyEnv-8×8 via PyO3                   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    let (mut env, act_space, obs_dim) = Python::attach(|py| {
        let mg = py.import("minigrid").unwrap();
        let cls = mg.getattr("envs").unwrap().getattr("EmptyEnv").unwrap();
        let e = cls.call1((8,)).unwrap();
        let a: usize = e.getattr("action_space").unwrap().getattr("n").unwrap().extract().unwrap();
        let od = 147usize; // 7×7×3 pour EmptyEnv-8
        let new_env = MinigridTrait { inner: e.into(), act_space: a, obs_dim: od };
        (new_env, a, od)
    });

    // Charger le VAE pré-entraîné
    use tso_engine::encoder::{Encoder, VaeEncoder};
    use std::fs;
    let vae_bytes = fs::read("vae_weights.bin").unwrap_or_else(|_| {
        eprintln!("⚠  vae_weights.bin not found, using AttractorField");
        vec![]
    });
    let mut engine = TsoEngine::with_hidden(obs_dim, act_space, 16);
    engine.cerebellum.epsilon = 0.2; engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.0; engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;
    engine.cogs.delta_clip_max = 5.0;

    if !vae_bytes.is_empty() {
        let vae: tso_engine::vae::Vae = bincode::deserialize(&vae_bytes).unwrap();
        let mut vae_enc = VaeEncoder::new(obs_dim, 32, 8, 0.5);
        vae_enc.vae = vae;
        vae_enc.deterministic = true;
        vae_enc.freeze = true;
        engine.encoder = Some(Box::new(vae_enc));
        eprintln!("VAE chargé et gelé (déterministe, freeze)");
    } else {
        engine.encoder = Some(Box::new(tso_engine::encoder::AttractorEncoder::new(obs_dim)));
        eprintln!("Utilise AttractorEncoder (pas de VAE)");
    }
    engine.cogs.delta_clip_max = 5.0;

    let t0 = Instant::now();
    let mut successes = 0usize; const EP: usize = 50;

    for ep in 1..=EP {
        let mut obs = env.reset(); engine.end_episode();
        let mut step_count = 0;
        loop {
            let action = engine.step(&obs, 0.0, None, &[]);
            step_count += 1;
            let r = env.step(action);
            if r.done { if r.reward > 0.0 { successes += 1; } engine.end_episode(); break; }
            obs = r.observation;
        }
        eprintln!("  ep={ep:3} succès={:.1}% steps={step_count:3}", successes as f64 / ep as f64 * 100.0);
    }

    let rate = successes as f64 / EP as f64 * 100.0;
    eprintln!("  Total: {successes}/{EP} = {rate:.1}% en {:.1?}", t0.elapsed());
}
}
