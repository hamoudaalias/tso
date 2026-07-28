/// MinigridEnv — Wrapper PyO3 pour les environnements Python Minigrid.
///
/// Implémente le trait Environment de tso-engine.
/// step(action) → (obs, reward, done) via PyO3 → Python Minigrid.
///
/// Usage:
/// ```python
/// from tso_env import MinigridEnv
/// env = MinigridEnv("EmptyEnv", 8, 8)
/// obs, info = env.reset()
/// obs, reward, done, truncated, info = env.step(0)
/// ```

use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::Python;
use tso_engine::environment::{Environment, StepResult};


/// Wrapper PyO3 : minigrid.XXEnv(8, 8) → Environment trait.
#[pyclass(unsendable)]
pub struct MinigridEnv {
    py_env: Py<PyAny>,
    action_space: usize,
    obs_dim: usize,
}

#[pymethods]
impl MinigridEnv {
    #[new]
    fn new(env_name: &str, width: usize, height: Option<usize>) -> PyResult<Self> {
        Python::attach(|py| {
            let minigrid = py.import("minigrid")?;
            let env_class = minigrid.getattr(env_name)?;
            let env = match height {
                Some(h) => env_class.call1((width, h))?,
                None => env_class.call1((width,))?,
            };
            let action_space: usize = env.getattr("action_space")?
                .getattr("n")?.extract()?;
            // Minigrid obs est un dict avec 'image' = (W, H, 3) ndarray
            let obs_space = env.getattr("observation_space")?;
            let obs_shape: Vec<usize> = obs_space.getattr("shape")?.extract()?;
            let obs_dim: usize = obs_shape.iter().product();

            let py_env: Py<PyAny> = env.unbind();
            Ok(MinigridEnv {
                py_env,
                action_space,
                obs_dim,
            })
        })
    }

    fn reset(&mut self) -> (Vec<f64>, Vec<(String, f64)>) {
        Python::attach(|py| {
            let result = self.py_env.bind(py).call_method1("reset", ((),)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            let obs_flat: Vec<f64> = if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                // obs shape (H, W, 3) → flatten
                arr.iter().flat_map(|row| row.iter().flat_map(|ch| ch.iter())).copied().collect()
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() {
                v
            } else {
                vec![0.0; self.obs_dim]
            };
            (obs_flat, vec![])
        })
    }

    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool, bool, Vec<(String, f64)>) {
        Python::attach(|py| {
            let result = self.py_env.bind(py).call_method1("step", (action as i64,)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            let obs_flat: Vec<f64> = if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                arr.iter().flat_map(|row| row.iter().flat_map(|ch| ch.iter())).copied().collect()
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() {
                v
            } else {
                vec![0.0; self.obs_dim]
            };
            let reward = result.get_item(1).unwrap().extract::<f64>().unwrap_or(0.0);
            let done = result.get_item(2).unwrap().extract::<bool>().unwrap_or(false);
            (obs_flat, reward, done, false, vec![])
        })
    }

    fn render(&self) {
        Python::attach(|py| {
            let _ = self.py_env.bind(py).call_method0("render");
        });
    }

    #[getter]
    fn get_action_space(&self) -> usize { self.action_space }
    #[getter]
    fn get_obs_dim(&self) -> usize { self.obs_dim }
}

// Implémentation du trait Environment de tso-engine
impl Environment for MinigridEnv {
    fn reset(&mut self) -> Array1<f64> {
        Python::attach(|py| {
            let result = self.py_env.bind(py).call_method1("reset", ((),)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            let obs_flat: Vec<f64> = if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                arr.iter().flat_map(|row| row.iter().flat_map(|ch| ch.iter())).copied().collect()
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() {
                v
            } else {
                vec![0.0; self.obs_dim]
            };
            Array1::from_vec(obs_flat)
        })
    }

    fn step(&mut self, action: usize) -> StepResult {
        Python::attach(|py| {
            let result = self.py_env.bind(py).call_method1("step", (action as i64,)).unwrap();
            let obs_py = result.get_item(0).unwrap();
            let obs_flat: Vec<f64> = if let Ok(arr) = obs_py.extract::<Vec<Vec<Vec<f64>>>>() {
                arr.iter().flat_map(|row| row.iter().flat_map(|ch| ch.iter())).copied().collect()
            } else if let Ok(v) = obs_py.extract::<Vec<f64>>() {
                v
            } else {
                vec![0.0; self.obs_dim]
            };
            let reward = result.get_item(1).unwrap().extract::<f64>().unwrap_or(0.0);
            let done = result.get_item(2).unwrap().extract::<bool>().unwrap_or(false);
            StepResult {
                observation: Array1::from_vec(obs_flat),
                reward,
                done,
            }
        })
    }

    fn action_space(&self) -> usize { self.action_space }
    fn observation_dim(&self) -> usize { self.obs_dim }
}

/// Module Python `tso_env`.
#[pymodule]
fn tso_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MinigridEnv>()?;
    Ok(())
}
