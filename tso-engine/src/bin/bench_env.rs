/// ════════════════════════════════════════════════════════════════════════════
///  bench_env — Benchmark Environment trait (GridWorld, Minigrid)
///
///  Mesure : µs/step, µs/reset, throughput (steps/s) pour chaque backend.
///  Sortie CSV : backend,step_latency_us,reset_latency_us,throughput
/// ════════════════════════════════════════════════════════════════════════════

use std::time::Instant;
use tso_engine::environment::{Environment, GridEnv};

fn bench<E: Environment>(env: &mut E, label: &str, n_steps: usize) {
    // Warmup
    env.reset();
    for _ in 0..10 { let _ = env.step(0); }

    // Reset latency
    let t0 = Instant::now();
    for _ in 0..100 { env.reset(); }
    let reset_us = t0.elapsed().as_nanos() as f64 / 100.0 / 1000.0; // µs

    // Step latency + throughput
    env.reset();
    let t0 = Instant::now();
    let mut total_steps = 0usize;
    for _ in 0..n_steps {
        env.reset();
        for _ in 0..100 {
            let r = env.step(0);
            total_steps += 1;
            if r.done { break; }
        }
    }
    let elapsed = t0.elapsed();
    let step_us = elapsed.as_nanos() as f64 / total_steps as f64 / 1000.0;
    let throughput = total_steps as f64 / elapsed.as_secs_f64();

    println!("{label},{step_us:.1},{reset_us:.1},{throughput:.0}");
    eprintln!("{label:>20}  step={step_us:>7.1} µs  reset={reset_us:>7.1} µs  throughput={throughput:>8.0} steps/s");
}

fn main() {
    println!("backend,step_latency_us,reset_latency_us,throughput");
    bench(&mut GridEnv::new(), "GridWorld 5×5", 1000);

    #[cfg(feature = "pyo3")] {
        // Minigrid via PyO3
        pyo3::prepare_freethreaded_python();
        let gil = pyo3::Python::acquire_gil();
        let minigrid = gil.python().import("minigrid").unwrap();
        let env_class = minigrid.getattr("EmptyEnv").unwrap();
        let mg_env = env_class.call1((8, 8)).unwrap();
        // MiniGrid 8×8: obs = 7×7×3 = 147D image
        struct MgWrapper {
            inner: pyo3::Py<any>, // fix with actual PyO3 bound
        }
        impl Environment for MgWrapper {
            fn reset(&mut self) -> Vec<f64> { vec![] }
            fn step(&mut self, _action: usize) -> tso_engine::environment::StepResult {
                tso_engine::environment::StepResult { observation: vec![], reward: 0.0, done: false }
            }
            fn action_space(&self) -> usize { 7 }
            fn observation_dim(&self) -> usize { 147 }
        }
        // TODO: proper PyO3 environment wrapper
        eprintln!("Minigrid: skip (PyO3 feature requires Python runtime)");
    }

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  5×5 trop petit pour scale test. Utiliser dim=64, 4096.             ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
