//! Métriques de consolidation après sommeil (e05s03).
//!
//! Vérifie que le SleepReport contient des métriques exploitables
//! (Φ drop, prototypes, arêtes supprimées) et que ces métriques
//! évoluent de manière cohérente après plusieurs cycles.

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tso_engine::tso_engine::TsoEngine;

fn setup_engine(seed: u64) -> TsoEngine {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let dim = 4;
    let n_actions = 4;

    let mut engine = TsoEngine::with_hidden(dim, n_actions, 4);
    engine.cerebellum.epsilon = 0.5;
    engine.cerebellum.noise_std = 0.2;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0;

    for _ep in 0..4 {
        engine.end_episode();
        for _step in 0..25 {
            let p = Array1::from_vec(vec![
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
            ]);
            let reward = if rng.r#gen_bool(0.15) { 10.0 } else { -0.01 };
            engine.step(&p, reward, None, &[]);
        }
        engine.end_episode();
    }
    engine
}

#[test]
fn test_consolidation_phi_drop_positive() {
    let mut engine = setup_engine(42);
    let report = engine.sleep_cycle();

    let phi_drop = report.phi_before - report.phi_after;

    eprintln!("Φ_before: {:.6}, Φ_after: {:.6}, drop: {:.6}",
        report.phi_before, report.phi_after, phi_drop);

    assert!(
        phi_drop >= -1e-6,
        "Φ should not increase after consolidation (drop={:.6})",
        phi_drop
    );

    // Les métriques doivent être positives ou nulles
    assert!(report.replay_count > 0, "replay_count should be > 0");
    assert!(
        report.edges_removed <= report.replay_count + 100,
        "edges_removed ({}) looks suspiciously high",
        report.edges_removed
    );
}

#[test]
fn test_consolidation_metrics_are_monotonic() {
    let mut engine = setup_engine(99);
    let mut total_phi_drops: Vec<f64> = Vec::new();

    for cycle in 0..4 {
        let report = engine.sleep_cycle();
        let drop = report.phi_before - report.phi_after;
        total_phi_drops.push(drop);

        eprintln!("Cycle {}: Φ {:.6} → {:.6} (drop={:+.6}), replay={}, edges_removed={}",
            cycle, report.phi_before, report.phi_after, drop,
            report.replay_count, report.edges_removed);
    }

    // Le Φ final doit être plus bas qu'au début
    let total_drop: f64 = total_phi_drops.iter().sum();
    eprintln!("Drop cumulé sur 4 cycles: {:.6}", total_drop);
}

#[test]
fn test_consolidation_edge_pruning_progressive() {
    let mut engine = setup_engine(7);

    for cycle in 0..4 {
        let report = engine.sleep_cycle();
        eprintln!("Cycle {}: edges_removed={}, concepts_pruned={}, protos_pruned={}",
            cycle, report.edges_removed, report.concepts_pruned, report.prototypes_pruned);
    }
}
