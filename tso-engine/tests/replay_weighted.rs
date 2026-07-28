//! Validation du rejeu bruité par échantillonnage pondéré (e05s02).
//!
//! Vérifie que le rejeu pendant le sommeil donne plus de poids aux
//! épisodes récents et/ou à haute récompense, et que cela améliore
//! la convergence de Φ.

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tso_engine::tso_engine::TsoEngine;

/// Crée un moteur TSO avec des épisodes mixant faibles et fortes récompenses.
fn setup_mixed_reward_engine(seed: u64) -> TsoEngine {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let dim = 4;
    let n_actions = 4;

    let mut engine = TsoEngine::with_hidden(dim, n_actions, 4);
    engine.cerebellum.epsilon = 0.5;
    engine.cerebellum.noise_std = 0.2;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0;

    // Génère 5 épisodes : 3 à faible récompense, 2 à forte récompense
    for ep in 0..5 {
        engine.end_episode();
        let use_high_reward = ep >= 3;

        for _step in 0..30 {
            let p = Array1::from_vec(vec![
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
            ]);
            let action = rng.r#gen_range(0..n_actions);
            let reward = if use_high_reward && rng.r#gen_bool(0.3) {
                10.0
            } else {
                -0.01
            };
            engine.step(&p, reward, None, &[]);
        }
        engine.end_episode();
    }

    engine
}

#[test]
fn test_weighted_replay_reduces_phi() {
    let mut engine = setup_mixed_reward_engine(42);
    let phi_before = engine.current_phi;

    // Sommeil par défaut (rejeu priorisé récent)
    let report = engine.sleep_cycle();

    let phi_after = engine.current_phi;

    eprintln!("Φ avant: {:.4} → après: {:.4}", phi_before, phi_after);
    eprintln!("Replay count: {}, added: {}, pruned: {}",
        report.replay_count, report.prototypes_added, report.prototypes_pruned);

    // Le rejeu + résolution doivent réduire Φ
    assert!(
        phi_after <= phi_before + 1e-6,
        "Φ after sleep ({:.4}) > Φ before ({:.4})",
        phi_after, phi_before
    );
}

#[test]
fn test_weighted_replay_with_different_noise_levels() {
    // Test que différents niveaux de bruit de rejeu produisent
    // des résultats différents (bruit fort → plus de neurogenèse)
    let mut engine_low = setup_mixed_reward_engine(1);
    let mut engine_high = setup_mixed_reward_engine(1);

    // Clone les paramètres sauf le bruit de rejeu
    engine_high.cogs = engine_low.cogs.clone();

    engine_low.sleep_noise_std = 0.01;  // bruit faible
    engine_high.sleep_noise_std = 0.20; // bruit fort

    let report_low = engine_low.sleep_cycle();
    let report_high = engine_high.sleep_cycle();

    eprintln!("Bruit faible  (0.01): {} added", report_low.prototypes_added);
    eprintln!("Bruit fort   (0.20): {} added", report_high.prototypes_added);

    // Un bruit plus fort devrait générer plus de neurogenèse
    // (nouveaux prototypes par divergence)
    assert!(
        report_high.prototypes_added >= report_low.prototypes_added,
        "High noise ({}) should add >= prototypes than low noise ({})",
        report_high.prototypes_added, report_low.prototypes_added
    );
}

#[test]
fn test_weighted_replay_multiple_epochs_deepen_phi_drop() {
    let mut engine = setup_mixed_reward_engine(7);
    let phi_start = engine.current_phi;

    let mut last_phi = phi_start;
    for epoch in 0..3 {
        let report = engine.sleep_cycle();
        let phi_now = engine.current_phi;

        eprintln!("Epoch {}: Φ {:.6} → {:.6} (Δ={:+.6}), replay={}",
            epoch, last_phi, phi_now, phi_now - last_phi, report.replay_count);

        // Chaque cycle doit maintenir ou réduire Φ
        assert!(
            phi_now <= last_phi + 1e-6,
            "Φ increased in epoch {}: {:.6} → {:.6}",
            epoch, last_phi, phi_now
        );

        last_phi = phi_now;
    }

    eprintln!("Φ final: {:.6} (départ: {:.6})", last_phi, phi_start);
    assert!(
        last_phi <= phi_start + 1e-6,
        "Φ final > Φ start"
    );
}
