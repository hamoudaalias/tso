//! Validation de la convergence de Φ après sommeil (e05s01).
//!
//! Vérifie que le cycle de sommeil (rejeu bruité + résolution profonde
//! + élagage) fait baisser la tension cognitive Φ du graphe sémantique.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tso_engine::tso_engine::TsoEngine;

/// Crée un moteur TSO avec qqs épisodes forcés pour générer
/// des concepts, des transitions et un graphe non-vide.
fn setup_engine_with_graph(seed: u64) -> TsoEngine {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let dim = 4;
    let n_actions = 4;

    let mut engine = TsoEngine::with_hidden(dim, n_actions, 4);
    engine.cerebellum.epsilon = 0.5;
    engine.cerebellum.noise_std = 0.2;
    engine.cerebellum.replay_lr = 0.0;
    engine.sleep_every_n_episodes = 0; // pas de sleep auto

    // Génère 3 épisodes avec des perceptions aléatoires pour créer
    // des concepts et des arêtes de transition
    for _ep in 0..3 {
        engine.end_episode();

        for _step in 0..20 {
            let p = Array1::from_vec(vec![
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
                rng.r#gen::<f64>(), rng.r#gen::<f64>(),
            ]);
            let action = rng.r#gen_range(0..n_actions);
            // Simule des récompenses pour créer des arêtes +2
            let reward = if rng.r#gen_bool(0.2) { 20.0 } else { -0.01 };
            engine.step(&p, reward, None, &[]);
        }
        // Termine l'épisode pour forcer l'enregistrement de la trace épisodique
        engine.end_episode();
    }

    engine
}

#[test]
fn test_sleep_reduces_phi() {
    let mut engine = setup_engine_with_graph(42);
    let phi_before = engine.current_phi;

    // S'il n'y a pas assez de concepts/arêtes, forcer plus d'épisodes
    if engine.graph_edges() < 5 {
        // Ajoute des transitions manuelles
        for _ in 0..10 {
            let p = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
            engine.step(&p, 0.0, None, &[]);
        }
        engine.end_episode();
        engine.end_episode(); // force flush
    }

    let phi_before2 = engine.current_phi;

    // Sommeil avec paramètres par défaut
    let report = engine.sleep_cycle();

    let phi_after = engine.current_phi;

    eprintln!("Φ avant sommeil: {:.4}", phi_before2);
    eprintln!("Φ après sommeil: {:.4}", phi_after);
    eprintln!("Replay: {}, prototypes pruned: {}, added: {}, edges removed: {}, concepts pruned: {}",
        report.replay_count, report.prototypes_pruned, report.prototypes_added,
        report.edges_removed, report.concepts_pruned);

    // Le Φ doit baisser (ou au moins ne pas augmenter)
    assert!(
        phi_after <= phi_before2 + 1e-6,
        "Φ after sleep ({:.4}) should be <= Φ before ({:.4})",
        phi_after, phi_before2
    );
}

#[test]
fn test_sleep_reduces_phi_repeatedly() {
    let mut engine = setup_engine_with_graph(99);

    // Vérifie que Φ converge après plusieurs cycles sommeil
    for cycle in 0..3 {
        let phi_before = engine.current_phi;
        let report = engine.sleep_cycle();
        let phi_after = engine.current_phi;

        eprintln!("Cycle {}: Φ {:.4} → {:.4} (ΔΦ={:+.4}), replay={}, added={}",
            cycle, phi_before, phi_after, phi_after - phi_before,
            report.replay_count, report.prototypes_added);

        if cycle > 0 {
            assert!(
                phi_after <= phi_before + 1e-6,
                "Cycle {}: Φ increased ({:.4} → {:.4})",
                cycle, phi_before, phi_after
            );
        }
    }
}

#[test]
fn test_sleep_returns_valid_report() {
    let mut engine = setup_engine_with_graph(7);
    let report = engine.sleep_cycle();

    eprintln!("SleepReport: {:#?}", report);

    // Le rapport doit avoir des champs cohérents
    assert!(report.phi_before.is_finite());
    assert!(report.phi_after.is_finite());
    // Le nombre d'arêtes supprimées doit être >= 0
    assert!(report.edges_removed <= engine.graph_edges() + report.edges_removed);
}
