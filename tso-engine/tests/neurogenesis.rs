// Tests d'intégration pour Sommeil Phase 3 — Neurogenèse structurelle
// Chaque test suit le cycle RED → GREEN → REFACTOR

use tso_engine::{CognitiveConfig, TsoEngine};

#[test]
fn test_neurogenesis_config_defaults() {
    // e10s01t01: Les nouveaux champs de CognitiveConfig ont les bonnes valeurs par défaut
    let cfg = CognitiveConfig::default();
    assert_eq!(
        cfg.sleep_neurogenesis_rate, 0.2,
        "sleep_neurogenesis_rate devrait être 0.2 par défaut"
    );
    assert_eq!(
        cfg.sleep_max_concepts, 50,
        "sleep_max_concepts devrait être 50 par défaut"
    );
    assert_eq!(
        cfg.sleep_maturation_cycles, 3,
        "sleep_maturation_cycles devrait être 3 par défaut"
    );
    assert!(
        cfg.sleep_synaptic_scaling,
        "sleep_synaptic_scaling devrait être true par défaut"
    );
}

#[test]
fn test_neurogenesis_concept_maturation_init() {
    // e10s01t02: concept_maturation est initialisé vide après new()
    let engine = TsoEngine::new(6, 4);
    assert!(
        engine.concept_maturation().is_empty(),
        "concept_maturation devrait être vide à l'initialisation"
    );
}

#[test]
fn test_neurogenesis_birth() {
    // e10s01t06: Phase 1.5 crée de nouveaux concepts pendant sleep_cycle()
    let mut engine = TsoEngine::new(6, 4);
    engine.cogs.sleep_neurogenesis_rate = 1.0; // forcer la neurogenèse
    engine.cogs.sleep_max_concepts = 50;
    engine.sleep_every_n_episodes = 0; // désactiver le sleep auto

    // Créer quelques concepts via des steps
    use ndarray::Array1;
    for _ in 0..20 {
        let obs = Array1::from_vec(vec![0.1; 6]);
        engine.step(&obs, 0.0, None, &[]);
    }

    let n_before = engine.num_concepts();
    let report = engine.sleep_cycle();
    let n_after = engine.num_concepts();

    assert!(
        report.prototypes_added > 0 || n_after > n_before,
        "La neurogenèse devrait créer au moins un nouveau concept (n_before={n_before}, n_after={n_after}, added={})",
        report.prototypes_added
    );
}
