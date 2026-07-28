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
