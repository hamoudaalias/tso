//! Tests pour le gating par Φ
use tso_engine::{CognitiveConfig, TsoEngine};

#[test]
fn test_compute_phi_empty() {
    let tso = TsoEngine::new(10, 4);
    assert_eq!(tso.graph.compute_phi(), 0.0);
}

#[test]
fn test_phi_gating_off_by_default() {
    let cfg = CognitiveConfig::default();
    assert!(!cfg.phi_gating, "phi_gating doit être false par défaut");
}

#[test]
fn test_phi_gating_config() {
    let mut cogs = CognitiveConfig::default();
    cogs.phi_gating = true;
    cogs.phi_threshold = 0.5;
    assert!(cogs.phi_gating);
    assert!((cogs.phi_threshold - 0.5).abs() < 1e-6);
}

#[test]
fn test_step_phi_gating() {
    // Vérifie que step() ne panique pas avec phi_gating activé
    let mut cogs = CognitiveConfig::default();
    cogs.phi_gating = true;
    let mut tso = TsoEngine::with_hidden(10, 4, 0);
    tso.cogs = cogs;
    let obs = ndarray::Array1::zeros(10);
    let action = tso.step(&obs, 0.0, None, &[]);
    assert!(action < 4);
}


#[test]
fn test_phi_gate_skip_executes_without_panic() {
    // Vérifie que phi_gating=true ne panique pas et qu'un step s'exécute.
    let mut cogs = CognitiveConfig::default();
    cogs.phi_gating = true;
    cogs.phi_threshold = 0.5;
    let mut tso = TsoEngine::with_hidden(10, 4, 0);
    tso.cogs = cogs;
    let obs = ndarray::Array1::zeros(10);
    for _ in 0..50 {
        let action = tso.step(&obs, 0.0, None, &[]);
        assert!(action < 4);
    }
}
