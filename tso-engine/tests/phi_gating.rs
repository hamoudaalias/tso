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
fn test_default_config_minimal() {
    // Vérifie que seuls les composants validés par ablation sont actifs par défaut
    let cfg = CognitiveConfig::default();
    assert!(cfg.attractor, "AttractorField validé (d=2.59)");
    assert!(cfg.graph_phi, "Graphe Φ partiellement validé");
    assert!(!cfg.attention, "Attention non validée sur MiniGrid");
    assert!(!cfg.episodic_curiosity, "Épisodique non validé");
    assert!(!cfg.metabolic_cost, "Coût métabolique non validé");
    assert!(cfg.hypothalamus, "Hypothalamus doit être activé par défaut (Phase 1)");
    assert!(!cfg.rstdp_enabled, "R-STDP non activé");
    assert!(!cfg.use_fpi, "FPI non activé par défaut");
    assert_eq!(cfg.sleep_neurogenesis_rate, 0.02, "Neurogenèse active (Phase 2)");
    assert_eq!(cfg.sleep_maturation_cycles, 3, "Maturation active (Phase 2)");
    assert!(cfg.sleep_synaptic_scaling, "Scaling synaptique actif (Phase 2)");
    assert!(cfg.autonomous_sleep, "Sommeil autonome actif (Phase 2)");
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
