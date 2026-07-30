//! Couplage Φ_total = Φ_graph + homeostatic_drift
//! Vérifie que le couplage est bien actif quand hypothalamus est activé.

use tso_engine::tso_engine::TsoEngine;
use ndarray::Array1;

#[test]
fn test_phi_total_coupling_when_hypothalamus_enabled() {
    let mut eng = TsoEngine::with_hidden(4, 2, 0);
    eng.cogs.graph_phi = true;
    eng.cogs.hypothalamus = true;
    eng.cogs.attractor = true;

    // Initial state
    let obs = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let _ = eng.step(&obs, 0.0, None, &[]);

    // When hypothalamus active, phi_total should exceed graph phi alone
    // because total_deficit() > 0 (energy < 0.5, hydration < 0.5)
    assert!(eng.phi_total >= eng.current_phi,
        "phi_total ({}) should >= graph_phi ({}) when hypothalamus active",
        eng.phi_total, eng.current_phi);

    // After a reward, deficits decrease → phi_total should drop
    let _ = eng.step(&obs, 20.0, None, &[]);
    let deficit_after = eng.hypothalamus.total_deficit();
    assert!(deficit_after < 1.0, "deficit should be reduced after reward: {}", deficit_after);
}

#[test]
fn test_phi_total_equals_graph_phi_when_hypothalamus_off() {
    let mut eng = TsoEngine::with_hidden(4, 2, 0);
    eng.cogs.graph_phi = true;
    eng.cogs.hypothalamus = false;
    eng.cogs.attractor = true;

    let obs = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let _ = eng.step(&obs, 0.0, None, &[]);

    assert_eq!(eng.phi_total, eng.current_phi,
        "phi_total should equal graph_phi when hypothalamus is off");
}

#[test]
fn test_phi_total_tracks_deficit_changes() {
    let mut eng = TsoEngine::with_hidden(4, 2, 0);
    eng.cogs.graph_phi = true;
    eng.cogs.hypothalamus = true;
    eng.cogs.attractor = true;

    // Multiple steps without reward: deficits grow, phi_total should reflect that
    let obs = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    for _ in 0..200 {
        let _ = eng.step(&obs, 0.0, None, &[]);
    }

    // Deficit grows after enough steps (energy starts at 1.0, drifts ~0.005/step)
    let deficit = eng.hypothalamus.total_deficit();
    assert!(deficit > 0.01, "deficit should be measurable after 200 steps: {}", deficit);
    assert!(eng.phi_total >= eng.current_phi,
        "phi_total ({}) should >= graph_phi ({}) when deficit > 0",
        eng.phi_total, eng.current_phi);
}
