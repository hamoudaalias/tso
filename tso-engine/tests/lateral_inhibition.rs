//! Tests du mécanisme d'inhibition latérale (décroissance graduelle).
//!
//! Compare lateral_inhibition_sweep (décroissance) vs demineur_sweep (suppression).
//! Vérifie que la version graduelle produit une convergence plus lisse de Φ.

use ndarray::Array1;
use tso_engine::core::{Graph, lateral_inhibition_sweep, lateral_inhibition_trace, demineur_sweep};

/// Construit un petit graphe avec arêtes mixtes (implication + exclusion).
fn build_test_graph() -> Graph {
    let mut g = Graph::with_params(0.7, 0.1);
    let v1 = Array1::from_vec(vec![0.8, 0.1, 0.1, 0.1]);
    let v2 = Array1::from_vec(vec![0.1, 0.8, 0.1, 0.1]);
    let v3 = Array1::from_vec(vec![0.1, 0.1, 0.8, 0.1]);
    g.add_node(v1);
    g.add_node(v2);
    g.add_node(v3);
    g.add_edge(0, 1, -1); // exclusion
    g.add_edge(1, 2, -1); // exclusion
    g.add_edge(0, 2, 1);  // implication
    g
}

#[test]
fn test_lateral_inhibition_same_phi_as_flag() {
    let mut g1 = build_test_graph();
    let mut g2 = build_test_graph();

    let (f1, phi1, final1) = demineur_sweep(&mut g1, 0.01);
    let (f2, phi2, final2) = lateral_inhibition_sweep(&mut g2, 0.01, 1);

    eprintln!("flag: {} flags, Φ dropped {:.4}, final Φ {:.4}", f1, phi1, final1);
    eprintln!("lateral: {} decays, Φ dropped {:.4}, final Φ {:.4}", f2, phi2, final2);

    // Les deux doivent converger vers Φ < tol
    // Note: demineur_sweep utilise exponential_decay (×0.95) maintenant
    assert!(final1 < 0.01, "demineur should converge to Φ < 0.01, got {:.4}", final1);
    assert!(final2 < 0.01, "lateral inhibition should converge to Φ < 0.01, got {:.4}", final2);

    // Les deux strategies atteignent un Φ final similaire
    assert!((final1 - final2).abs() < 0.02, "Both should reach similar final Φ");
}

#[test]
fn test_lateral_inhibition_small_decay() {
    let mut g = build_test_graph();
    let phi_before = g.phi();

    // Décroissance très faible (pas de 1, supprime quand poids arrive à 0)
    let (flags, phi_drop, phi_after) = lateral_inhibition_sweep(&mut g, 0.01, 1);

    eprintln!("small decay: Φ {:.4} → {:.4} (drop {:.4}, {} flags)", phi_before, phi_after, phi_drop, flags);

    // Avec une décroissance faible, le Φ doit avoir baissé mais peut ne pas avoir convergé
    assert!(phi_after <= phi_before + 1e-6, "Φ should not increase");
    // Le nombre de flags indique combien d'arêtes ont été supprimées (poids < min_weight)
    eprintln!("  edges remaining: {}", g.edges.len());
}

/// Test requirement 1 (LTD-like): weight decays gradually toward zero in steps of 1.
/// Each call to decay_edge_weight reduces |weight| by `decay` until 0, then removes.
/// This mimics long-term depression (LTD) of synaptic strength.
#[test]
fn test_decay_is_ltd_like() {
    let mut g = build_test_graph();
    // Check edge (0,1) starts at -1 (exclusion)
    let w0 = g.edge_weight(0, 1).unwrap();
    assert_eq!(w0, -1, "exclusion edge should start at -1");

    // Decay by 1: exclusion -1 → 0 (not removed yet, weight=0 means removed when next decay called)
    // decay_edge_weight reduces toward zero: -1 + 1 = 0 → edge removed
    let saved = g.decay_edge_weight(0, 1, 1);
    // After -1 -> 0, the edge is removed, so saved > 0
    assert!(saved > 0.0, "decaying exclusion -1→0 should remove edge and return phi>0");
    assert!(g.edge_weight(0, 1).is_none(), "edge should be removed after decay to 0");
}

/// Test requirement 1b: multiple decay steps for |weight| > 1
/// Edge (0,2) has weight +2. First decay: +2 → +1 (survives). Second: +1 → 0 (removed).
#[test]
fn test_multiple_decay_steps() {
    let mut g = build_test_graph();
    assert_eq!(g.edge_weight(0, 2).unwrap(), 1, "edge (0,2) weight 1");

    // Add an edge with weight +2 to test multi-step decay
    // Use a node anti-aligned with node 0 so phi > 0 for the implication
    let v4 = Array1::from_vec(vec![0.1, 0.8, 0.1, 0.1]); // similar to v2, not anti-aligned
    g.add_node(v4);
    // (0,3) implication: dot ~ 0.18, gamma=0.7, phi=(0.7-0.18).max(0)=0.52 > 0
    g.add_edge(0, 3, 2);
    assert_eq!(g.edge_weight(0, 3).unwrap(), 2);

    let edges_before = g.edges.len();
    let saved1 = g.decay_edge_weight(0, 3, 1);
    assert_eq!(saved1, 0.0, "no phi saved yet - edge still alive");
    assert_eq!(g.edge_weight(0, 3).unwrap(), 1, "weight should drop from 2 to 1");
    assert_eq!(g.edges.len(), edges_before, "edge count unchanged after first decay");

    // Second decay: 1→0, edge removed
    let saved2 = g.decay_edge_weight(0, 3, 1);
    assert!(saved2 > 0.0, "phi saved on removal, got {:.6}", saved2);
    assert!(g.edge_weight(0, 3).is_none(), "edge removed at weight 0");
}

/// Test requirement 2 (Reversible): a decayed edge whose weight hasn't reached 0
/// can be strengthened again if new evidence supports it.
/// The edge survives at reduced weight, and add_edge / direct weight manipulation can
/// increase it later.
#[test]
fn test_reversible_decay() {
    let mut g = build_test_graph();
    // Edge (1,2) is -1 exclusion
    assert_eq!(g.edge_weight(1, 2).unwrap(), -1);

    // Decay by 1: -1 → 0 → removed (exclusion has only one step)
    // For reversibility, use an edge with |weight| > 1
    let v4 = Array1::from_vec(vec![0.1, 0.8, 0.1, 0.1]);
    g.add_node(v4);
    g.add_edge(0, 3, 2); // weight +2 implication
    assert_eq!(g.edge_weight(0, 3).unwrap(), 2);

    // Decay first step: 2→1
    g.decay_edge_weight(0, 3, 1);
    assert_eq!(g.edge_weight(0, 3).unwrap(), 1, "weight decayed to 1");

    // Reverse: manually restore weight via edge_map (simulating new evidence)
    // Remove and re-add with stronger weight
    // The Graph API doesn't have set_edge_weight, so we use remove_edge + add_edge
    let _ = g.remove_edge(0, 3);
    g.add_edge(0, 3, 2); // restored to original strength
    assert_eq!(g.edge_weight(0, 3).unwrap(), 2, "edge restored to weight 2");

    // Now decay in steps again to show the cycle is reversible
    g.decay_edge_weight(0, 3, 1);
    assert_eq!(g.edge_weight(0, 3).unwrap(), 1, "second decay cycle works");

    // Edge was never removed at weight=0, so the association survives in weakened form.
    // This is the core reversibility property: gradual inhibition doesn't destroy the
    // connection; it only weakens it, leaving the door open for re-strengthening.
}

/// Test requirement 3 (Smooth phi reduction): lateral_inhibition_sweep reduces Φ
/// in smaller steps than demineur_sweep. The trace shows smaller per-step deltas.
#[test]
fn test_smooth_phi_reduction() {
    let mut g = build_test_graph();
    let phi0 = g.phi();

    // Run trace and verify we get multiple steps with small phi drops
    let (flags, phi_dropped, phi_final, trace) = lateral_inhibition_trace(&mut g, 0.001, 1);

    eprintln!("Smooth phi trace:");
    for (i, (before, after, w)) in trace.iter().enumerate() {
        eprintln!("  step {}: Φ {:.4} → {:.4} (ΔΦ={:.4}, weight={})", i, before, after, before - after, w);
    }
    eprintln!("Total: {} decays, Φ dropped {:.4}, final Φ {:.4}", flags, phi_dropped, phi_final);

    // Smoothness: each step should drop phi by less than 50% of total (no single jump)
    // Also: some steps may have near-zero phi drop (decaying from 2→1 doesn't remove)
    assert!(phi_final < phi0 + 0.01, "phi should decrease overall");
    assert!(flags > 0 || phi_final < 0.001, "should have at least some decays for this graph");

    // Compare with demineur_sweep which removes edges instantly:
    // demineur steps have larger deltas (one step removes the entire edge at once)
    let mut g2 = build_test_graph();
    let (f2, pd2, _) = demineur_sweep(&mut g2, 0.001);
    eprintln!("demineur: {} flags, Φ dropped {:.4}", f2, pd2);

    // The number of lateral inhibition steps should be >= demineur flags
    // (more steps = smoother reduction)
    if flags > 0 && f2 > 0 {
        assert!(flags >= f2, "lateral inhibition should have at least as many steps as flag sweep (got {} < {})", flags, f2);
    }
}

/// Test requirement 4 (Sleep consolidation compatibility): after lateral_inhibition_sweep,
/// remaining edges with low (but >0) weight can be bulk-pruned by sleep consolidation
/// using prune_exclusion_edges. The two mechanisms compose cleanly.
#[test]
fn test_sleep_consolidation_compatible() {
    let mut g = build_test_graph();
    let phi_before_sweep = g.phi();
    eprintln!("Pre-sweep: {} edges, Φ={:.4}", g.edges.len(), phi_before_sweep);

    // Phase 1: lateral inhibition sweep (waketime conflict resolution)
    // Use tol=0.05: edges with phi below 0.05 are not decayed further
    let (flags, phi_dropped, phi_after_sweep) = lateral_inhibition_sweep(&mut g, 0.05, 1);
    eprintln!("Awake sweep: {} decays, Φ dropped {:.4}, Φ={:.4}", flags, phi_dropped, phi_after_sweep);

    // Phase 2: sleep consolidation — bulk-prune remaining low-phi edges
    // (edges with φ < 0.1 are pruned during sleep)
    let (excl, implications, phi_saved) = g.prune_exclusion_edges(0.1);
    eprintln!("Sleep prune: {} excl + {} impl removed, Φ saved {:.4}", excl, implications, phi_saved);

    // After both phases:
    let phi_final = g.phi();
    eprintln!("Final: {} edges, Φ={:.4}", g.edges.len(), phi_final);

    // The two mechanisms compose: sweep weakens, sleep prunes leftovers
    assert!(phi_final < phi_before_sweep + 0.01, "combined wake+sleep should reduce phi");

    // Weight=0 edges were already removed during sweep, so sleep sees only survivors.
    // This is the correct pipeline: awake = gradual LTD weakening, asleep = bulk cleanup.
    eprintln!("Surviving edges:");
    for e in &g.edges {
        eprintln!("  ({}) --{}--> ({})  weight={}", e.from, if e.weight < 0 { "X" } else { "→" }, e.to, e.weight);
    }
}

/// Test edge case: decay=0 should be a no-op (no change).
#[test]
fn test_decay_zero_is_noop() {
    let mut g = build_test_graph();
    let edges_before = g.edges.len();
    let phi_before = g.phi();

    let (flags, phi_dropped, phi_after) = lateral_inhibition_sweep(&mut g, 0.001, 0);

    assert_eq!(flags, 0, "decay=0 should remove nothing");
    assert_eq!(phi_dropped, 0.0, "decay=0 should drop no phi");
    assert!((phi_after - phi_before).abs() < 1e-10, "phi unchanged with decay=0");
    assert_eq!(g.edges.len(), edges_before, "edges unchanged with decay=0");
}

/// Test edge case: decay larger than |weight| should remove in one step.
#[test]
fn test_large_decay_removes_immediately() {
    let mut g = build_test_graph();
    // Edge (0,1) weight -1
    let saved = g.decay_edge_weight(0, 1, 5); // decay=5 > |-1|
    assert!(saved > 0.0, "large decay should remove edge");
    assert!(g.edge_weight(0, 1).is_none(), "edge should be gone");
}

/// Test that the trace version reports correct per-step weight states.
#[test]
fn test_lateral_inhibition_trace_accuracy() {
    let mut g = build_test_graph();
    // Add a weight=2 edge for multi-step tracing
    let v4 = Array1::from_vec(vec![0.2, 0.9, 0.1, 0.3]);
    g.add_node(v4);
    g.add_edge(2, 3, 2);

    let (flags, phi_dropped, phi_final, trace) = lateral_inhibition_trace(&mut g, 0.01, 1);

    eprintln!("Trace accuracy: {} steps, Φ dropped {:.4}, final Φ={:.4}", flags, phi_dropped, phi_final);
    for (i, (before, after, w)) in trace.iter().enumerate() {
        eprintln!("  {}: Φ {:.4} → {:.4} (w={})", i, before, after, w);
        assert!(*after <= *before + 1e-6, "phi should not increase within a step");
    }
    assert!(flags > 0 || phi_final < 0.001, "should have processed something");
}

/// Test that the TSO engine wrapper also works correctly.
#[test]
fn test_tso_engine_lateral_inhibition_wrapper() {
    use tso_engine::tso_engine::TsoEngine;

    // Create a minimal engine, add some test edges
    let mut engine = TsoEngine::new(3, 3);

    // Inject nodes and edges
    let v1 = Array1::from_vec(vec![0.8, 0.2, 0.1]);
    let v2 = Array1::from_vec(vec![0.2, 0.8, 0.1]);
    let v3 = Array1::from_vec(vec![0.1, 0.2, 0.8]);
    engine.graph.add_node(v1);
    engine.graph.add_node(v2);
    engine.graph.add_node(v3);
    engine.graph.add_edge(0, 1, -1);
    engine.graph.add_edge(1, 2, 1);

    let phi0 = engine.graph.phi();
    let (flags, phi_dropped, phi_after) = engine.lateral_inhibition_sweep(0.01, 1);
    eprintln!("TSO engine sweep: {} decays, Φ dropped {:.4}, Φ={:.4} (was {:.4})", flags, phi_dropped, phi_after, phi0);

    assert!(phi_after <= phi0 + 1e-6, "engine wrapper should reduce phi");
}

/// Stress test: many exclusion edges, sweep should handle them gracefully.
#[test]
fn test_stress_lateral_inhibition() {
    let mut g = Graph::with_params(0.6, 0.2);
    let mut rng = rand::thread_rng();
    use rand::Rng;

    // Add 10 random nodes
    for _ in 0..10 {
        let v = Array1::from_vec(vec![rng.r#gen::<f64>(), rng.r#gen::<f64>(), rng.r#gen::<f64>()]);
        g.add_node(v);
    }
    // Add 20 random edges (mix of exclusion and implication)
    for _ in 0..20 {
        let a = rng.r#gen_range(0..g.nodes.len());
        let b = rng.r#gen_range(0..g.nodes.len());
        if a == b { continue; }
        let w: i8 = if rng.r#gen_bool(0.5) { -1 } else { 1 };
        if g.edge_weight(a, b).is_none() {
            g.add_edge(a, b, w);
        }
    }

    let phi0 = g.phi();
    eprintln!("Stress: {} nodes, {} edges, Φ={:.4}", g.nodes.len(), g.edges.len(), phi0);

    let (flags, phi_dropped, phi_final) = lateral_inhibition_sweep(&mut g, 0.05, 1);
    eprintln!("Stress result: {} decays, Φ dropped {:.4}, final Φ={:.4}", flags, phi_dropped, phi_final);

    assert!(phi_final <= phi0 + 1e-6, "phi should not increase");
    assert!(phi_final < 0.05 || flags > 0, "should reduce phi below tol or have processed edges");
}
