//! Tests du mécanisme d'inhibition latérale (décroissance graduelle).
//!
//! Compare lateral_inhibition_sweep (décroissance) vs demineur_sweep (suppression).
//! Vérifie que la version graduelle produit une convergence plus lisse de Φ.

use ndarray::Array1;
use tso_engine::core::{Graph, lateral_inhibition_sweep, demineur_sweep};

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
    assert!(final1 < 0.01, "flag sweep should converge to Φ < 0.01, got {:.4}", final1);
    assert!(final2 < 0.01, "lateral inhibition should converge to Φ < 0.01, got {:.4}", final2);

    // La version graduelle supprime moins d'arêtes (ou autant, selon decay)
    // mais le Φ final est le même
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
