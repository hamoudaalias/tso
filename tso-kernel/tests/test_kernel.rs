use ndarray::Array1;
use std::collections::HashMap;
use tso_kernel::{
    compute_trifriction, FrictionCalculator, LIFCluster, LIFNeuron, RSTDPPlasticity,
    TopographicOperator,
};

#[test]
fn test_friction_zero() {
    let calc = FrictionCalculator::new(0.5, 0.3);
    let rates = Array1::from_vec(vec![0.8, 0.8]);
    let edges = vec![(0, 1, 1.0, 1.0)];
    let phi = calc.compute_phi(&rates, &edges);
    assert!((phi - 0.0).abs() < 1e-6, "Expected 0, got {phi}");
}

#[test]
fn test_friction_positive() {
    let calc = FrictionCalculator::new(0.5, 0.3);
    let rates = Array1::from_vec(vec![0.1, 0.1]);
    let edges = vec![(0, 1, 1.0, 1.0)];
    let phi = calc.compute_phi(&rates, &edges);
    assert!(phi > 0.0);
    assert!((phi - 0.49).abs() < 1e-6, "Expected 0.49, got {phi}");
}

#[test]
fn test_friction_exclusion() {
    let calc = FrictionCalculator::new(0.5, 0.3);
    let rates = Array1::from_vec(vec![0.7, 0.7]);
    let edges = vec![(0, 1, -1.0, 1.0)];
    let phi = calc.compute_phi(&rates, &edges);
    assert!(phi > 0.0);
    assert!((phi - 0.19).abs() < 1e-6, "Expected 0.19, got {phi}"); // 0.7*0.7 - 0.3 = 0.19
}

#[test]
fn test_double_mapping_orthogonality() {
    let d = 5;
    let mut vectors = HashMap::new();
    vectors.insert("a".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]));
    vectors.insert("b".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]));
    vectors.insert("ctx".to_string(), Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]));

    let concepts = vec!["a".to_string(), "b".to_string()];
    let (new_vecs, _) =
        TopographicOperator::double_mapping(&vectors, &concepts, "ctx", d);

    let dot_ab = new_vecs["a"].dot(&new_vecs["b"]).abs();
    assert!(dot_ab < 1e-6, "Expected 0, got {dot_ab}");
}

#[test]
fn test_double_mapping_preserves_context() {
    let d = 5;
    let mut vectors = HashMap::new();
    vectors.insert("a".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]));
    vectors.insert("b".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]));
    vectors.insert("ctx".to_string(), Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]));

    let orig_a_ctx = vectors["a"].dot(&vectors["ctx"]);
    let orig_b_ctx = vectors["b"].dot(&vectors["ctx"]);

    let concepts = vec!["a".to_string(), "b".to_string()];
    let (new_vecs, _) =
        TopographicOperator::double_mapping(&vectors, &concepts, "ctx", d);

    let new_a_ctx = new_vecs["a"].dot(&new_vecs["ctx"]);
    let new_b_ctx = new_vecs["b"].dot(&new_vecs["ctx"]);

    assert!((new_a_ctx - orig_a_ctx).abs() < 1e-6, "Context dot changed");
    assert!((new_b_ctx - orig_b_ctx).abs() < 1e-6, "Context dot changed");
}

#[test]
fn test_lif_step() {
    let mut cluster = LIFCluster::new(5);
    let _ = cluster.step(&Array1::from_elem(5, 20.0), 0.5);
    assert!(cluster.rate >= 0.0);
    assert!(cluster.rate <= 1.0);
}

#[test]
fn test_lif_neuron_fires() {
    let mut n = LIFNeuron::new();
    let mut fired = false;
    for _ in 0..20 {
        if n.step(30.0, 0.5) > 0.0 {
            fired = true;
            break;
        }
    }
    assert!(fired, "Neuron should fire with strong input");
}

#[test]
fn test_gated_plasticity_blocks_exclusives() {
    let mut p = RSTDPPlasticity::new(3, 0.1, 0.02, 0.05);
    let z_chat = Array1::from_vec(vec![1.0, 0.0]);
    let z_chien = Array1::from_vec(vec![-1.0, 0.0]);
    let z_animal = Array1::from_vec(vec![0.0, 1.0]);
    p.register_target(0, z_chat);
    p.register_target(1, z_chien);
    p.register_target(2, z_animal);

    for _ in 0..10 {
        p.consolidate(0, 2);
        p.consolidate(1, 2);
        p.consolidate(0, 1);
    }

    let w_cd = p.w[[0, 1]];
    assert!(w_cd < 0.1, "Gate failed: W(C→D)={:.4}", w_cd);
}

#[test]
fn test_gated_plasticity_allows_implications() {
    let mut p = RSTDPPlasticity::new(3, 0.1, 0.02, 0.05);
    let z_chat = Array1::from_vec(vec![1.0, 0.0]);
    let z_animal = Array1::from_vec(vec![0.0, 1.0]);
    p.register_target(0, z_chat);
    p.register_target(2, z_animal);

    for _ in 0..5 {
        p.consolidate(0, 2);
    }

    let w_ca = p.w[[0, 2]];
    assert!(w_ca > 0.1, "Gate incorrectly blocked: W(C→A)={:.4}", w_ca);
}

#[test]
fn test_ungated_plasticity_via_no_targets() {
    let mut p = RSTDPPlasticity::new(3, 0.1, 0.02, 0.05);
    for _ in 0..8 {
        p.consolidate(0, 2);
        p.consolidate(1, 2);
        p.consolidate(0, 1);
    }

    let w_cd = p.w[[0, 1]];
    assert!(w_cd > 0.2, "No gate should be active, W(C→D)={:.4}", w_cd);
}

#[test]
fn test_conceptual_phi() {
    let calc = FrictionCalculator::default();
    // Build a flat transition matrix: 3 concepts, row-major
    let n_concepts = 3;
    let n_cols = 3;
    let mut w = Array1::from_elem(n_concepts * n_cols, 1.0);
    w[0 * n_cols + 1] = 5.0; // concept 0 -> concept 1 has weight 5
    w[0 * n_cols + 2] = 1.0;

    let phi = calc.conceptual_phi(0, 1, &w, n_concepts, n_cols);
    // p = 5/(5+1+1) = 5/7 ≈ 0.714
    // phi = 1 - 0.714*3 = 1 - 2.143 = -1.143
    let expected = 1.0 - (5.0 / 7.0) * 3.0;
    assert!((phi - expected).abs() < 1e-6, "Expected {expected}, got {phi}");
}

#[test]
fn test_trifriction_basic() {
    let mut graph = HashMap::new();
    graph.insert("cat".to_string(),
        [("animal".into(), 5.0), ("pet".into(), 3.0)].into());
    graph.insert("animal".to_string(),
        [("cat".into(), 5.0), ("pet".into(), 4.0), ("dog".into(), 2.0)].into());
    graph.insert("dog".to_string(),
        [("animal".into(), 4.0), ("pet".into(), 2.0)].into());
    graph.insert("pet".to_string(),
        [("cat".into(), 3.0), ("animal".into(), 4.0), ("dog".into(), 2.0)].into());

    let trif = compute_trifriction("cat", "animal", &graph, 20);
    assert!(trif[0] > 0.0, "support should be > 0, got {}", trif[0]);
    assert!(trif[0] <= 1.0);
    assert!(trif[1] >= 0.0);
    assert!(trif[2] >= 0.0);
}
