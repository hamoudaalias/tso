use ndarray::{Array1, Array2};
use std::collections::HashMap;
use tso_kernel::{
    compute_trifriction, AnchoredTSODecoder, FrictionCalculator, LIFCluster, LIFNeuron,
    LocalWaveCritic, RSTDPPlasticity, TopographicOperator, WaveContext,
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

#[test]
fn test_local_wave_critic() {
    // Graph: 0--1--2  (linear chain)
    // Edges: (0,1) implication, (1,2) implication
    let states = vec![0.8, 0.1, 0.8]; // 0 and 2 are active, 1 is inactive → tension on (0,1) and (1,2)
    let edges = vec![
        (0, 1, 1.0, 1.0),
        (1, 2, 1.0, 1.0),
    ];
    let adjacency = vec![
        vec![1],
        vec![0, 2],
        vec![1],
    ];

    let critic = LocalWaveCritic::new(2, 0.5, 0.3);
    let ctx = WaveContext {
        node_states: &states,
        edges: &edges,
        adjacency: &adjacency,
    };

    // Validate the initial local phi is positive (conflict on (0,1) and (1,2))
    let initial = critic.local_phi(&ctx, &[0, 1], 1);
    assert!(initial > 0.0, "Expected positive friction on conflict, got {initial}");

    // Simulate inverting node 1's activation (0.1 → -0.1)
    // This should NOT help because both edges are implication (need positive dot)
    let bad_action =
        critic.evaluate_action(&ctx, 1, 0, |node| -ctx.node_states[node]);
    assert!(!bad_action, "Inverting node 1 should not reduce friction");

    // Simulate boosting node 1's activation to match (0.1 → 0.8)
    let good_action =
        critic.evaluate_action(&ctx, 1, 0, |_node| 0.8);
    assert!(good_action, "Boosting node 1 should reduce friction");

    // Local phi on neighbourhood of (1) should now be lower
    let post: f64 = {
        let mut patched = states.clone();
        patched[1] = 0.8;
        let pctx = WaveContext {
            node_states: &patched,
            edges: &edges,
            adjacency: &adjacency,
        };
        critic.local_phi(&pctx, &[0, 1], 1)
    };
    assert!(post < initial, "Local phi should decrease after good action");
}

fn make_toy_decoder() -> AnchoredTSODecoder {
    // 5 words, 4D embeddings
    let mut idx_to_word: HashMap<usize, String> = HashMap::new();
    idx_to_word.insert(0, "dog".into());
    idx_to_word.insert(1, "park".into());
    idx_to_word.insert(2, "run".into());
    idx_to_word.insert(3, "ball".into());
    idx_to_word.insert(4, "cat".into());
    let mut word_to_idx = HashMap::new();
    for (k, v) in &idx_to_word {
        word_to_idx.insert(v.clone(), *k);
    }
    // Orthogonal-ish 4D vectors
    let data = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.5, 0.5, 0.0, 0.0,
    ];
    let embeddings = Array2::from_shape_vec((5, 4), data).unwrap();
    AnchoredTSODecoder::new(idx_to_word, word_to_idx, embeddings, 0.9, 0.5)
}

#[test]
fn test_triple_lif_predictive_state() {
    let mut dec = make_toy_decoder();
    // Ingest "dog" → slow and medium should be near dog's embedding
    let dog_vec = dec.embeddings.row(0).to_owned();
    dec.ingest(&[("dog".into(), dog_vec)]);
    let pred = dec.predictive_state();
    // pred should be closer to dog (1,0,0,0) than to cat (0.5,0.5,0,0)
    let dog_cos = pred.dot(&dec.embeddings.row(0));
    let cat_cos = pred.dot(&dec.embeddings.row(4));
    assert!(dog_cos > cat_cos, "predictive state should favor dog over cat");
}

#[test]
fn test_anchor_teleport_on_low_medium_friction() {
    let mut dec = make_toy_decoder();
    dec.anchor_interval = 5;
    dec.anchor_friction_threshold = 0.1;

    let dog_vec = dec.embeddings.row(0).to_owned();
    dec.ingest(&[("dog".into(), dog_vec)]);

    // Emit 5 "dog" tokens — medium state stays near equilibrium → low friction
    for _ in 0..5 {
        dec.slow_state = dec.alpha_slow * &dec.slow_state + (1.0 - dec.alpha_slow) * &dec.embeddings.row(0);
        dec.medium_state = dec.alpha_medium * &dec.medium_state + (1.0 - dec.alpha_medium) * &dec.embeddings.row(0);
        dec.fast_state = dec.alpha_fast * &dec.fast_state + (1.0 - dec.alpha_fast) * &dec.embeddings.row(0);
        dec.normalize_states();
        dec.token_count_since_anchor += 1;
    }

    // Trigger teleport check
    if dec.token_count_since_anchor >= dec.anchor_interval {
        let m_delta = &dec.medium_state - &dec.last_medium_snapshot;
        let medium_friction: f64 = m_delta.mapv(|x| x * x).sum();
        if medium_friction < dec.anchor_friction_threshold {
            dec.anchor_state = dec.medium_state.clone();
            dec.last_medium_snapshot = dec.medium_state.clone();
            dec.token_count_since_anchor = 0;
        }
    }

    // Anchor should have teleported (counter reset, anchor = medium_state)
    assert_eq!(
        dec.token_count_since_anchor, 0,
        "Anchor teleport should reset token counter"
    );
    let dog_cos = dec.anchor_state.dot(&dec.embeddings.row(0));
    assert!(
        dog_cos > 0.9,
        "Anchor should still be near 'dog' after teleport (cos={:.4})",
        dog_cos
    );
}

#[test]
fn test_anchor_no_teleport_on_high_medium_friction() {
    let mut dec = make_toy_decoder();
    dec.anchor_interval = 5;
    dec.anchor_friction_threshold = 0.001; // very low threshold

    let dog_vec = dec.embeddings.row(0).to_owned();
    dec.ingest(&[("dog".into(), dog_vec)]);
    let anchor_before = dec.anchor_state.clone();

    // Emit 5 "cat" tokens — medium state drifts significantly from dog to cat
    for _ in 0..5 {
        dec.slow_state = dec.alpha_slow * &dec.slow_state + (1.0 - dec.alpha_slow) * &dec.embeddings.row(4);
        dec.medium_state = dec.alpha_medium * &dec.medium_state + (1.0 - dec.alpha_medium) * &dec.embeddings.row(4);
        dec.fast_state = dec.alpha_fast * &dec.fast_state + (1.0 - dec.alpha_fast) * &dec.embeddings.row(4);
        dec.normalize_states();
        dec.token_count_since_anchor += 1;
    }

    if dec.token_count_since_anchor >= dec.anchor_interval {
        let m_delta = &dec.medium_state - &dec.last_medium_snapshot;
        let medium_friction: f64 = m_delta.mapv(|x| x * x).sum();
        if medium_friction < dec.anchor_friction_threshold {
            dec.anchor_state = dec.medium_state.clone();
        }
    }

    // Anchor should NOT have teleported because friction was too high
    let dog_cos_before = anchor_before.dot(&dec.embeddings.row(0));
    let dog_cos_after = dec.anchor_state.dot(&dec.embeddings.row(0));
    assert!(
        (dog_cos_before - dog_cos_after).abs() < 1e-6,
        "Anchor should be unchanged when medium friction exceeds threshold"
    );
}
