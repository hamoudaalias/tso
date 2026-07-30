use ndarray::Array1;
use tso_engine::perceptual_belt::PerceptualBelt;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

fn p4(a: f64, b: f64, c: f64, d: f64) -> Array1<f64> {
    Array1::from_vec(vec![a, b, c, d])
}

#[test]
fn test_new_creates_belt() {
    let pb = PerceptualBelt::new(4);
    assert_eq!(pb.dim(), 4);
    assert_eq!(pb.num_concepts(), 8);
    assert!(pb.episode_trace().is_empty());
}

#[test]
fn test_new_with_zero_dim() {
    let pb = PerceptualBelt::new(0);
    assert_eq!(pb.dim(), 0);
}

#[test]
fn test_process_fast_path_no_backend() {
    let mut pb = PerceptualBelt::new(4);
    let percept = pb.process(&p4(0.5, 0.3, 0.8, 0.1), None, &[], false, false, false, false);
    assert_eq!(percept.concept_id, 0);
    assert!(!percept.is_new);
    assert!(e(percept.intrinsic, 0.0));
    assert!(e(percept.shaping, 0.0));
    assert_eq!(percept.gated.len(), 4);
}

#[test]
fn test_process_fast_path_passthrough() {
    let mut pb = PerceptualBelt::new(4);
    let input = p4(0.5, 0.3, 0.8, 0.1);
    let percept = pb.process(&input, None, &[], false, false, false, false);
    for i in 0..4 {
        assert!(e(percept.gated[i], input[i]));
    }
}

#[test]
fn test_process_attractor_creates_new_concept_for_distant_input() {
    let mut pb = PerceptualBelt::new(4);
    let input = p4(1.0, 1.0, 1.0, 1.0);
    let percept = pb.process(&input, None, &[], false, true, false, false);
    assert_eq!(percept.concept_id, 8);
    assert!(percept.is_new);
    assert_eq!(pb.num_concepts(), 9);
}

#[test]
fn test_process_attractor_reuses_existing_concept() {
    let mut pb = PerceptualBelt::new(4);
    let near = p4(0.5, 0.5, 0.5, 0.5);
    let near2 = p4(0.51, 0.5, 0.49, 0.5);
    pb.process(&near, None, &[], false, true, false, false);
    let p2 = pb.process(&near2, None, &[], false, true, false, false);
    assert_eq!(p2.concept_id, 8);
}

#[test]
fn test_process_with_curiosity_gives_intrinsic() {
    let mut pb = PerceptualBelt::new(4);
    let input = p4(0.9, 0.9, 0.9, 0.9);
    let percept = pb.process(&input, None, &[], false, true, true, false);
    assert!(percept.intrinsic >= 0.0);
}

#[test]
fn test_process_multiple_calls_grow_episode_trace() {
    let mut pb = PerceptualBelt::new(4);
    pb.process(&p4(0.1, 0.2, 0.3, 0.4), None, &[], false, true, false, false);
    pb.process(&p4(0.5, 0.6, 0.7, 0.8), None, &[], false, true, false, false);
    assert_eq!(pb.episode_trace().len(), 2);
}

#[test]
fn test_process_with_bfs_value_sets_concept_value() {
    let mut pb = PerceptualBelt::new(4);
    let input = p4(0.9, 0.9, 0.9, 0.9);
    pb.process(&input, Some(0.7), &[], false, true, false, false);
    assert!(!pb.concept_values().is_empty());
    assert!(e(pb.concept_values()[8], 0.7));
}

#[test]
fn test_reset_clears_state() {
    let mut pb = PerceptualBelt::new(4);
    pb.process(&p4(0.1, 0.2, 0.3, 0.4), None, &[], false, true, false, false);
    pb.reset();
    assert!(pb.episode_trace().is_empty());
}

#[test]
fn test_recall_on_empty_returns_none() {
    let pb = PerceptualBelt::new(4);
    let result = pb.recall(&p4(0.5, 0.5, 0.5, 0.5));
    assert!(result.is_none());
}

#[test]
fn test_lif_state_accessor() {
    let pb = PerceptualBelt::new(4);
    let lif = pb.lif_state();
    assert_eq!(lif.slow.state.len(), 4);
    assert_eq!(lif.fast.state.len(), 4);
}

#[test]
fn test_configure_updates_grid_cells() {
    let mut pb = PerceptualBelt::new(4);
    pb.configure(10, 10);
    assert!(pb.extra_dim() > 0);
}

#[test]
fn test_get_prototype_returns_some_for_existing_class() {
    let pb = PerceptualBelt::new(4);
    let proto = pb.get_prototype(0);
    assert!(proto.is_some());
    assert_eq!(proto.unwrap().len(), 4);
}

#[test]
fn test_get_prototype_returns_none_for_missing() {
    let pb = PerceptualBelt::new(4);
    assert!(pb.get_prototype(99).is_none());
}

#[test]
fn test_predicted_concept_id() {
    let mut pb = PerceptualBelt::new(4);
    assert!(pb.predicted_concept_id().is_none());
    pb.set_predicted_concept_id(Some(3));
    assert_eq!(pb.predicted_concept_id(), Some(3));
}

#[test]
fn test_concept_maturation_accessors() {
    let mut pb = PerceptualBelt::new(4);
    assert!(pb.concept_maturation().is_empty());
    let input = p4(0.9, 0.9, 0.9, 0.9);
    pb.process(&input, None, &[], false, true, false, false);
    assert!(!pb.concept_maturation().is_empty());
    pb.concept_maturation_mut()[8] = 5;
    assert_eq!(pb.concept_maturation()[8], 5);
}

#[test]
fn test_last_active_step_mut() {
    let mut pb = PerceptualBelt::new(4);
    pb.last_active_step_mut();
}

#[test]
fn test_attention_gating() {
    let mut pb = PerceptualBelt::new(4);
    let input = p4(0.5, 0.3, 0.8, 0.1);
    let percept = pb.process(&input, None, &[], false, true, false, true);
    assert_eq!(percept.gated.len(), 4);
}

#[test]
fn test_process_does_not_panic_with_bias() {
    let mut pb = PerceptualBelt::new(4);
    let percept = pb.process(&p4(0.5, 0.5, 0.5, 0.5), Some(0.5), &[0.0, 0.0, 0.0, 0.0], false, true, false, false);
    assert!(percept.concept_id < 20);
}
