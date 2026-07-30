use ndarray::Array1;
use tso_engine::neurons::{DualLIFState, LIFState};

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

#[test]
fn test_lif_new_has_zeros() {
    let l = LIFState::new(4, 0.9);
    assert_eq!(l.state.len(), 4);
    assert!(l.state.iter().all(|x| *x == 0.0));
    assert!(e(l.alpha, 0.9));
}

#[test]
fn test_lif_step_updates_state() {
    let mut l = LIFState::new(3, 0.5);
    l.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), false);
    assert!(!e(l.state[0], 0.0));
    assert!(e(l.state[1], 0.0));
}

#[test]
fn test_lif_step_negate() {
    let mut l = LIFState::new(3, 0.5);
    l.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), true);
    assert!(l.state[0] < 0.0);
}

#[test]
fn test_lif_reset() {
    let mut l = LIFState::new(3, 0.5);
    l.step(&Array1::from_vec(vec![1.0, 1.0, 1.0]), false);
    l.reset();
    assert!(l.state.iter().all(|x| *x == 0.0));
}

#[test]
fn test_lif_converges_asymptotically() {
    let mut l = LIFState::new(1, 0.9);
    let input = Array1::from_vec(vec![1.0]);
    for _ in 0..100 {
        l.step(&input, false);
    }
    let err = (1.0 - l.state[0]).abs();
    assert!(err < 1e-4, "error={} should be tiny after 100 steps", err);
}

#[test]
fn test_dual_lif_new() {
    let d = DualLIFState::new(3, 0.95, 0.5);
    assert_eq!(d.slow.state.len(), 3);
    assert_eq!(d.fast.state.len(), 3);
    assert!(e(d.slow.alpha, 0.95));
    assert!(e(d.fast.alpha, 0.5));
}

#[test]
fn test_dual_lif_step_updates_both() {
    let mut d = DualLIFState::new(3, 0.9, 0.5);
    d.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), false);
    assert!(d.slow.state[0] > 0.0);
    assert!(d.fast.state[0] > 0.0);
}

#[test]
fn test_dual_lif_alignment_fast_converges_quicker() {
    let mut d = DualLIFState::new(3, 0.95, 0.5);
    let emb = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    for _ in 0..10 {
        d.step(&emb, false);
    }
    let align_fast = d.alignment(&emb, 0.0);
    let align_slow = d.alignment(&emb, 1.0);
    assert!(align_fast > align_slow);
}

#[test]
fn test_dual_lif_alignment_zero_embedding() {
    let mut d = DualLIFState::new(3, 0.9, 0.5);
    d.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), false);
    let align = d.alignment(&Array1::zeros(3), 0.5);
    assert!(e(align, 0.0));
}

#[test]
fn test_dual_lif_reset() {
    let mut d = DualLIFState::new(3, 0.9, 0.5);
    d.step(&Array1::from_vec(vec![1.0, 1.0, 1.0]), false);
    d.reset();
    assert!(d.slow.state.iter().all(|x| *x == 0.0));
    assert!(d.fast.state.iter().all(|x| *x == 0.0));
}

#[test]
fn test_alignment_after_reset() {
    let mut d = DualLIFState::new(3, 0.9, 0.5);
    d.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), false);
    d.reset();
    let align = d.alignment(&Array1::from_vec(vec![1.0, 0.0, 0.0]), 0.5);
    assert!(e(align, 0.0));
}
