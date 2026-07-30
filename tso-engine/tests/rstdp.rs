#![cfg(feature = "rstdp")]

use tso_engine::{TsoEngine, plasticity::RstdpPlasticity};
use ndarray::Array1;

#[test]
fn test_rstdp_disabled_default() {
    let tso = TsoEngine::new(10, 4);
    assert!(tso.cerebellum.rstdp.is_none());
}

#[test]
fn test_step_rstdp_enabled() {
    let mut cfg = tso_engine::CognitiveConfig::default();
    cfg.rstdp_enabled = true;
    let mut tso = TsoEngine::with_hidden(10, 4, 0);
    let r = RstdpPlasticity::new(10, 0, 4, 0.01);
    tso.cerebellum.rstdp = Some(r);
    let obs = ndarray::Array1::zeros(10);
    let action = tso.step(&obs, 1.0, None, &[]);
    assert!(action < 4);
}

#[test]
fn test_plasticity_reset() {
    let mut p = RstdpPlasticity::new(5, 0, 3, 0.01);
    p.update_trace(&Array1::from_vec(vec![1.0; 5]), &Array1::from_vec(vec![1.0; 3]), &Array1::zeros(0));
    p.reset();
    for v in p.e_lin.iter() {
        assert!((*v).abs() < 1e-10);
    }
}

#[test]
fn test_rstdp_trace_updated_in_step() {
    // Verify that calling step() updates the R-STDP eligibility traces
    let mut tso = TsoEngine::with_hidden(10, 4, 0);
    let r = RstdpPlasticity::new(10, 0, 4, 0.01);
    tso.cerebellum.rstdp = Some(r);
    // Use non-zero input so the cerebellum produces non-zero logits
    let obs = Array1::from_vec(vec![0.5; 10]);
    let _ = tso.step(&obs, 1.0, None, &[]);
    let rstdp = tso.cerebellum.rstdp.as_ref().unwrap();
    let trace_sum: f64 = rstdp.e_lin.iter().sum();
    assert!(trace_sum > 0.0, "R-STDP traces should be non-zero after step (sum={})", trace_sum);
}
