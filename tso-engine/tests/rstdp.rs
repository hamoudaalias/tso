use tso_engine::{TsoEngine, plasticity::RstdpPlasticity};

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
    // init via new() — with_hidden doesn't call init_rstdp
    // Force rstdp init manually
    let r = RstdpPlasticity::new(10, 0, 4, 0.01);
    tso.cerebellum.rstdp = Some(r);
    let obs = ndarray::Array1::zeros(10);
    let action = tso.step(&obs, 1.0, None, &[]);
    assert!(action < 4);
}

#[test]
fn test_plasticity_reset() {
    let mut p = RstdpPlasticity::new(5, 0, 3, 0.01);
    p.update_trace(&[1.0; 5], &[1.0; 3], &[]);
    p.reset();
    for row in p.e_lin.iter() {
        for v in row.iter() {
            assert!((*v).abs() < 1e-10);
        }
    }
}
