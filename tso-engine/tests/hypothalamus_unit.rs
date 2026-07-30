use tso_engine::hypothalamus::Hypothalamus;

#[test]
fn test_new_defaults() {
    let h = Hypothalamus::new();
    assert!((h.energy - 1.0).abs() < 1e-6, "energy={}", h.energy);
    assert!((h.hydration - 1.0).abs() < 1e-6, "hydration={}", h.hydration);
    assert!((h.temperature - 0.5).abs() < 1e-6, "temperature={}", h.temperature);
    assert!((h.phi - 0.0).abs() < 1e-6);
    assert!((h.sleep_debt - 0.0).abs() < 1e-6);
}

#[test]
fn test_step_drift() {
    let mut h = Hypothalamus::new();
    h.step();
    assert!(h.energy < 1.0, "energy={} did not decrease", h.energy);
    assert!(h.hydration < 1.0, "hydration={} did not decrease", h.hydration);
}

#[test]
fn test_step_dt_scales_with_dt() {
    let mut h1 = Hypothalamus::new();
    let mut h2 = Hypothalamus::new();
    h1.step_dt(0.1);
    h2.step_dt(0.2);
    assert!(h2.energy < h1.energy, "larger dt should consume more energy");
}

#[test]
fn test_gate_reward_amplified_when_deficit() {
    let mut h = Hypothalamus::new();
    h.energy = 0.1;
    let gated = h.gate_reward(1.0);
    assert!(gated > 1.0, "gated={} should be > 1.0 when deficit", gated);
}

#[test]
fn test_gate_reward_no_amplification_when_satiated() {
    let h = Hypothalamus::new();
    let gated = h.gate_reward(1.0);
    assert!((gated - 1.0).abs() < 1e-6, "gated={}, expected 1.0", gated);
}

#[test]
fn test_consummatory_value_positive_when_reward() {
    let mut h = Hypothalamus::new();
    h.energy = 0.1;
    let cv = h.consummatory_value(1.0);
    assert!(cv > 0.0, "consummatory={} should be > 0", cv);
}

#[test]
fn test_consummatory_value_zero_when_no_reward() {
    let mut h = Hypothalamus::new();
    h.energy = 0.1;
    let cv = h.consummatory_value(0.0);
    assert!((cv - 0.0).abs() < 1e-6, "consummatory={}, expected 0.0", cv);
}

#[test]
fn test_consume_full_restoration() {
    let mut h = Hypothalamus::new();
    h.energy = 0.1;
    h.hydration = 0.2;
    h.temperature = 0.9;
    h.consume(20.0);
    assert!((h.energy - 1.0).abs() < 1e-6);
    assert!((h.hydration - 1.0).abs() < 1e-6);
    assert!((h.temperature - 0.5).abs() < 1e-6);
}

#[test]
fn test_consume_partial() {
    let mut h = Hypothalamus::new();
    h.energy = 0.1;
    h.consume(1.0);
    assert!(h.energy > 0.1, "energy={} should increase", h.energy);
    assert!(h.energy < 1.0, "energy={} should not fully restore", h.energy);
}

#[test]
fn test_total_deficit_components() {
    let mut h = Hypothalamus::new();
    h.energy = 0.0;
    h.sleep_debt = 0.0;
    assert!((h.total_deficit() - 0.5).abs() < 1e-6);
}

#[test]
fn test_primary_deficit() {
    let mut h = Hypothalamus::new();
    h.energy = 0.0;
    h.hydration = 0.4;
    h.temperature = 0.5;
    let pd = h.primary_deficit();
    assert!((pd - 0.5).abs() < 1e-6, "primary={}, expected 0.5", pd);
}

#[test]
fn test_total_drive_includes_phi_and_sleep() {
    let mut h = Hypothalamus::new();
    h.energy = 0.0;
    h.phi = 0.5;
    h.sleep_debt = 0.2;
    let drive = h.total_drive();
    assert!(drive > 0.5 && drive < 2.0);
}

#[test]
fn test_reset_sleep() {
    let mut h = Hypothalamus::new();
    h.sleep_debt = 1.5;
    h.reset_sleep();
    assert!((h.sleep_debt - 0.0).abs() < 1e-6);
}

#[test]
fn test_apply_metabolic_cost() {
    let mut h = Hypothalamus::new();
    h.apply_metabolic_cost(0.5, 1.0, 0.0);
    assert!((h.cerebellum_cost - 0.5).abs() < 1e-6);
    assert!((h.graph_cost - 1.0).abs() < 1e-6);
    assert!(h.energy < 1.0, "energy={} should decrease after cost", h.energy);
    assert!(h.total_cost > 0.0, "total_cost={} should be positive", h.total_cost);
}

#[test]
fn test_habit_efficiency_reduces_cost() {
    let mut h1 = Hypothalamus::new();
    let mut h2 = Hypothalamus::new();
    h1.apply_metabolic_cost(0.5, 1.0, 0.0);
    h2.apply_metabolic_cost(0.5, 1.0, 1.0);
    assert!(h2.total_cost < h1.total_cost, "habit efficiency should reduce cost");
}

#[test]
fn test_homeostatic_state() {
    let h = Hypothalamus::new();
    let state = h.homeostatic_state();
    assert_eq!(state.len(), 3);
    assert!((state[0] - 1.0).abs() < 1e-6);
    assert!((state[1] - 1.0).abs() < 1e-6);
    assert!((state[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_set_phi() {
    let mut h = Hypothalamus::new();
    h.set_phi(0.75);
    assert!((h.phi - 0.75).abs() < 1e-6);
}

#[test]
fn test_sleep_drive_and_pressure() {
    let mut h = Hypothalamus::new();
    h.sleep_debt = 0.8;
    assert!((h.sleep_drive() - 0.8).abs() < 1e-6);
    assert!((h.sleep_pressure() - 0.8).abs() < 1e-6);
}

#[test]
fn test_sleep_drive_capped_at_one() {
    let mut h = Hypothalamus::new();
    h.sleep_debt = 2.0;
    assert!((h.sleep_drive() - 1.0).abs() < 1e-6);
    assert!((h.sleep_pressure() - 1.0).abs() < 1e-6);
}

#[test]
fn test_energy_never_goes_negative() {
    let mut h = Hypothalamus::new();
    h.energy = 0.01;
    h.apply_metabolic_cost(100.0, 100.0, 0.0);
    assert!(h.energy >= 0.0, "energy={} went negative", h.energy);
}

#[test]
fn test_temperature_stays_clamped() {
    let mut h = Hypothalamus::new();
    for _ in 0..1000 {
        h.step_dt(10.0);
        assert!(h.temperature >= 0.0 && h.temperature <= 1.0,
            "temperature={} out of [0,1]", h.temperature);
    }
}
