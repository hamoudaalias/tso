use ndarray::Array1;
use tso_engine::action::ActionMotor;
use tso_engine::neurons::DualLIFState;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

fn make_context() -> DualLIFState {
    let mut d = DualLIFState::new(3, 0.9, 0.5);
    d.step(&Array1::from_vec(vec![1.0, 0.0, 0.0]), false);
    d
}

fn actions() -> Vec<Array1<f64>> {
    vec![
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.0, 1.0]),
    ]
}

#[test]
fn test_select_best_action() {
    let motor = ActionMotor::new(0.5);
    let act = actions();
    let ctx = make_context();
    let (idx, score) = motor.select(&ctx, &act);
    assert_eq!(idx, 0);
    assert!(score > 0.0);
}

#[test]
fn test_select_all_equal() {
    let mut ctx = DualLIFState::new(3, 0.9, 0.5);
    ctx.step(&Array1::from_vec(vec![1.0, 1.0, 1.0]), false);
    let motor = ActionMotor::new(0.5);
    let equal_acts: Vec<Array1<f64>> = (0..3).map(|_| Array1::from_vec(vec![1.0, 1.0, 1.0])).collect();
    let (idx, score) = motor.select(&ctx, &equal_acts);
    assert!(idx < 3);
    assert!(score > 0.0);
}

#[test]
fn test_select_beta_zero_picks_fast() {
    let motor = ActionMotor::new(0.0);
    let act = actions();
    let ctx = make_context();
    let (idx, _) = motor.select(&ctx, &act);
    assert_eq!(idx, 0);
}

#[test]
fn test_select_beta_one_picks_slow() {
    let motor = ActionMotor::new(1.0);
    let act = actions();
    let ctx = make_context();
    let (idx, _) = motor.select(&ctx, &act);
    assert_eq!(idx, 0);
}

#[test]
fn test_select_no_actions() {
    let motor = ActionMotor::new(0.5);
    let ctx = make_context();
    let (idx, score) = motor.select(&ctx, &[]);
    assert_eq!(idx, 0);
    assert!(score.is_infinite() && score.is_sign_negative());
}

#[test]
fn test_select_with_bonus_overrides() {
    let motor = ActionMotor::new(0.5);
    let act = actions();
    let ctx = make_context();
    let (idx, score) = motor.select_with_bonus(&ctx, &act, &[0.0, 10.0, 0.0]);
    assert_eq!(idx, 1);
    assert!(score > 0.0);
}

#[test]
fn test_select_with_bonus_no_bonus_falls_back() {
    let motor = ActionMotor::new(0.5);
    let act = actions();
    let ctx = make_context();
    let (idx, _) = motor.select_with_bonus(&ctx, &act, &[]);
    assert_eq!(idx, 0);
}

#[test]
fn test_select_with_bonus_all_negative() {
    let motor = ActionMotor::new(0.5);
    let ctx = make_context();
    let act = actions();
    let (idx, _) = motor.select_with_bonus(&ctx, &act, &[-10.0, -10.0, -10.0]);
    assert_eq!(idx, 0);
}

#[test]
fn test_new_beta_stored() {
    let motor = ActionMotor::new(0.75);
    assert!(e(motor.beta, 0.75));
}
