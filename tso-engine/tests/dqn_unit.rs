//! Tests pour DqnAgent
use tso_engine::baselines::dqn::DqnAgent;
use ndarray::Array1;

#[test]
fn test_dqn_create() {
    let agent = DqnAgent::new(4, 4, 64, 0.01, 0.1);
    assert_eq!(agent.n_actions, 4);
}

#[test]
fn test_dqn_act_returns_valid_action() {
    let mut agent = DqnAgent::new(4, 4, 64, 0.01, 0.1);
    let obs = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let action = agent.act(&obs);
    assert!(action < 4);
}

#[test]
fn test_dqn_train_step_td() {
    let mut agent = DqnAgent::new(4, 4, 64, 0.01, 1.0);
    let obs = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let next = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    agent.store(&obs, 0, 1.0, &next, false);
    agent.store(&obs, 1, 0.0, &next, false);
    let loss = agent.train(2);
    assert!(loss.is_finite(), "TD loss must be finite, got {loss}");
}

#[test]
fn test_dqn_target_update_copies() {
    let mut agent = DqnAgent::new(4, 4, 64, 0.01, 0.1);
    // Modify online network
    agent.q.w1[0][0] = 42.0;
    agent.update_target(1.0);
    assert!((agent.q_target.w1[0][0] - 42.0).abs() < 1e-10, "hard update should copy online weights to target, got {}", agent.q_target.w1[0][0]);
}
