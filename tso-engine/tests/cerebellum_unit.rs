use ndarray::Array1;
use tso_engine::cerebellum::Cerebellum;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

fn s() -> Array1<f64> { Array1::from_vec(vec![0.5, 0.3, 0.8, 0.1]) }
fn s2() -> Array1<f64> { Array1::from_vec(vec![0.6, 0.2, 0.7, 0.3]) }

// ── Linear mode ──

#[test]
fn test_new_linear() {
    let c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 0);
    assert!(c.is_linear());
    assert!(e(c.lr, 0.01));
    assert_eq!(c.n_actions, 3);
}

#[test]
fn test_forward_linear() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 0);
    for _ in 0..10 { assert!(c.forward(&s()) < 3); }
}

#[test]
fn test_forward_logits_linear() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 0);
    assert_eq!(c.forward_logits(&s()).len(), 3);
}

#[test]
fn test_learn_linear_changes_weights() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    let w0 = c.get_lin_weight(0, 0);
    c.learn(&s(), 0, 1.0);
    assert!(!e(c.get_lin_weight(0, 0), w0));
}

#[test]
fn test_learn_linear_zero_reward_skips() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    let w0 = c.get_lin_weight(0, 0);
    c.learn(&s(), 0, 0.0);
    assert!(e(c.get_lin_weight(0, 0), w0));
}

#[test]
fn test_reinforce_linear_changes_weights() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    c.mark(&s(), 0);
    let w0 = c.get_lin_weight(0, 0);
    c.reinforce(1.0);
    assert!(!e(c.get_lin_weight(0, 0), w0));
}

#[test]
fn test_reinforce_linear_zero_reward_skips() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    c.mark(&s(), 0);
    let w0 = c.get_lin_weight(0, 0);
    c.reinforce(0.0);
    assert!(e(c.get_lin_weight(0, 0), w0));
}

#[test]
fn test_epsilon_exploration() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.1, 1.0, 0);
    let mut saw_diff = false;
    for _ in 0..100 {
        if c.forward(&s()) != 0 { saw_diff = true; break; }
    }
    assert!(saw_diff);
}

#[test]
fn test_reset_trace_linear() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    c.mark(&s(), 0);
    c.reinforce(1.0);
    c.reset_trace();
    // After reset_trace + reinforce, weights should not change
    let w0 = c.get_lin_weight(0, 0);
    c.reinforce(1.0);
    assert!(e(c.get_lin_weight(0, 0), w0));
}

// ── MLP mode ──

#[test]
fn test_new_mlp() {
    let c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    assert!(!c.is_linear());
    assert_eq!(c.n_actions, 3);
}

#[test]
fn test_forward_mlp() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    for _ in 0..10 { assert!(c.forward(&s()) < 3); }
}

#[test]
fn test_forward_logits_mlp() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    assert_eq!(c.forward_logits(&s()).len(), 3);
}

#[test]
fn test_forward_with_hidden_mlp() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    let (a, h) = c.forward_with_hidden(&s());
    assert!(a < 3);
    assert_eq!(h.len(), 8);
}

#[test]
fn test_predict_value_mlp() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    c.forward_logits(&s());
    assert!(e(c.predict_value(), 0.0));
}

#[test]
fn test_mark_and_reinforce_td_updates_weights() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    let w0 = c.get_out_weight(0, 0);
    c.forward_logits(&s());
    c.mark(&s(), 0);
    c.forward_logits(&s2());
    c.reinforce_td(1.0, 0.9);
    assert!(!e(c.get_out_weight(0, 0), w0));
}

#[test]
fn test_reinforce_td_sets_last_delta() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.forward_logits(&s());
    c.mark(&s(), 0);
    c.forward_logits(&s2());
    c.reinforce_td(1.0, 0.9);
    assert!(c.last_delta != 0.0);
}

#[test]
fn test_reinforce_td_zero_reward_skips() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.forward_logits(&s());
    c.mark(&s(), 0);
    c.forward_logits(&s2());
    let w0 = c.get_out_weight(0, 0);
    c.reinforce_td(0.0, 0.9);
    assert!(e(c.get_out_weight(0, 0), w0));
}

#[test]
fn test_decay_trace_does_not_panic() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.forward_logits(&s());
    c.mark(&s(), 0);
    c.decay_trace(0.9, 0.7);
    c.forward_logits(&s2());
    c.reinforce_td(1.0, 0.9);
    // If we get here, decay_trace + reinforce_td didn't panic
}

#[test]
fn test_store_transition() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.store_transition(&s(), 0, 1.0, &s2(), false);
    assert_eq!(c.replay.len(), 1);
}

#[test]
fn test_replay_only_skips_online_td() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.replay_only = true;
    c.forward_logits(&s());
    c.mark(&s(), 0);
    c.forward_logits(&s2());
    let w0 = c.get_out_weight(0, 0);
    c.reinforce_td(1.0, 0.9);
    assert!(e(c.get_out_weight(0, 0), w0));
}

#[test]
fn test_replay_train() {
    let mut c = Cerebellum::new(4, 3, 0.1, 0.0, 0.0, 8);
    for _ in 0..20 {
        c.store_transition(&s(), 0, 1.0, &s2(), false);
        c.store_transition(&s2(), 1, 0.0, &s(), true);
    }
    let avg = c.replay_train(4, 0.9, 5);
    assert!(avg >= 0.0);
}

#[test]
fn test_replay_train_empty_buffer() {
    let mut c = Cerebellum::new(4, 3, 0.1, 0.0, 0.0, 8);
    assert!(e(c.replay_train(4, 0.9, 5), 0.0));
}

#[test]
fn test_compute_cost() {
    let c_lin = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 0);
    let c_mlp = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    assert!(e(c_lin.compute_cost(), 1.0));
    assert!(e(c_mlp.compute_cost(), 2.0));
}

#[test]
fn test_set_lr_critic() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    c.set_lr_critic(0.05);
    assert!(e(c.critic_learning_rate(), 0.05));
}

#[test]
fn test_linear_reinforce_td_falls_back() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 0);
    c.mark(&s(), 0);
    let w0 = c.get_lin_weight(0, 0);
    c.reinforce_td(1.0, 0.9);
    assert!(!e(c.get_lin_weight(0, 0), w0));
}

#[test]
fn test_inspection_helpers_linear() {
    let c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 0);
    assert!(!e(c.get_lin_weight(0, 0), 0.0));
    assert!(e(c.get_hidden_weight(0, 0), 0.0));
    assert!(e(c.get_out_weight(0, 0), 0.0));
}

#[test]
fn test_inspection_helpers_mlp() {
    let c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    assert!(e(c.get_lin_weight(0, 0), 0.0));
    assert!(!e(c.get_hidden_weight(0, 0), 0.0));
    assert!(!e(c.get_out_weight(0, 0), 0.0));
}

#[test]
fn test_get_hidden() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    c.forward_logits(&s());
    assert_eq!(c.get_hidden().len(), 8);
}

#[test]
fn test_reset_mlp_clears_state() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.learn(&s(), 0, 1.0);
    c.reset();
    // After reset, forward should still produce valid actions
    assert!(c.forward(&s()) < 3);
}

#[test]
fn test_cache_value() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.forward_logits(&s());
    // cache_value should not crash
    c.cache_value();
}

#[test]
fn test_forward_deterministic_without_noise_mlp() {
    let mut c = Cerebellum::new(4, 3, 0.01, 0.0, 0.0, 8);
    let a1 = c.forward(&s());
    let a2 = c.forward(&s());
    assert_eq!(a1, a2);
}

#[test]
fn test_reset_frees_memory() {
    let mut c = Cerebellum::new(4, 3, 0.5, 0.0, 0.0, 8);
    c.reset();
    assert!(c.forward(&s()) < 3);
}
