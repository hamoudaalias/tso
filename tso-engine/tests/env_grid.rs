//! Tests de l'Environment trait sur GridEnv.
//! Vérifie : reset, step, action_space, observation_dim.

use tso_engine::environment::{Environment, GridEnv};

#[test]
fn test_grid_env_reset() {
    let mut env = GridEnv::new();
    let obs = env.reset();
    assert_eq!(obs.len(), 6, "GridEnv observation dim should be 6");
    assert!(!env.done, "after reset, env should not be done");
}

#[test]
fn test_grid_env_step_actions() {
    let mut env = GridEnv::new();
    env.reset();
    // Toutes les actions doivent produire une observation valide
    for a in 0..4 {
        let env2 = {
            let mut e = GridEnv::new();
            e.reset();
            e
        };
        // on utilise un clone manuel
        env.agent = env2.agent;
        env.step_count = 0;
        env.done = false;
        let r = env.step(a);
        assert_eq!(r.observation.len(), 6, "observation should be 6D");
        assert!(r.reward.is_finite(), "reward should be finite");
    }
}

#[test]
fn test_grid_env_action_space() {
    let env = GridEnv::new();
    assert_eq!(env.action_space(), 4, "GridEnv should have 4 actions");
}

#[test]
fn test_grid_env_observation_dim() {
    let env = GridEnv::new();
    assert_eq!(env.observation_dim(), 6, "GridEnv observation dim 6");
}

#[test]
fn test_grid_env_step_until_done() {
    let mut env = GridEnv::new();
    env.reset();
    for _ in 0..200 {
        if env.done {
            break;
        }
        let r = env.step(0); // toujours nord
        if env.done {
            assert!(r.done, "step should report done");
        }
    }
    // doit être terminé après 150 steps max
    assert!(env.done, "GridEnv should be done within max steps");
}
