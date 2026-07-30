use tso_engine::zigzag_grid::ZigzagGrid;

#[test]
fn test_new_creates_grid() {
    let g = ZigzagGrid::new();
    assert!(!g.done);
    assert_eq!(g.step, 0);
    assert_eq!(g.agent, (1, 1));
    assert_eq!(g.bfs.len(), 10);
    assert_eq!(g.bfs[0].len(), 10);
}

#[test]
fn test_reset_returns_to_start() {
    let mut g = ZigzagGrid::new();
    g.step_env(0);
    g.step_env(0);
    let p = g.reset();
    assert_eq!(g.agent, (1, 1));
    assert_eq!(g.step, 0);
    assert!(!g.done);
    assert_eq!(p.len(), 5);
}

#[test]
fn test_perceive_has_five_dims() {
    let g = ZigzagGrid::new();
    let p = g.perceive();
    assert_eq!(p.len(), 5);
}

#[test]
fn test_step_env_moves_agent() {
    let mut g = ZigzagGrid::new();
    g.step_env(1);
    assert_eq!(g.agent, (1, 2));
}

#[test]
fn test_step_env_wall_penalty() {
    let mut g = ZigzagGrid::new();
    let (r, _) = g.step_env(0);
    assert!(r < 0.0);
    assert_eq!(g.agent, (1, 1));
}

#[test]
fn test_step_env_boundary_penalty() {
    let mut g = ZigzagGrid::new();
    let (r, _) = g.step_env(2);
    assert!(r < 0.0);
    assert_eq!(g.agent, (1, 1));
}

#[test]
fn test_step_advances_counter() {
    let mut g = ZigzagGrid::new();
    g.step_env(1);
    assert_eq!(g.step, 1);
}

#[test]
fn test_goal_reward() {
    let mut g = ZigzagGrid::new();
    g.agent = (8, 7);
    let (r, p) = g.step_env(1);
    assert!(g.done);
    assert!((r - 20.0).abs() < 1e-6);
}

#[test]
fn test_max_steps_ends_episode() {
    let mut g = ZigzagGrid::new();
    g.step = 199;
    let (r, _) = g.step_env(1);
    assert!(g.done);
}

#[test]
fn test_done_no_further_movement() {
    let mut g = ZigzagGrid::new();
    g.agent = (8, 8);
    g.done = true;
    let (r, _) = g.step_env(1);
    assert!(g.done);
    assert!((r - 0.0).abs() < 1e-6);
}

#[test]
fn test_bfs_computed_for_start() {
    let g = ZigzagGrid::new();
    assert!(g.bfs[1][1] < usize::MAX);
}

#[test]
fn test_bfs_goal_is_zero() {
    let g = ZigzagGrid::new();
    assert_eq!(g.bfs[8][8], 0);
}
