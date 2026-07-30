use tso_engine::grid_world::GridWorld;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

#[test]
fn test_empty_room_creation() {
    let g = GridWorld::empty_room();
    assert_eq!(g.width, 5);
    assert_eq!(g.height, 5);
    assert_eq!(g.agent, (1, 1));
    assert_eq!(g.goal, (3, 3));
    assert!(!g.done);
    assert_eq!(g.max_steps, 50);
}

#[test]
fn test_straight_creation() {
    let g = GridWorld::straight();
    assert_eq!(g.width, 10);
    assert_eq!(g.height, 1);
    assert_eq!(g.agent, (1, 0));
    assert_eq!(g.goal, (8, 0));
}

#[test]
fn test_corridor_creation() {
    let g = GridWorld::corridor();
    assert_eq!(g.width, 10);
    assert_eq!(g.height, 10);
    assert_eq!(g.agent, (1, 1));
    assert_eq!(g.goal, (8, 8));
}

#[test]
fn test_random_creation() {
    let g = GridWorld::random(10, 10);
    assert_eq!(g.width, 10);
    assert_eq!(g.height, 10);
    assert_eq!(g.agent, (1, 1));
    assert!(!g.is_wall(1, 1));
    assert!(g.goal.0 > 0 && g.goal.1 > 0);
}

#[test]
fn test_reset() {
    let mut g = GridWorld::empty_room();
    g.step(1);
    assert!(g.agent != (1, 1));
    g.reset();
    assert_eq!(g.agent, (1, 1));
    assert!(!g.done);
    assert_eq!(g.steps, 0);
}

#[test]
fn test_is_wall_boundary() {
    let g = GridWorld::empty_room();
    assert!(g.is_wall(-1, 0));
    assert!(g.is_wall(0, -1));
    assert!(g.is_wall(5, 0));
    assert!(g.is_wall(0, 5));
}

#[test]
fn test_is_wall_inner() {
    let g = GridWorld::empty_room();
    assert!(g.is_wall(0, 0));
    assert!(!g.is_wall(1, 1));
}

#[test]
fn test_perception_4d() {
    let g = GridWorld::empty_room();
    let p = g.perception_4d();
    assert_eq!(p.len(), 4);
}

#[test]
fn test_perception_5d() {
    let g = GridWorld::empty_room();
    let p = g.perception();
    assert_eq!(p.len(), 5);
}

#[test]
fn test_perception_whiskers_down_right_open() {
    let g = GridWorld::empty_room();
    let p = g.perception_4d();
    // order: [up, down, right, left]
    assert!(p[0] == 0.0);  // up = wall immediately
    assert!(p[1] > 0.0);   // down = open
    assert!(p[2] > 0.0);   // right = open
    assert!(p[3] == 0.0);  // left = wall immediately
}

#[test]
fn test_step_moves_agent() {
    let mut g = GridWorld::empty_room();
    g.step(1);
    assert_eq!(g.agent, (1, 2));
}

#[test]
fn test_step_wall_penalty() {
    let mut g = GridWorld::empty_room();
    let r = g.step(0);
    assert!(e(r, -0.5));
    assert_eq!(g.agent, (1, 1));
}

#[test]
fn test_step_goal_reward() {
    let mut g = GridWorld::empty_room();
    g.agent = (3, 2);
    let r = g.step(1);
    assert!(g.done);
    assert!(r > 19.0);
}

#[test]
fn test_step_max_steps() {
    let mut g = GridWorld::empty_room();
    g.steps = 49;
    let _r = g.step(0);
    assert!(g.done);
}

#[test]
fn test_step_flat() {
    let mut g = GridWorld::empty_room();
    let r = g.step_flat(1);
    assert!(e(r, 0.0));
    assert_eq!(g.agent, (1, 2));
}

#[test]
fn test_step_flat_wall() {
    let mut g = GridWorld::empty_room();
    let r = g.step_flat(0);
    assert!(e(r, -0.5));
}

#[test]
fn test_step_flat_goal() {
    let mut g = GridWorld::empty_room();
    g.agent = (3, 2);
    let r = g.step_flat(1);
    assert!(e(r, 20.0));
    assert!(g.done);
}

#[test]
fn test_done_returns_zero() {
    let mut g = GridWorld::empty_room();
    g.done = true;
    assert!(e(g.step(1), 0.0));
    assert!(e(g.step_flat(1), 0.0));
}

#[test]
fn test_bfs_gradient_length() {
    let g = GridWorld::straight();
    let grad = g.bfs_gradient();
    assert_eq!(grad.len(), 4);
}

#[test]
fn test_bfs_gradient_wall_marker() {
    let g = GridWorld::empty_room();
    let grad = g.bfs_gradient();
    // At (1,1) in empty room: up = wall (-999), down = free, left = wall (-999), right = free
    assert!(e(grad[0], -999.0));
    assert!(e(grad[2], -999.0));
}

#[test]
fn test_open_cells() {
    let g = GridWorld::empty_room();
    let cells = g.open_cells();
    let total = cells.len();
    let all_wall = cells.iter().all(|&(x, y)| !g.walls[x][y]);
    assert!(all_wall);
    assert!(total > 0 && total < 25);
}

#[test]
fn test_exploration_bonus_decays() {
    let mut g = GridWorld::empty_room();
    let b1 = g.exploration_bonus();
    g.visit_count[1][1] = 100;
    let b2 = g.exploration_bonus();
    assert!(b2 < b1);
}

#[test]
fn test_render_ascii_nonempty() {
    let g = GridWorld::empty_room();
    let s = g.render_ascii();
    assert!(!s.is_empty());
    assert!(s.contains('@'));
    assert!(s.contains('G'));
}

#[test]
fn test_bfs_at_current_pos() {
    let g = GridWorld::empty_room();
    let d = g.bfs_at_current_pos();
    assert!(d.is_some());
    assert!(d.unwrap() > 0);
}

#[test]
fn test_step_count_norm() {
    let mut g = GridWorld::empty_room();
    assert!(e(g.step_count_norm(), 0.0));
    g.steps = 25;
    assert!(e(g.step_count_norm(), 0.5));
}

#[test]
fn test_visit_count_increments_step_flat() {
    let mut g = GridWorld::empty_room();
    g.step_flat(1);
    assert_eq!(g.visit_count[1][2], 1);
}

#[test]
fn test_wall_visit_count_increments_flat() {
    let mut g = GridWorld::empty_room();
    g.step_flat(0);
    assert_eq!(g.visit_count[1][1], 1);
}
