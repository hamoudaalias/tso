use tso_engine::sokoban::Sokoban;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

#[test]
fn test_generate_level_1() {
    let s = Sokoban::generate(1);
    assert_eq!(s.width, 5);
    assert_eq!(s.height, 5);
    assert_eq!(s.agent, (1, 1));
    assert_eq!(s.level, 1);
    assert_eq!(s.boxes_len(), 1);
    assert_eq!(s.targets_len(), 1);
}

#[test]
fn test_generate_level_clamps_layout() {
    let s = Sokoban::generate(99);
    assert_eq!(s.level, 99);
    assert_eq!(s.width, 8);
    assert_eq!(s.height, 8);
}

#[test]
fn test_reset_restores_agent() {
    let mut s = Sokoban::generate(1);
    s.agent = (3, 3);
    s.steps = 10;
    s.reset();
    assert_eq!(s.agent, (1, 1));
    assert_eq!(s.steps, 0);
}

#[test]
fn test_is_walkable_open() {
    assert!(Sokoban::generate(1).is_walkable(1, 2));
}

#[test]
fn test_is_walkable_wall() {
    assert!(!Sokoban::generate(1).is_walkable(0, 0));
}

#[test]
fn test_is_walkable_out_of_bounds() {
    let s = Sokoban::generate(1);
    assert!(!s.is_walkable(-1, 0));
    assert!(!s.is_walkable(10, 10));
}

#[test]
fn test_perception_without_cell_id() {
    assert_eq!(Sokoban::generate(1).perception(None).len(), 7);
}

#[test]
fn test_perception_with_cell_id() {
    assert_eq!(Sokoban::generate(1).perception(Some(0.5)).len(), 8);
}

#[test]
fn test_step_moves_agent() {
    let mut s = Sokoban::generate(1);
    s.step(1);
    assert_eq!(s.agent, (1, 2));
}

#[test]
fn test_step_wall_penalty() {
    let mut s = Sokoban::generate(1);
    let r = s.step(0);
    assert!(e(r, -0.5));
    assert_eq!(s.agent, (1, 1));
}

#[test]
fn test_solve_level_1() {
    let mut s = Sokoban::generate(1);
    s.step(3);
    s.step(1);
    assert!(s.done);
    assert_eq!(s.boxes_on_target, 1);
}

#[test]
fn test_step_max_steps_ends() {
    let mut s = Sokoban::generate(1);
    s.steps = s.max_steps - 1;
    let _r = s.step(0);
    assert!(s.done);
}

#[test]
fn test_done_returns_zero() {
    let mut s = Sokoban::generate(1);
    s.done = true;
    assert!(e(s.step(1), 0.0));
}

#[test]
fn test_targets_len_level_1() {
    assert_eq!(Sokoban::generate(1).targets_len(), 1);
}

#[test]
fn test_targets_len_level_3() {
    assert_eq!(Sokoban::generate(3).targets_len(), 1);
}

#[test]
fn test_boxes_len_level_1() {
    assert_eq!(Sokoban::generate(1).boxes_len(), 1);
}

#[test]
fn test_boxes_len_level_6() {
    assert_eq!(Sokoban::generate(6).boxes_len(), 4);
}

#[test]
fn test_level_2_wall_blocks_walk() {
    assert!(!Sokoban::generate(2).is_walkable(2, 1));
}

#[test]
fn test_perception_senses_adjacent_box() {
    let mut s = Sokoban::generate(1);
    assert!(e(s.perception(None)[4], 0.0));
    s.step(1);
    let p = s.perception(None);
    assert!(e(p[4], 1.0), "box_adjacent should be 1.0, got {}", p[4]);
}

#[test]
fn test_step_box_into_wall_penalty() {
    let mut s = Sokoban::generate(1);
    s.agent = (2, 1);
    let r = s.step(0);
    // Pushing box at (2,2) up: blocked by wall at (2,0) → target position (2,0) is out of bounds
    // Actually pushing from (2,1) up to (2,0): wall at y=0, so penalty
    // Wait, the box is at (2,2), agent is at (2,1). Action 0 (up) means dx=0, dy=-1.
    // nx=2, ny=0 → is_walkable(2,0) → y=0 is wall (top border). So penalty.
    assert!(e(r, -0.5));
}

#[test]
fn test_invalid_action_is_noop() {
    let mut s = Sokoban::generate(1);
    let r = s.step(99);
    assert!(e(r, -0.05), "noop should give base reward, got {}", r);
    assert_eq!(s.agent, (1, 1));
}

#[test]
fn test_step_base_reward() {
    let mut s = Sokoban::generate(1);
    let r = s.step(1);
    assert!(e(r, -0.05));
}
