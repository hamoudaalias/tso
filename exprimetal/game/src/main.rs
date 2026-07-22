mod grid;
mod encoder;
mod agent;

use grid::{GridWorld, Action, StepResult};
use agent::TSOAgent;
use encoder::StateEncoder;

fn run_episode_fast(gw: &mut GridWorld, agent: &mut TSOAgent) -> (usize, bool, usize, usize) {
    agent.reset();
    gw.reset();
    let encoder = StateEncoder::new();
    let mut collisions = 0;

    for step in 0..200 {
        let action = agent.act(gw);
        let pos_before = (gw.agent_x, gw.agent_y);
        let state_before = encoder.encode(gw);
        let result = gw.step(action);
        let state_after = encoder.encode(gw);
        let pos_after = (gw.agent_x, gw.agent_y);

        agent.learn(&state_before, action, result, &state_after);
        agent.q_update(result, pos_after);

        if result == StepResult::Collision {
            collisions += 1;
        }
        if gw.done {
            return (step + 1, true, collisions, agent.field.n_classes());
        }
    }
    (200, false, collisions, agent.field.n_classes())
}

fn main() {
    let walls = vec![
        (2, 1), (2, 2), (2, 3),
        (4, 3), (4, 4),
        (6, 1), (6, 2),
        (1, 4), (3, 2), (5, 1),
    ];

    let mut gw = GridWorld::new(8, 6, &walls);
    let mut agent = TSOAgent::new(16);

    println!("ep,steps,success,collisions,classes,epsilon,goal_x,goal_y");
    for episode in 0..100 {
        let (steps, success, collisions, classes) = run_episode_fast(&mut gw, &mut agent);
        println!("{},{},{},{},{},{:.3},{},{}",
            episode, steps, if success { 1 } else { 0 }, collisions, classes,
            agent.epsilon, gw.goal_x, gw.goal_y);
        agent.epsilon = (agent.epsilon * 0.99).max(0.05);
    }
}
