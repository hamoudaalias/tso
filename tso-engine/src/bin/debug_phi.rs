//! Debug phi gating — prints graph state after each episode
use tso_engine::{TsoEngine, minigrid_env::MiniGridEnv};
fn main() {
    let mut env = MiniGridEnv::new();
    let mut eng = TsoEngine::with_hidden(147, 7, 0);
    eng.cogs.graph_phi = true;
    eng.cogs.phi_gating = false;
    let mut total = 0.0;
    for ep in 0..10 {
        let mut obs = env.reset();
        eng.end_episode();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (r, o, done) = env.step(action);
            obs = o; prev_r = r;
            if done { total += r; break; }
        }
        let subs = eng.cogs.subsystems();
        eprintln!("ep={} edges={} nodes={} steps={} concepts={} cogs{:?}",
            ep, eng.graph.edges.len(), eng.graph.nodes.len(),
            eng.total_steps, eng.attractor.prototypes.len(),
            eng.cogs);
    }
    eprintln!("reward={:.3}", total / 10.0);
}
