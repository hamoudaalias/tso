#![allow(dead_code, unused_assignments, unused_variables)]
use tso_engine::tso_engine::TsoEngine;
use tso_engine::grid_world::GridWorld;

const EPS: usize = 200;

fn run_ep(engine: &mut TsoEngine, env: &mut GridWorld, ep: usize) -> bool {
    let remain = (EPS - ep).max(0) as f64 / EPS as f64;
    engine.cerebellum.epsilon = 0.8 * remain;
    engine.cerebellum.noise_std = 0.3 * remain;

    env.reset();
    let mut total = 0.0;
    let mut p = env.perception_4d();
    let mut a = engine.step(&p, 0.0, None, &[]);
    while !env.done {
        let r = env.step_flat(a);
        total += r;
        if env.done { engine.step(&env.perception_4d(), r, None, &[]); break; }
        p = env.perception_4d();
        a = engine.step(&p, r, None, &[]);
    }
    engine.end_episode();
    env.agent == env.goal
}

fn main() {
    eprintln!("=== Test A : Zigzag 10×10, 4D pur, goal=+100, annealing ε 0.8→0 ===");
    let mut engine = TsoEngine::with_hidden(4, 4, 16);
    engine.curiosity_weight = 1.0;
    engine.cerebellum.noise_std = 0.3;
    engine.cerebellum.epsilon = 0.8;
    let _env = GridWorld::corridor();

    let mut ep_successes = 0;
    let mut first_success_ep = None;
    let mut successes: Vec<bool> = Vec::new();

    for ep in 1..=EPS {
        let ok = run_ep(&mut engine, &mut Box::new(GridWorld::corridor()), ep);
        successes.push(ok);
        if ok {
            ep_successes += 1;
            if first_success_ep.is_none() { first_success_ep = Some(ep); }
        }
        if ep % 10 == 0 || ep == EPS {
            let rate = ep_successes as f64 / ep as f64 * 100.0;
            eprintln!("ep={:4}  success={}/{:4} ({:5.1}%)  first_goal={:?}  ε={:.3}  noise={:.3}  conc={}  edges={}  Φ={:.3}  osc={}",
                ep, ep_successes, ep, rate,
                first_success_ep.unwrap_or(0),
                engine.cerebellum.epsilon, engine.cerebellum.noise_std,
                engine.num_concepts(), engine.graph_edges(), engine.current_phi,
                engine.oscillation_breaks);
        }
    }

    // Test final en exploitation pure (ε=0, noise=0)
    eprintln!("\n=== Test final : 100 épisodes en exploitation pure (ε=0, noise=0) ===");
    let mut test_env = GridWorld::corridor();
    let test_ok = (0..100).filter(|_| {
        test_env.reset();
        let mut p = test_env.perception_4d();
        engine.cerebellum.epsilon = 0.0;
        engine.cerebellum.noise_std = 0.0;
        let mut a = engine.step(&p, 0.0, None, &[]);
        loop {
            let r = test_env.step_flat(a);
            if test_env.done {
                engine.step(&test_env.perception_4d(), r, None, &[]);
                break;
            }
            p = test_env.perception_4d();
            a = engine.step(&p, r, None, &[]);
        }
        engine.end_episode();
        test_env.agent == test_env.goal
    }).count();
    eprintln!("Final exploitation rate: {}% ({} / 100)", test_ok, test_ok);

    // Résumé
    eprintln!("\n=== RÉSUMÉ TEST A ===");
    eprintln!("Config: 4D pur | goal=+100 | curiosity=1.0 | annealing ε 0.8→0 noise 0.3→0");
    eprintln!("Grille: Zigzag 10×10 (optimal ~28 steps)");
    eprintln!("Premier succès: épisode {:?}", first_success_ep);
    eprintln!("Succès total: {}/{} ({:.1}%)", ep_successes, EPS, ep_successes as f64 / EPS as f64 * 100.0);
    eprintln!("Exploitation pure: {}%", test_ok);
    eprintln!("Concepts créés: {}", engine.num_concepts());
    eprintln!("Arêtes dans le graphe: {}", engine.graph_edges());
    eprintln!("Φ final: {:.3}", engine.current_phi);
}
