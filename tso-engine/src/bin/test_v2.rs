use tso_engine::tso_engine::TsoEngine;
use tso_engine::grid_world::GridWorld;

const EPS: usize = 200;

fn run_ep(engine: &mut TsoEngine, env: &mut GridWorld, ep: usize, eps_total: usize) -> bool {
    // Annealing only for Phase 1 (corridor). Phase 2 (zigzag) uses eps_total=0 to skip.
    if eps_total > 0 {
        let remain = if ep >= eps_total { 0.0 } else { (eps_total - ep) as f64 / eps_total as f64 };
        engine.cerebellum.epsilon = 0.8 * remain;
        engine.cerebellum.noise_std = 0.3 * remain;
    }

    env.reset();
    let mut p = env.perception();
    let bfs = env.bfs_at_current_pos().map(|d| (20.0 - 0.5 * d as f64).max(0.0));
    let grad = env.bfs_gradient();
    let mut a = engine.step(&p, 0.0, bfs, &grad);
    while !env.done {
        let r = env.step(a);
        if env.done { engine.step(&env.perception(), r, bfs, &[]); break; }
        p = env.perception();
        let bfs = env.bfs_at_current_pos().map(|d| (20.0 - 0.5 * d as f64).max(0.0));
        let grad = env.bfs_gradient();
        a = engine.step(&p, r, bfs, &grad);
    }
    let ok = env.agent == env.goal;
    engine.end_episode();
    ok
}

fn main() {
    // Phase 1 : Corridor horizontal 10×1 (le plus simple possible)
    eprintln!("=== Phase 1 : Corridor 10×1 (start→goal = 7 steps) ===");
    eprintln!("Shaping: phi=-2.5×bfs_frac, step=-0.01, wall=-0.5, lr_critic: δ>0 ×5");
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    engine.cerebellum.noise_std = 0.3;
    engine.cerebellum.epsilon = 0.8;

    let mut successes = Vec::new();
    let mut first = None;

    for ep in 1..=EPS {
        let ok = run_ep(&mut engine, &mut Box::new(GridWorld::straight()), ep, EPS);
        successes.push(ok);
        if ok && first.is_none() { first = Some(ep); }
        if ep % 20 == 0 || ep == EPS {
            let rate = successes.iter().filter(|&&x| x).count() as f64 / ep as f64 * 100.0;
            eprintln!("ep={:4}  success={:3}/{:4} ({:5.1}%)  first={:?}  ε={:.3}  noise={:.3}  conc={}  edges={}  Φ={:.3}  osc={}",
                ep, successes.iter().filter(|&&x| x).count(), ep, rate,
                first, engine.cerebellum.epsilon, engine.cerebellum.noise_std,
                engine.num_concepts(), engine.graph_edges(), engine.current_phi,
                engine.oscillation_breaks);
        }
    }

    let total_ok = successes.iter().filter(|&&x| x).count();
    eprintln!("\n=== RÉSULTAT CORRIDOR V2 ===");
    eprintln!("Succès: {}/{} ({:.1}%)", total_ok, EPS, total_ok as f64 / EPS as f64 * 100.0);
    eprintln!("Premier: épisode {:?}", first);
    eprintln!("Concepts: {}  Edges: {}  Φ: {:.3}", engine.num_concepts(), engine.graph_edges(), engine.current_phi);

    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    let test_ok = (0..100).filter(|_| {
        run_ep(&mut engine, &mut GridWorld::straight(), EPS, EPS)
    }).count();
    eprintln!("Exploitation pure: {}% ({} / 100)", test_ok, test_ok);

    // Phase 2 : Zigzag 10×10
    eprintln!("\n=== Phase 2 : Zigzag 10×10 (optimal ~28 steps) ===");
    let mut engine2 = TsoEngine::with_hidden(5, 4, 16);
    engine2.curiosity_weight = 0.5;
    engine2.cerebellum.noise_std = 0.1;
    engine2.cerebellum.epsilon = 0.1;

    const EPS2: usize = 500;
    successes.clear();
    first = None;

    for ep in 1..=EPS2 {
        let ok = run_ep(&mut engine2, &mut Box::new(GridWorld::corridor()), ep, 0);
        successes.push(ok);
        if ok && first.is_none() { first = Some(ep); }
        if ep % 50 == 0 || ep == EPS2 {
            let rate = successes.iter().filter(|&&x| x).count() as f64 / ep as f64 * 100.0;
            eprintln!("ep={:4}  success={:3}/{:4} ({:5.1}%)  first={:?}  conc={}  edges={}  Φ={:.3}  osc={}",
                ep, successes.iter().filter(|&&x| x).count(), ep, rate, first,
                engine2.num_concepts(), engine2.graph_edges(), engine2.current_phi,
                engine2.oscillation_breaks);
        }
    }

    let total_ok = successes.iter().filter(|&&x| x).count();
    eprintln!("\n=== RÉSULTAT ZIGZAG V2 (500 eps) ===");
    eprintln!("Succès: {}/{} ({:.1}%)", total_ok, EPS2, total_ok as f64 / EPS2 as f64 * 100.0);
    eprintln!("Premier: épisode {:?}", first);
    eprintln!("Concepts: {}  Edges: {}  Φ: {:.3}", engine2.num_concepts(), engine2.graph_edges(), engine2.current_phi);

    engine2.cerebellum.epsilon = 0.0;
    engine2.cerebellum.noise_std = 0.0;
    let test_ok2 = (0..100).filter(|_| {
        run_ep(&mut engine2, &mut GridWorld::corridor(), 0, 0)
    }).count();
    eprintln!("Exploitation pure: {}% ({} / 100)", test_ok2, test_ok2);
}
