use tso_engine::tso_engine::TsoEngine;
use tso_engine::grid_world::GridWorld;

const EPS: usize = 200;

fn run_ep(engine: &mut TsoEngine, env: &mut GridWorld, ep: usize, eps_total: usize) -> bool {
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

fn test_exploitation(engine: &mut TsoEngine) -> usize {
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    (0..100).filter(|_| {
        run_ep(engine, &mut GridWorld::straight(), EPS, EPS)
    }).count()
}

fn run_config(label: &str, engine: &mut TsoEngine) {
    let mut successes = Vec::new();
    for ep in 1..=EPS {
        let ok = run_ep(engine, &mut Box::new(GridWorld::straight()), ep, EPS);
        successes.push(ok);
    }
    let train = successes.iter().filter(|&&x| x).count();
    let exploit = test_exploitation(engine);
    eprintln!("{:<30}  train={:>3}/{:>3} ({:>5.1}%)  exploit={:>3}%  conc={:>3}  edges={:>3}  Φ={:.2}",
        label, train, EPS, train as f64 / EPS as f64 * 100.0, exploit,
        engine.num_concepts(), engine.graph_edges(), engine.current_phi);
}

fn main() {
    eprintln!("=== Ablation Study : V2 Corridor 10×1 ===");
    eprintln!("{:<30}  {:>20}  {:>8}  {:>5}  {:>5}  {:>5}",
        "Configuration", "train", "exploit", "conc", "edges", "Φ");

    // 1. CONTRÔLE : complet
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    run_config("1. Complet (référence)", &mut engine);

    // 2. Sans sommeil
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    engine.sleep_every_n_episodes = 0;
    run_config("2. Sans sommeil", &mut engine);

    // 3. Sans coût métabolique
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    engine.hypothalamus.metabolic_rate = 0.0;
    run_config("3. Sans métabolisme", &mut engine);

    // 4. Sans curiosité
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.0;
    engine.cerebellum.noise_std = 0.3;
    engine.cerebellum.epsilon = 0.8;
    run_config("4. Sans curiosité", &mut engine);

    // 5. Sans élagage conceptuel
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    engine.concept_prune_threshold = 0;
    run_config("5. Sans élagage", &mut engine);

    // 6. Sans attention (température très haute = uniforme)
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    engine.curiosity_weight = 0.5;
    engine.attention.temperature = 100.0;
    run_config("6. Sans attention", &mut engine);

    eprintln!("\n=== Fin de l'ablation ===");
}
