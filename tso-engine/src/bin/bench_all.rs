//! bench_all: toutes les variantes sur RotatingT
use tso_engine::baselines::multi_seed::{run_bench, SeedResults};
use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::baselines::dqn::DqnAgent;

fn main() {
    let n_seeds = 30;
    println!("# Benchmark RotatingT — Toutes variantes\n");
    println!("N seeds = {n_seeds}, 100 episodes\n");
    println!("| Agent | Mean | σ | IC 95% | Cohen d vs Linear |");
    println!("|-------|------|---|--------|-------------------|");

    let ref_r = run_bench(n_seeds, || bench_linear());
    let ci = ref_r.ci95();
    println!("| Linear AC | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        ref_r.mean, ref_r.std, ci.0, ci.1, 0.0);

    let mlp_r = run_bench(n_seeds, || bench_mlp());
    let ci = mlp_r.ci95();
    println!("| MLP AC | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        mlp_r.mean, mlp_r.std, ci.0, ci.1, mlp_r.cohens_d(&ref_r));

    let dqn_r = run_bench(n_seeds, || bench_dqn());
    let ci = dqn_r.ci95();
    println!("| DQN | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        dqn_r.mean, dqn_r.std, ci.0, ci.1, dqn_r.cohens_d(&ref_r));

    let tso_r = run_bench(n_seeds, || bench_tso_attractor());
    let ci = tso_r.ci95();
    println!("| TSO attracteur | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        tso_r.mean, tso_r.std, ci.0, ci.1, tso_r.cohens_d(&ref_r));

    let tso_full_r = run_bench(n_seeds, || bench_tso_full());
    let ci = tso_full_r.ci95();
    println!("| TSO full | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        tso_full_r.mean, tso_full_r.std, ci.0, ci.1, tso_full_r.cohens_d(&ref_r));

    let tso_gate_r = run_bench(n_seeds, || bench_tso_gating());
    let ci = tso_gate_r.ci95();
    println!("| TSO gating | {:.2} | {:.2} | [{:.2},{:.2}] | {:.2} |",
        tso_gate_r.mean, tso_gate_r.std, ci.0, ci.1, tso_gate_r.cohens_d(&ref_r));
}

fn bench_linear() -> f64 {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 0);
    run_episodes(&mut cb)
}

fn bench_mlp() -> f64 {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 64);
    run_episodes(&mut cb)
}

fn bench_dqn() -> f64 {
    let mut agent = DqnAgent::new(4, 4, 64, 0.001, 0.1);
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset();
        let mut obs = rt.observation();
        let mut prev_r = 0.0;
        loop {
            let action = agent.act(&obs);
            let (reward, next_obs, done) = rt.step(action);
            agent.store(&obs, action, reward, &next_obs, done);
            agent.train(32);
            obs = next_obs;
            prev_r = reward;
            if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
}

fn bench_tso_attractor() -> f64 {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true;
    engine.cogs.hypothalamus = false; engine.cogs.episodic_curiosity = false;
    engine.cogs.attention = false; engine.cogs.graph_phi = false; engine.cogs.metabolic_cost = false;
    run_episodes_tso(&mut engine)
}

fn bench_tso_full() -> f64 {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true; engine.cogs.hypothalamus = true;
    engine.cogs.episodic_curiosity = true; engine.cogs.attention = true;
    engine.cogs.graph_phi = true; engine.cogs.metabolic_cost = true;
    run_episodes_tso(&mut engine)
}

fn bench_tso_gating() -> f64 {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true; engine.cogs.hypothalamus = true;
    engine.cogs.episodic_curiosity = true; engine.cogs.attention = true;
    engine.cogs.graph_phi = true; engine.cogs.metabolic_cost = true;
    engine.cogs.phi_gating = true;
    engine.cogs.phi_threshold = 0.5;
    run_episodes_tso(&mut engine)
}

fn run_episodes(cb: &mut Cerebellum) -> f64 {
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset();
        let mut obs = rt.observation();
        let mut prev_r = 0.0;
        loop {
            let logits = cb.forward_logits(&obs);
            let action = if rand::random::<f64>() < cb.epsilon {
                rand::random::<usize>() % 4
            } else {
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap()
            };
            let (reward, next_obs, done) = rt.step(action);
            cb.reinforce_td(prev_r, 0.99);
            cb.decay_trace(0.99, 0.98);
            cb.mark(&obs, action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
}

fn run_episodes_tso(engine: &mut TsoEngine) -> f64 {
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset();
        let mut obs = rt.observation();
        let mut prev_r = 0.0;
        loop {
            let action = engine.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rt.step(action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
}
