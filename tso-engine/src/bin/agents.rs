//! Fonctions de benchmark pour chaque variante.

use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use tso_engine::baselines::dqn::DqnAgent;
use ndarray::Array1;

macro_rules! make_bench {
    ($name:ident, $body:expr) => {
        pub fn $name() -> f64 {
            $body
        }
    };
}

make_bench!(run_linear_rotatingt, {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 0);
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
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
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
});

make_bench!(run_mlp_rotatingt, {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 64);
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
        loop {
            let logits = cb.forward_logits(&obs);
            let action = if rand::random::<f64>() < cb.epsilon { rand::random::<usize>() % 4 }
                else { logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap() };
            let (reward, next_obs, done) = rt.step(action);
            cb.reinforce_td(prev_r, 0.99); cb.decay_trace(0.99, 0.98); cb.mark(&obs, action);
            obs = next_obs; prev_r = reward; if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
});

make_bench!(run_dqn_rotatingt, {
    let mut agent = DqnAgent::new(4, 4, 64, 0.001, 0.1);
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
        loop {
            let action = agent.act(&obs);
            let (reward, next_obs, done) = rt.step(action);
            agent.store(&obs, action, prev_r, &next_obs, done);
            agent.train(32);
            obs = next_obs; prev_r = reward; if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
});

make_bench!(run_tso_attractor_rotatingt, {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true; engine.cogs.hypothalamus = false;
    engine.cogs.episodic_curiosity = false; engine.cogs.attention = false;
    engine.cogs.graph_phi = false; engine.cogs.metabolic_cost = false;
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
        loop {
            let action = engine.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rt.step(action);
            obs = next_obs; prev_r = reward; if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
});

make_bench!(run_tso_full_rotatingt, {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true; engine.cogs.hypothalamus = true;
    engine.cogs.episodic_curiosity = true; engine.cogs.attention = true;
    engine.cogs.graph_phi = true; engine.cogs.metabolic_cost = true;
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
        loop {
            let action = engine.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rt.step(action);
            obs = next_obs; prev_r = reward; if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
});

make_bench!(run_tso_gating_rotatingt, {
    let mut engine = TsoEngine::new(4, 4);
    engine.cogs.attractor = true; engine.cogs.hypothalamus = true;
    engine.cogs.episodic_curiosity = true; engine.cogs.attention = true;
    engine.cogs.graph_phi = true; engine.cogs.metabolic_cost = true;
    engine.cogs.phi_gating = true;
    engine.cogs.phi_threshold = 0.5;
    let mut rt = RotatingT::new(50);
    let mut total = 0.0;
    for _ in 0..100 {
        rt.reset(); let mut obs = rt.observation(); let mut prev_r = 0.0;
        loop {
            let action = engine.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rt.step(action);
            obs = next_obs; prev_r = reward; if done { break; }
        }
        total += prev_r;
    }
    total / 100.0
});
