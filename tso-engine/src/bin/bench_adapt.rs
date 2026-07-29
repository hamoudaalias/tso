//! bench_adapt: adaptation au but tournant — courbes par phase.
//! 30 seeds, 200 épisodes, switch tous les 50.
//! Mesure : reward avant/après switch, convergence.
//! Usage: cargo run --release --bin bench_adapt

use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::cerebellum::Cerebellum;
use ndarray::Array1;

fn main() {
    let n_seeds = 30;
    let n_ep = 200;
    let sw = 50;
    let n_phases = n_ep / sw;

    println!("# Adaptation but tournant — RotatingT 5×5\n");
    println!("Switch tous les {sw} épisodes, {n_phases} phases, {n_seeds} seeds\n");

    for &(name, config_fn) in &[
        ("Linear AC", bench_linear as fn() -> Vec<f64>),
        ("TSO attracteur", bench_tso_attractor as fn() -> Vec<f64>),
        ("TSO full", bench_tso_full as fn() -> Vec<f64>),
        ("TSO + épisodique", bench_tso_episodic as fn() -> Vec<f64>),
    ] {
        // Récolter les courbes pour N seeds
        let mut all_curves: Vec<Vec<f64>> = Vec::new();
        for _ in 0..n_seeds {
            all_curves.push(config_fn());
        }

        // Moyenne par épisode
        let mut mean_curve = vec![0.0; n_ep];
        for ep in 0..n_ep {
            let s: f64 = all_curves.iter().map(|c| c[ep]).sum();
            mean_curve[ep] = s / n_seeds as f64;
        }

        println!("## {name}\n");
        println!("| Phase | Avant switch (40-49) | Après switch (50-59) | Convergence (ép) |");
        println!("|-------|----------------------|----------------------|------------------|");

        for phase in 0..n_phases {
            let switch_at = phase * sw;
            let before_start = switch_at + sw - 10;
            let after_end = (phase + 1) * sw + 10;
            if after_end > n_ep { break; }

            let before: Vec<f64> = (before_start..switch_at).map(|e| mean_curve[e]).collect();
            let after: Vec<f64> = (switch_at..after_end).map(|e| mean_curve[e]).collect();

            let b_mean = mean(&before);
            let a_mean = mean(&after);

            // Convergence : premier épisode après switch où reward > 80% du plateau
            let plateau = if after_end + 40 < n_ep {
                mean(&mean_curve[after_end..after_end + 40])
            } else {
                a_mean
            };
            let threshold = 0.8 * plateau;
            let converge_ep = (switch_at..n_ep)
                .find(|&e| mean_curve[e] > threshold)
                .map(|e| e - switch_at)
                .unwrap_or(99);

            println!("| Phase {} | {:.2} | {:.2} | {} |", phase + 1, b_mean, a_mean, converge_ep);
        }
        println!();
    }
}

fn bench_linear() -> Vec<f64> {
    let mut cb = Cerebellum::new(4, 4, 0.01, 0.3, 0.1, 0);
    let mut rt = RotatingT::new(50);
    let mut curve = vec![0.0; 200];
    for ep in 0..200 {
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
        curve[ep] = prev_r;
    }
    curve
}

fn bench_tso_attractor() -> Vec<f64> {
    let mut eng = TsoEngine::new(4, 4);
    eng.cogs.attractor = true; eng.cogs.hypothalamus = false;
    eng.cogs.episodic_curiosity = false; eng.cogs.attention = false;
    eng.cogs.graph_phi = false; eng.cogs.metabolic_cost = false;
    run_tso_episodes(&mut eng)
}

fn bench_tso_full() -> Vec<f64> {
    let mut eng = TsoEngine::new(4, 4);
    eng.cogs.attractor = true; eng.cogs.hypothalamus = true;
    eng.cogs.episodic_curiosity = true; eng.cogs.attention = true;
    eng.cogs.graph_phi = true; eng.cogs.metabolic_cost = true;
    run_tso_episodes(&mut eng)
}

fn bench_tso_episodic() -> Vec<f64> {
    let mut eng = TsoEngine::new(4, 4);
    eng.cogs.attractor = true; eng.cogs.hypothalamus = false;
    eng.cogs.episodic_curiosity = true; eng.cogs.attention = false;
    eng.cogs.graph_phi = false; eng.cogs.metabolic_cost = false;
    run_tso_episodes(&mut eng)
}

fn run_tso_episodes(eng: &mut TsoEngine) -> Vec<f64> {
    let mut rt = RotatingT::new(50);
    let mut curve = vec![0.0; 200];
    for ep in 0..200 {
        rt.reset();
        let mut obs = rt.observation();
        let mut prev_r = 0.0;
        loop {
            let action = eng.step(&obs, prev_r, None, &[]);
            let (reward, next_obs, done) = rt.step(action);
            obs = next_obs; prev_r = reward;
            if done { break; }
        }
        curve[ep] = prev_r;
    }
    curve
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}
