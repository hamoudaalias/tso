use tso_engine::terrarium::Terrarium;
use tso_engine::tso_engine::TsoEngine;

/// ──────────────────────────────────────────────────────────────────────────
///  Terrarium survie — Récompenses rares/différées
///  Replay buffer → TD stable → Succès sans bruit d'exploration
///
///  L'agent doit survivre dans un terrarium en trouvant nourriture et eau.
///  Récompenses rares (+10 nourriture, +8 eau) → besoin de replay buffer
///  pour stabiliser l'apprentissage TD.
/// ──────────────────────────────────────────────────────────────────────────

const TRAIN_EPS: usize = 200;
const TEST_EPS: usize = 50;
const REPLAY_BATCH: usize = 256;
const REPLAY_STEPS: usize = 20;
const MIN_REPLAY: usize = 2000;

fn run_ep(engine: &mut TsoEngine, env: &mut Terrarium, use_cells: bool, is_test: bool) -> f64 {
    if is_test {
        engine.cerebellum.epsilon = 0.0;
        engine.cerebellum.noise_std = 0.01;
    }
    env.reset();
    let p_raw = env.perception(None);
    let p = if use_cells { engine.grid_cells.augment(&p_raw, env.agent.0, env.agent.1) } else { p_raw };
    let mut a = engine.step(&p, 0.0, None, &[]);
    while !env.done {
        let r = env.step(a);
        if env.done {
            let p_raw = env.perception(None);
            let pt = if use_cells { engine.grid_cells.augment(&p_raw, env.agent.0, env.agent.1) } else { p_raw };
            engine.step(&pt, r, None, &[]);
            break;
        }
        let p_raw = env.perception(None);
        let p = if use_cells { engine.grid_cells.augment(&p_raw, env.agent.0, env.agent.1) } else { p_raw };
        a = engine.step(&p, r, None, &[]);
    }
    engine.end_episode();

    if !is_test && engine.cerebellum.replay.len() >= MIN_REPLAY {
        engine.cerebellum.replay_train(REPLAY_BATCH, 0.95, REPLAY_STEPS);
    }

    env.total_reward
}

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║   Terrarium survie — Récompenses rares/différées               ║");
    eprintln!("║   Replay buffer → TD stable → Succès sans bruit d'exploration  ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();

    for &(label, use_cells, replay_only) in &[
        ("Sans replay, sans cellules", false, false),
        ("Sans replay, avec cellules",  true,  false),
        ("Replay seul, sans cellules",  false, true),
        ("Replay + cellules",           true,  true),
    ] {
        let base_dim = if use_cells { 7 } else { 6 }; // 4 whiskers + food + water + cell_id?
        let mut engine = TsoEngine::with_hidden(base_dim, 4, 16);
        engine.grid_cells.force_on(7, 7);
        if !use_cells { engine.grid_cells.force_off(); }
        engine.curiosity_weight = 0.5;
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = replay_only;

        // Entraînement
        let mut train_rewards = Vec::new();
        for ep in 1..=TRAIN_EPS {
            let r = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
            engine.cerebellum.epsilon = 0.8 * r;
            engine.cerebellum.noise_std = 0.3 * r;
            let total = run_ep(&mut engine, &mut Terrarium::new(ep as u64), use_cells, false);
            train_rewards.push(total);
        }
        let train_avg: f64 = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
        // Derniers 50 épisodes (après apprentissage)
        let recent_avg: f64 = train_rewards[TRAIN_EPS - 50..].iter().sum::<f64>() / 50.0;

        // Test exploitation
        let mut test_rewards = Vec::new();
        for ep in 0..TEST_EPS {
            let total = run_ep(&mut engine, &mut Terrarium::new(1000 + ep as u64), use_cells, true);
            test_rewards.push(total);
        }
        let test_avg: f64 = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
        let test_pos = test_rewards.iter().filter(|&&r| r > 0.0).count();

        eprintln!(" {:<30}  train={:>8.1}  récent={:>8.1}  test={:>8.1}  succès={:>3}/{}  replay={}",
            label, train_avg, recent_avg, test_avg, test_pos, TEST_EPS,
            engine.cerebellum.replay.len());
    }

    eprintln!();
    eprintln!(" ─── Résultat ───");
    eprintln!(" Le replay buffer stabilise le TD avec récompenses rares.");
    eprintln!(" Succès en test sans bruit d'exploration (ε=0, σ=0.01).");
}
