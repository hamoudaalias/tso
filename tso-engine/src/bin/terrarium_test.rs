use tso_engine::terrarium::Terrarium;
use tso_engine::tso_engine::TsoEngine;
use std::time::Instant;

/// ──────────────────────────────────────────────────────────────────────────
///  Terrarium survie — Test final avec TsoEngine complet
///
///  Résultat des tests précédents (TinyTD pur) :
///    - 0% de succès en test (ε=0) dans TOUTES les configurations
///    - Cause : aliasing perceptuel + pas de mémoire temporelle
///
///  Ce test utilise le TsoEngine complet avec :
///    - Working memory (trace temporelle)
///    - Attractor field (catégorisation)
///    - Context buffer (contexte séquentiel)
///    - Episodic memory (prédiction)
///    - Replay buffer (TD stable)
///
///  Hypothèse : le cycle cognitif complet peut désambiguïser
///  l'aliasing et apprendre une politique exploitable sans bruit.
/// ──────────────────────────────────────────────────────────────────────────

const TRAIN_EPS: usize = 300;
const TEST_EPS: usize = 100;
const PERCEPTION_DIM: usize = 6; // 4 whiskers + food_sensed + water_sensed

fn run_ep(engine: &mut TsoEngine, env: &mut Terrarium, is_test: bool, use_grid: bool) -> f64 {
    if is_test {
        engine.cerebellum.epsilon = 0.0;
        engine.cerebellum.noise_std = 0.0;
    }

    env.reset();
    let p = if use_grid { engine.augment_perception(&env.perception(None), env.agent.0, env.agent.1) } else { env.perception(None) };
    let mut a = engine.step(&p, 0.0, None, &[]);

    while !env.done {
        let r = env.step(a);
        if env.done {
            let pt = if use_grid { engine.augment_perception(&env.perception(None), env.agent.0, env.agent.1) } else { env.perception(None) };
            engine.step(&pt, r, None, &[]);
            break;
        }
        let p = if use_grid { engine.augment_perception(&env.perception(None), env.agent.0, env.agent.1) } else { env.perception(None) };
        a = engine.step(&p, r, None, &[]);
    }
    engine.end_episode();

    // Replay training après chaque épisode
    if !is_test && engine.cerebellum.replay.len() >= 500 {
        engine.cerebellum.replay_train(64, 0.95, 10);
    }

    env.total_reward
}

fn run_config(
    label: &str,
    hidden_dim: usize,
    replay_lr: f64,
    replay_only: bool,
    curiosity: f64,
    use_grid_cells: bool,
    prune_threshold: usize,
    lr_critic_mult: f64,
) {
    let total_dim = if use_grid_cells { PERCEPTION_DIM + 1 } else { PERCEPTION_DIM };
    let mut engine = TsoEngine::with_hidden(total_dim, 4, hidden_dim);
    
    if use_grid_cells {
        engine.grid_cells.force_on(7, 7);
    } else {
        engine.grid_cells.force_off();
    }

    engine.curiosity_weight = curiosity;
    engine.cerebellum.epsilon = 0.8;
    engine.cerebellum.noise_std = 0.3;
    engine.cerebellum.replay_lr = replay_lr;
    engine.cerebellum.replay_only = replay_only;
    engine.cerebellum.set_lr_critic(0.05 * lr_critic_mult);
    engine.novelty_threshold = 0.25;
    if prune_threshold > 0 {
        engine.concept_prune_threshold = prune_threshold;
    }
    engine.sleep_every_n_episodes = 0;

    let t0 = Instant::now();
    let mut train_rewards: Vec<f64> = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.05;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;

        let total = run_ep(&mut engine, &mut Terrarium::new(ep as u64), false, use_grid_cells);
        train_rewards.push(total);
    }

    let elapsed = t0.elapsed();
    let train_avg: f64 = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_100: f64 = train_rewards[TRAIN_EPS - 100..].iter().sum::<f64>() / 100.0;
    let train_pos = train_rewards.iter().filter(|&&r| r > 0.0).count();

    // Test
    let mut test_rewards: Vec<f64> = Vec::with_capacity(TEST_EPS);
    for ep in 0..TEST_EPS {
        let total = run_ep(&mut engine, &mut Terrarium::new(1000 + ep as u64), true, use_grid_cells);
        test_rewards.push(total);
    }

    let test_avg: f64 = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_pos = test_rewards.iter().filter(|&&r| r > 0.0).count();
    let test_alive = test_rewards.iter().filter(|&&r| r > -5.0).count();
    let test_rich = test_rewards.iter().filter(|&&r| r > 50.0).count(); // vraiment réussi

    // Afficher les 10 premiers tests pour debug
    let debug_sample: Vec<f64> = test_rewards.iter().take(10).copied().collect();

    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  {:<60} ║", label);
    eprintln!("╠══════════════════════════════════════════════════════════════════╣");
    eprintln!("║  hd={} grid={} replay_lr={} rp_only={} cur={:.1} prune={} lr_c={}",
        hidden_dim, use_grid_cells, replay_lr, replay_only, curiosity,
        prune_threshold, lr_critic_mult);
    eprintln!("║  Dimensions: base={} + grid={} = {}",
        PERCEPTION_DIM, engine.grid_cells.extra_dim(),
        PERCEPTION_DIM + engine.grid_cells.extra_dim());
    eprintln!("║  TRAIN {}eps {}s  avg={:>7.1}  last100={:>7.1}  pos={}/{}",
        TRAIN_EPS, elapsed.as_secs_f64() as usize, train_avg, train_last_100,
        train_pos, TRAIN_EPS);
    eprintln!("║  TEST  {}eps ε=0  avg={:>7.1}  pos={}/{}  alive={}/{}  rich={}/{}",
        TEST_EPS, test_avg, test_pos, TEST_EPS, test_alive, TEST_EPS,
        test_rich, TEST_EPS);
    eprintln!("║  10 premiers tests: {:?}", debug_sample);
    eprintln!("║  Concepts: {}  Edges: {}  Φ: {:.3}  Replay: {}",
        engine.num_concepts(), engine.graph_edges(), engine.current_phi,
        engine.cerebellum.replay.len());
    eprintln!("╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();
}

fn main() {
    eprintln!("\n");
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  TERRARIUM SURVIE — Test TsoEngine complet                           ║");
    eprintln!("║  Avec working memory, attracteurs, contexte épisodique               ║");
    eprintln!("║  Objectif : réussir en exploitation pure (ε=0, σ=0)                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Série A : MLP limité (hd=4), sans replay ──
    run_config("A1. hd=4, base", 4, 0.0, false, 0.3, false, 0, 1.0);
    run_config("A2. hd=4, grid", 4, 0.0, false, 0.3, true, 0, 1.0);

    // ── Série B : MLP limité (hd=4), avec replay ──
    run_config("B1. hd=4, replay", 4, 0.05, false, 0.3, false, 0, 1.0);
    run_config("B2. hd=4, grid+rp", 4, 0.05, false, 0.3, true, 0, 1.0);

    // ── Série C : MLP standard (hd=16), avec replay ──
    run_config("C1. hd=16, replay", 16, 0.05, false, 0.3, false, 0, 1.0);
    run_config("C2. hd=16, grid+rp", 16, 0.05, false, 0.3, true, 0, 1.0);

    // ── Série D : Pruning actif (évite le bruit des concepts morts) ──
    run_config("D1. hd=4, grid+rp, prune", 4, 0.05, false, 0.3, true, 500, 1.0);
    run_config("D2. hd=16, grid+rp, prune", 16, 0.05, false, 0.3, true, 500, 1.0);

    // ── Série E : Sans curiosity (moins de bruit dans well-being) ──
    run_config("E1. hd=4, grid+rp, no_cur", 4, 0.05, false, 0.0, true, 500, 1.0);
    run_config("E2. hd=8, grid+rp, no_cur", 8, 0.05, false, 0.0, true, 500, 1.0);

    // ── Série F : Critic asymétrique (δ>0 appris vite) ──
    run_config("F1. hd=4, grid+rp, lr_c*5", 4, 0.05, false, 0.0, true, 500, 5.0);
    run_config("F2. hd=8, grid+rp, lr_c*5", 8, 0.05, false, 0.0, true, 500, 5.0);

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  CONCLUSION                                                           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Le problème est que le replay buffer stocke well_being               ║");
    eprintln!("║  (récompense composée), pas la récompense externe.                    ║");
    eprintln!("║  well_being inclut homéostasie, Φ, curiosité, shaping —               ║");
    eprintln!("║  signaux non reproductibles hors-contexte.                            ║");
    eprintln!("║  Solution : stocker la récompense externe dans le replay.             ║");
    eprintln!("║  Ou utiliser heartbeat_dt (temps réel) avec sleep.                     ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
