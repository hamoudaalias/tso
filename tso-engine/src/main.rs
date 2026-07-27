use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tso_engine::tso_engine::{TsoEngine, SleepReport};
use tso_engine::grid_world::GridWorld;

const TICK_MS: u64 = 100;
const TICK: Duration = Duration::from_millis(TICK_MS);
const SAVE_PATH: &str = "tso_state.bin";
const AUTOSAVE_EPISODES: usize = 10;

fn draw_bar(val: f64, w: usize) -> String {
    let filled = (val * w as f64).round() as usize;
    let filled = filled.min(w);
    let mut s = String::with_capacity(w + 2);
    s.push('[');
    for i in 0..w {
        s.push(if i < filled { '█' } else { '░' });
    }
    s.push(']');
    s
}

fn draw_grid(env: &GridWorld) -> Vec<String> {
    let mut rows = Vec::with_capacity(env.height);
    for y in 0..env.height {
        let mut row = String::with_capacity(env.width);
        for x in 0..env.width {
            if env.walls[x][y] {
                row.push('█');
            } else if (x, y) == env.agent {
                row.push('●');
            } else if (x, y) == env.goal {
                row.push('★');
            } else {
                row.push(' ');
            }
        }
        rows.push(row);
    }
    rows
}

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let mut engine = if let Ok(loaded) = TsoEngine::load(SAVE_PATH) {
        eprintln!("\x1b[2J\x1b[HLoaded saved brain state from {}", SAVE_PATH);
        loaded
    } else {
        let mut e = TsoEngine::with_hidden(4, 4, 16);
        e.curiosity_weight = 0.5;
        e.phi_threshold = 2.0;
        e.cerebellum.epsilon = 0.6;
        e.cerebellum.noise_std = 0.2;
        e
    };

    // L-shaped corridor — the organism lives in ONE consistent world and must
    // learn to navigate it. A real organism doesn't get a new planet each episode.
    // The corridor is deliberately simple (one decision point) so that random
    // exploration discovers the goal quickly, giving the cerebellum its first
    // positive reward signal. Maze complexity should grow with the organism.
    let mut env = GridWorld::corridor();

    let t0 = Instant::now();
    let mut step_count = 0usize;
    let mut goal_count = 0usize;
    let mut ep_count = 0usize;
    let mut last_autosave_ep = 0usize;
    let mut last_sleep: Option<SleepReport> = None;
    let mut display_timer = Instant::now();

    env.reset();
    let mut p = env.perception_4d();
    let mut last_tick = Instant::now();
    let saved_eps = engine.episode_count();
    let mut a = engine.heartbeat_dt(&p, 0.0, 0.0);

    eprint!("\x1b[2J\x1b[H");

    while running.load(Ordering::SeqCst) {
        let tick_start = Instant::now();
        let dt = tick_start.duration_since(last_tick).as_secs_f64().clamp(0.001, 1.0);
        last_tick = tick_start;

        let ext_r = env.step_flat(a);
        let bonus = env.exploration_bonus();
        let r = ext_r + bonus;
        step_count += 1;

        if env.done {
            let p_term = env.perception_4d();
            engine.heartbeat_dt(&p_term, r, dt);
            if env.agent == env.goal {
                goal_count += 1;
                // Exploration burst after finding the goal: temporarily boost
                // epsilon so the organism re-discovers the goal quickly.
                // This creates a virtuous cycle of discovery → learning → exploration.
                engine.cerebellum.epsilon = engine.cerebellum.epsilon.max(0.5);
                engine.cerebellum.noise_std = engine.cerebellum.noise_std.max(0.1);
            }
            engine.end_episode();
            ep_count += 1;

            // Reset homeostasis at episode boundaries so each episode starts
            // with a clear signal. Without this, the organism lives in permanent
            // max deficit and the constant -0.5 penalty drowns all learning.
            engine.hypothalamus.energy = 1.0;
            engine.hypothalamus.hydration = 1.0;
            engine.hypothalamus.temperature = 0.5;

            // ── Sleep / Consolidation Phase ──
            // Periodically the organism replays episodic traces offline,
            // consolidates attractor prototypes, and resolves semantic
            // graph conflicts with many iterations.  Sensor input stops.
            // Sleep is triggered by either built-up sleep pressure (awake
            // steps exceeding threshold) or by the episode-interval timer.
            if engine.should_sleep(engine.episode_count()) {
                last_sleep = Some(engine.sleep_cycle());
            }

            // Stay in the same world — the organism learns this maze for life.
            env.reset();
            p = env.perception_4d();
            a = engine.heartbeat_dt(&p, 0.0, dt);

            if ep_count - last_autosave_ep >= AUTOSAVE_EPISODES {
                let _ = engine.save(SAVE_PATH);
                last_autosave_ep = ep_count;
            }
        } else {
            p = env.perception_4d();
            a = engine.heartbeat_dt(&p, r, dt);
        }

        // Real-time decay: epsilon halves every ~7 seconds. Floor at 0.2 keeps
        // enough random exploration to rediscover the goal periodically, which
        // prevents Q-value extinction from the constant deficit penalty.
        // Noise_std decays independently to 0.02 for ongoing subtle perturbation.
        if engine.cerebellum.epsilon > 0.2 {
            engine.cerebellum.epsilon *= 0.99_f64.powf(dt * 10.0);
        }
        if engine.cerebellum.noise_std > 0.02 {
            engine.cerebellum.noise_std *= 0.995_f64.powf(dt * 10.0);
        }
        if engine.curiosity_weight > 0.01 {
            engine.curiosity_weight *= 0.99_f64.powf(dt * 10.0);
        }

        if display_timer.elapsed() >= Duration::from_millis(200) {
            let homeo = engine.hypothalamus.homeostatic_state();
            let elapsed = t0.elapsed();
            let rate = step_count as f64 / elapsed.as_secs_f64().max(0.001);
            let deficit = engine.hypothalamus.total_deficit();
            let drive = engine.hypothalamus.total_drive();
            let grid = draw_grid(&env);
            let total_lines = 9 + env.height;

            eprint!("\x1b[{}A", total_lines);

            eprintln!("╔═══ TSO Engine — Real-Time Living Organism ═══╗");
            eprintln!("  Fixed World {}×{} │ 10 Hz │ Ctrl+C to stop", env.width, env.height);
            eprintln!("");
            for row in &grid {
                eprint!("  ");
                for ch in row.chars() {
                    match ch {
                        '●' => eprint!("\x1b[33m●\x1b[0m"),
                        '★' => eprint!("\x1b[32m★\x1b[0m"),
                        '█' => eprint!("\x1b[90m█\x1b[0m"),
                        _ => eprint!(" "),
                    }
                }
                eprintln!("");
            }
            eprintln!("");
            let nts = &engine.concept_novelty_thresholds;
            let nt_min = if nts.is_empty() { engine.novelty_threshold } else { nts.iter().copied().fold(f64::MAX, f64::min) };
            let nt_max = if nts.is_empty() { engine.novelty_threshold } else { nts.iter().copied().fold(f64::MIN, f64::max) };
            let nt_mean = if nts.is_empty() { engine.novelty_threshold } else { nts.iter().copied().sum::<f64>() / nts.len() as f64 };
            eprintln!("  Step {:>6}  Goals {:>3}  Ep {:>4}  {:5.0} st/s  {:>4.0}s  Φ={:.3}  conc={}  mem={}  {}",
                step_count, goal_count, ep_count + saved_eps, rate, elapsed.as_secs_f64(),
                engine.current_phi, engine.attractor.n_classes(),
                engine.episodic_size(),
                if engine.anxious { "\x1b[31m⚠ ANXIOUS\x1b[0m" } else { "" });
            let sp = engine.sleep_pressure();
            let sleep_bar = if sp > 0.0 {
                format!("  sleep={:.0}%", sp * 100.0)
            } else {
                String::new()
            };
            eprintln!("  Drive {:.2}  Deficit {:.2}  ε={:.3}  curiosity={:.3}  noise={:.3}  edges={}{}",
                drive, deficit, engine.cerebellum.epsilon, engine.curiosity_weight, engine.cerebellum.noise_std,
                engine.graph_edges(), sleep_bar);
            eprintln!("  Pos ({:>2},{:>2})  Goal ({},{})  Steps {:>3}/{}  dt={:.0}ms  bonus={:.3}",
                env.agent.0, env.agent.1, env.goal.0, env.goal.1,
                env.steps, env.max_steps, dt * 1000.0, bonus);
            eprintln!("  threshold  min={:.3}  max={:.3}  mean={:.3}",
                nt_min, nt_max, nt_mean);
            if let Some(ref sr) = last_sleep {
                eprintln!("  sleep #{}  Φ {:.3}→{:.3}  replay={}  +{}proto  -{}proto  -{}edges  conc={}",
                    engine.sleep_cycles, sr.phi_before, sr.phi_after,
                    sr.replay_count, sr.prototypes_added, sr.prototypes_pruned,
                    sr.edges_removed, engine.attractor.n_classes());
                last_sleep = None;
            } else {
                eprintln!("  sleep={}  Φ={:.3}  edges={}  conc={}  mem={}",
                    engine.sleep_cycles, engine.current_phi,
                    engine.graph_edges(), engine.attractor.n_classes(),
                    engine.episodic_size());
            }
            let sleep_bar_val = 1.0 - sp;
            let mc = &engine.hypothalamus;
            let habits = engine.habit_counts.len();
            eprintln!("  Energy    {:>5.2} {}  Hydration {:>5.2} {}  Temp {:>5.2} {}  Sleep {:>5.2} {}",
                homeo[0], draw_bar(homeo[0], 10),
                homeo[1], draw_bar(homeo[1], 10),
                homeo[2], draw_bar(homeo[2], 10),
                sp, draw_bar(sleep_bar_val, 10));
            eprintln!("  Metab. cerebellum={:.4}  graph={:.4}  motor={:.4}  total={:.4}  habits={}",
                mc.cerebellum_cost, mc.graph_cost, mc.motor_cost, mc.total_cost, habits);

            std::io::stderr().flush().ok();
            display_timer = Instant::now();
        }

        let elapsed_tick = tick_start.elapsed();
        if elapsed_tick < TICK {
            std::thread::sleep(TICK - elapsed_tick);
        }
    }

    let _ = engine.save(SAVE_PATH);
    eprintln!("\nSaved brain state to {}", SAVE_PATH);
}
