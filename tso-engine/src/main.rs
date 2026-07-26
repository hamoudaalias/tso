use std::io::Write;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::grid_world::GridWorld;

fn bfs_val(env: &GridWorld) -> Option<f64> {
    env.bfs_at_current_pos().map(|d| (20.0 - 0.5 * d as f64).max(0.0))
}

fn run_ep(engine: &mut TsoEngine, env: &mut GridWorld, noise: f64) -> f64 {
    engine.cerebellum.noise_std = noise;
    env.reset();
    let mut total = 0.0;
    let mut p = env.perception();
    let mut a = engine.step(&p, 0.0, bfs_val(env), &env.bfs_gradient());
    while !env.done {
        let r = env.step(a); total += r;
        if env.done {
            engine.step(&env.perception(), r, bfs_val(env), &env.bfs_gradient());
            break;
        }
        p = env.perception();
        a = engine.step(&p, r, bfs_val(env), &env.bfs_gradient());
    }
    engine.end_episode();
    total
}

fn eval(maze: &str, env: GridWorld, episodes: usize, dim: usize) {
    let mut engine = TsoEngine::new(dim, 4);
    let mut env = env;
    eprint!("{:18} {} eps: ", maze, episodes);
    let t0 = Instant::now();

    let mut train_ok = 0usize;
    for i in 0..episodes {
        if run_ep(&mut engine, &mut env, 0.1) > 0.0 { train_ok += 1; }
        if i % 100 == 99 { eprint!("."); std::io::stderr().flush().ok(); }
    }
    let mut test_ok = 0usize;
    for _ in 0..200 {
        if run_ep(&mut engine, &mut env, 0.0) > 0.0 { test_ok += 1; }
    }
    eprintln!("  train {:.1}%  test {:.1}%  |  {:?}  |  {} concepts  |  {} trans",
        train_ok as f64 / episodes as f64 * 100.0, test_ok as f64 / 200.0 * 100.0, t0.elapsed(),
        engine.attractor.n_classes(), engine.trans_log.len());
}

fn main() {
    eprintln!("╔═══ TSO Engine — 1000 eps ═══╗\n");
    let t0 = Instant::now();
    eval("Empty Room 5×5", GridWorld::empty_room(), 1000, 5);
    eval("L-Maze 7×7", GridWorld::l_maze(), 1000, 5);
    eval("Zigzag 10×10", GridWorld::zigzag(), 1000, 5);
    eprintln!("\n╚═══ {:?} ═══╝", t0.elapsed());
}
