//! Benchmark DQN sur RotatingT
use tso_engine::rotating_t::RotatingT;
use tso_engine::baselines::dqn::DqnAgent;
use ndarray::Array1;

fn main() {
    let ep = 150;
    let sw = 50;
    let seeds = 30;

    println!("=== DQN baseline ({seeds} seeds, {ep} episodes) ===");
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut agent = DqnAgent::new(4, 4, 64, 0.001, 0.1);
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..ep {
            rt.reset();
            let mut obs = rt.observation();
            let mut prev_r = 0.0;
            loop {
                let action = agent.act(&obs);
                let (reward, next_obs, done) = rt.step(action);
                agent.store(&obs, action, prev_r, &next_obs, done);
                agent.train(32);
                obs = next_obs;
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / ep as f64);
    }
    let m = scores.iter().sum::<f64>() / seeds as f64;
    let var = scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / seeds as f64;
    println!("DQN (hidden=64):           {:7.2} ± {:5.2}", m, var.sqrt());
}
