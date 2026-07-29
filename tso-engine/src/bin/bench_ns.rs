use tso_engine::rotating_t::RotatingT;
use tso_engine::tso_engine::TsoEngine;

fn main() {
    let ep = 150;
    let sw = 50;
    let seeds = 50;

    println!("=== Rotating-T (all through engine, subsystems toggled) ===\n");

    // All off = baseline (cerebellum + working_mem only)
    let off = run(ep, sw, seeds, |e| {
        e.cogs.attractor = false; e.cogs.hypothalamus = false;
        e.cogs.episodic_curiosity = false; e.cogs.attention = false;
        e.cogs.graph_phi = false; e.cogs.metabolic_cost = false;
        e.cogs.use_fpi = false;
    });
    println!("all-off (baseline):  {:7.2} ± {:5.2}", off.0, off.1);

    // Full
    let full = run(ep, sw, seeds, |e| {
        e.cogs.attractor = true; e.cogs.hypothalamus = true;
        e.cogs.episodic_curiosity = true; e.cogs.attention = true;
        e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
    });
    println!("full:                {:7.2} ± {:5.2}", full.0, full.1);

    // No episodic
    let ne = run(ep, sw, seeds, |e| {
        e.cogs.attractor = true; e.cogs.hypothalamus = true;
        e.cogs.episodic_curiosity = false; e.cogs.attention = true;
        e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
    });
    println!("no-epi:              {:7.2} ± {:5.2}", ne.0, ne.1);

    // No attention
    let na = run(ep, sw, seeds, |e| {
        e.cogs.attractor = true; e.cogs.hypothalamus = true;
        e.cogs.episodic_curiosity = true; e.cogs.attention = false;
        e.cogs.graph_phi = true; e.cogs.metabolic_cost = true;
    });
    println!("no-attn:             {:7.2} ± {:5.2}", na.0, na.1);

    println!();
    println!("Δ full – baseline:   {:7.2}", full.0 - off.0);
    println!("Δ full – no-epi:     {:7.2}", full.0 - ne.0);
    println!("Δ full – no-attn:    {:7.2}", full.0 - na.0);
}

fn run(n_ep: usize, sw: usize, seeds: usize, config: impl Fn(&mut TsoEngine)) -> (f64, f64) {
    let mut scores = Vec::new();
    for _ in 0..seeds {
        let mut engine = TsoEngine::new(5, 4);
        config(&mut engine);
        let mut rt = RotatingT::new(sw);
        let mut total = 0.0;
        for _ in 0..n_ep {
            rt.reset();
            let mut obs = rt.observation();
            let mut prev_r = 0.0;
            loop {
                let action = engine.step(&obs, prev_r, None, &[]);
                let (reward, next_obs, done) = rt.step(action);
                obs = next_obs;
                prev_r = reward;
                if done { break; }
            }
            total += prev_r;
        }
        scores.push(total / n_ep as f64);
    }
    let m = scores.iter().sum::<f64>() / scores.len() as f64;
    let v = scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / scores.len() as f64;
    (m, v.sqrt())
}
