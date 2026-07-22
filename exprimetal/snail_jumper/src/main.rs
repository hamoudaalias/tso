mod game;
mod agent;
mod evolution;

use std::fs::File;
use std::io::Write;
use evolution::Evolution;

fn export_agent_json(path: &str, agent: &agent::TSOAgent) {
    let mut json = String::from("{\"prototypes\":[");
    for (ci, class_protos) in agent.field.prototypes.iter().enumerate() {
        if ci > 0 { json.push(','); }
        json.push('[');
        for (pi, p) in class_protos.iter().enumerate() {
            if pi > 0 { json.push(','); }
            json.push('[');
            for (di, v) in p.iter().enumerate() {
                if di > 0 { json.push(','); }
                json.push_str(&format!("{:.10}", v));
            }
            json.push(']');
        }
        json.push(']');
    }
    json.push_str("]}");

    let mut f = File::create(path).expect("cannot create agent file");
    f.write_all(json.as_bytes()).expect("cannot write agent");
    println!("  Agent exported to {}", path);
}

fn main() {
    println!("=== TSO Snail Jumper (Neuroevolution) ===");
    println!("Population: {} | Generations: {} | Dim: {}", 100, 50, 8);
    println!("Agent: TSO engine (LIF + LVQ1) | Evolution: crossover + mutation");
    println!("{}", "-".repeat(50));

    let mut evo = Evolution::new();
    evo.run();

    println!("{}", "-".repeat(50));
    println!("Best fitness: {}", evo.best_fitness);
    println!("Best agent prototypes: {} classes x {} protos",
        evo.agents[0].field.n_classes(),
        evo.agents[0].field.prototypes[0].len());

    export_agent_json("exprimetal/snail_jumper_py/best_agent.json", &evo.agents[0]);
}
