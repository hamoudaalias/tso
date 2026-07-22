mod game;
mod agent;
mod evolution;

use evolution::Evolution;

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
}
