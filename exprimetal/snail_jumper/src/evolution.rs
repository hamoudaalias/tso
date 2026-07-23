use tso_engine::attractor::AttractorField;
use crate::agent::TSOAgent;
use crate::game::GameState;

const POP_SIZE: usize = 100;
const EMBED_DIM: usize = 8;
const GENERATIONS: usize = 20;

pub struct Evolution {
    pub agents: Vec<TSOAgent>,
    pub generation: usize,
    pub best_fitness: u64,
}

impl Evolution {
    pub fn new() -> Self {
        let mut agents = Vec::with_capacity(POP_SIZE);
        for _ in 0..POP_SIZE {
            agents.push(TSOAgent::new());
        }
        Evolution { agents, generation: 0, best_fitness: 0 }
    }

    pub fn evaluate(&mut self) {
        for agent in &mut self.agents {
            let mut game = GameState::new(500.0, 700.0);
            while game.alive {
                let left = agent.decide(&game);
                game.step(left);
                agent.learn(&game);
            }
            agent.fitness = game.score;
            if game.score > self.best_fitness {
                self.best_fitness = game.score;
            }
        }
        self.agents.sort_by(|a, b| b.fitness.cmp(&a.fitness));
    }

    pub fn next_generation(&mut self) {
        self.evaluate();

        let avg: f64 = self.agents.iter().map(|a| a.fitness as f64).sum::<f64>() / POP_SIZE as f64;
        println!("Gen {:>3} | best={:>4} | avg={:.1}", self.generation, self.agents[0].fitness, avg);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let survivors = self.agents[..POP_SIZE / 4].to_vec();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut next = Vec::with_capacity(POP_SIZE);

        while next.len() < POP_SIZE {
            let p1 = &survivors[rng.gen_range(0..survivors.len())];
            let p2 = &survivors[rng.gen_range(0..survivors.len())];
            let child = Self::crossover(p1, p2);
            next.push(Self::mutate(child));
        }

        self.agents = next;
        self.generation += 1;
    }

    fn crossover(p1: &TSOAgent, p2: &TSOAgent) -> TSOAgent {
        let mut field = AttractorField::new(EMBED_DIM, 0, 0, 0.05);

        for class_idx in 0..p1.field.prototypes.len().min(p2.field.prototypes.len()) {
            let protos1 = &p1.field.prototypes[class_idx];
            let protos2 = &p2.field.prototypes[class_idx];

            let mut child_protos = Vec::new();
            let half = protos1.len() / 2;
            for (i, p) in protos1.iter().enumerate() {
                if i < half {
                    child_protos.push(p.clone());
                } else if i < half + protos2.len() {
                    child_protos.push(protos2[i - half].clone());
                }
            }

            if child_protos.is_empty() { continue; }

            field.add_class(&child_protos[0]);
            for p in &child_protos[1..] {
                field.add_prototype(p, class_idx);
            }
        }

        TSOAgent::from_field(field)
    }

    fn mutate(mut agent: TSOAgent) -> TSOAgent {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for class_protos in &mut agent.field.prototypes {
            for p in class_protos {
                if rng.r#gen::<f64>() < 0.3 {
                    let noise: f64 = rng.r#gen::<f64>() * 0.2 - 0.1;
                    *p = p.clone() + noise;
                    let n = p.dot(p).sqrt().max(1e-12);
                    *p /= n;
                }
            }
        }
        agent
    }

    pub fn run(&mut self) {
        for _ in 0..GENERATIONS {
            self.next_generation();
        }
    }
}
