# TSO-Snail Jumper: Neuroevolution with Topographic Stabilization

## A neuromorphic agent evolved through genetic algorithms on TSO engine

**Author:** Hamouda ALIAS
**Engine:** tso-engine (LIF + LVQ1)
**Game:** Snail Jumper (dodge obstacles by switching gravity)

### Neuroevolution Approach

Standard neuroevolution (NEAT) evolves neural network weights and topologies.
TSO neuroevolution evolves **LVQ1 prototype vectors** that encode the agent's
policy. Each prototype represents a prototypical state-action pair in a
geometric embedding space.

- **Population**: 100 agents per generation
- **Genome**: 8 LVQ1 prototypes (4 per class × 2 classes)
- **Selection**: Top 25% survivors
- **Crossover**: Half from parent1, half from parent2
- **Mutation**: Gaussian noise (σ=0.1, p=0.3) on prototype vectors
- **Fitness**: Survival time (game score)

### Why TSO for Neuroevolution?

Traditional neuroevolution requires backpropagation-amenable architectures.
TSO replaces this with geometric prototype evolution: the fittest agents
are those whose LVQ1 prototypes best separate safe states from dangerous
states in cosine distance space.

### Cross-Reference: MiniGrid One-Shot

The LVQ1 prototype mechanism used in Snail Jumper is the same primitive
that drives **geometric binding** in the MiniGrid one-shot matching task
(see `exprimetal/game/paper.md`, Section 3). In both cases, the agent
stores a prototypical state-action vector — whether evolved over
generations (Snail Jumper) or learned in a single trial via
WorkingMemory (OneShot-v0; NORMAL 100% vs AMNÉSIQUE 0%). This demonstrates that TSO's prototype mechanism supports both evolutionary and instant learning within the same geometric framework.
