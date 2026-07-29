# Baselines MLP/DQN/SNN

## Story s01: DQN baseline
- **s01t01** — Créer `src/baselines/mod.rs` avec exports
- **s01t02** — Implémenter `DqnAgent` :
  - Q-network (w1: [hidden×dim], b1: [hidden], w2: [n_actions×hidden], b2: [n_actions])
  - Target network (copie périodique soft/hard)
  - act(&obs) → action (ε-greedy)
  - train(batch) → TD loss via replay buffer
- **s01t03** — Créer `bin/bench_dqn.rs` sur RotatingT (4D, 30 seeds, 150 ep)

## Story s02: SNN baseline
- **s02t01** — Implémenter `SnnBaseline` :
  - Dual-LIF reservoir (réutiliser `DualLIF` from `neurons.rs`)
  - Readout linéaire (w_readout: [n_actions × reservoir_dim])
  - Pas d'apprentissage (SNN pur, pas de STDP)
- **s02t02** — `bin/bench_snn.rs` sur RotatingT (4D, 30 seeds, 150 ep)

## Story s03: Full benchmark suite
- **s03t01** — `bin/bench_all.rs` qui run linear/MLP/DQN/SNN/TSO et imprime tableau
- **s03t02** — Recording des résultats dans `specs/benchmarks/baseline-results.yaml`
