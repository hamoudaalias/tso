# Spike: Baseline MLP/DQN/SNN pour MiniGrid

## Question
Les baselines standard (MLP, DQN, SNN) battent-elles TSO sur MiniGrid DoorKey 7×7 ?

## Résultat
**Oui — fortement.** Sur RotatingT 5×5 : Linear AC (3.05) bat TSO-full (1.96) de −36%.
MLP (64 hidden, 2.66) bat aussi TSO. Le +105% du papier était un artefact de
benchmark cassé (dim=5 vs obs=4).

## Ce qui existe déjà
- **MLP AC** : Cerebellum avec hidden_dim=64 (non-linear mode). Fonctionnel.
- **Linear AC** : Cerebellum avec hidden_dim=0. Fonctionnel.
- **Replay buffer** : `tso-engine/src/replay_buffer.rs`. Existe.
- **MiniGrid env** : `tso-engine/src/minigrid_env.rs`. Rust native, 147D RGB.

## Ce qu'il faut construire (minimal)
1. **DQN** : Cerebellum-like Q-network (w1/w2 output Q-values, target network frozen,
   replay buffer, TD loss minibatch). Pas de actor-critic, pas de policy gradient.
   2 files: `dqn.rs` (struct + train) + `bin/bench_dqn.rs` (benchmark).
2. **SNN** : Dual-LIF reservoir + readout layer. Spike-based, pas de backprop.
   Réutiliser `neurons.rs` (LIF clusters). 1 file: `snn_baseline.rs`.

## Non construit (YAGNI)
- **CNN** : nécessite convolution ndarray → complexe pour 7×7 RGB (147D = 3×7×7, CNN
  serait mieux mais baseline injuste — TSO n'a pas non plus de conv).
- **PPO** : nécessite plusieurs rollouts synchronisés, complexité inutile.

## Architecture proposée
```
src/
  baselines/
    mod.rs (pub use)
    dqn.rs       # DqnAgent: Q-network + target + replay
    snn.rs       # SnnBaseline: Dual-LIF reservoir + linear readout
  bin/
    bench_dqn.rs
    bench_snn.rs
```

## Plan
1. DQN spike → test sur RotatingT (4D)
2. SNN spike → test sur RotatingT (4D)
3. Benchmarks : 30 seeds, 150 episodes, comparer linear/MLP/DQN/SNN/TSO
