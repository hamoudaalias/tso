# Refactor Plan: Environment trait + Minigrid integration + scaling metrics

## Problem Statement
Le moteur TSO est couplé à `grid_world.rs` (hard-coded Env5x5). Impossible de changer d'environnement (Minigrid, Procgen, Habitat) sans modifier le code du moteur. Le trait `Encoder` a déjà montré que l'architecture à trait interchangeable fonctionne. Même pattern pour l'environnement.

## Solution
1. Définir un trait `Environment` (step, reset, action_space, obs_dim).
2. Implémenter pour GridWorld (adapter `grid_world.rs` existant).
3. Ajouter un champ `env: Box<dyn Environment>` à `TsoEngine`.
4. Adapter `step()` et le binaire `main` pour boucler sur `self.env` au lieu de l'env local.
5. Mesurer le gap de scaling (dim=4→64→4096, nombre de steps).

## Commits

```
1. Définir Environment trait dans src/environment.rs
   → verify: cargo check --lib

2. Implémenter pour GridWorld (env5x5 -> TsoGridEnv)
   → verify: cargo test --test env_grid 2>&1 | grep PASS

3. Ajouter env: Option<Box<dyn Environment>> à TsoEngine
   Modifier step() pour utiliser self.env.step(action) si Some
   → verify: cargo check --bins

4. Adapter experiment_e03.rs pour utiliser l'Environment trait
   → verify: cargo run --bin experiment_e03 2>&1 | grep '100.0%'

5. Créer MinigridEnv (wrapper PyO3) dans tso_env/src/environment.rs
   Implémenter le trait Environment pour MinigridEnv
   → verify: cargo check --manifest-path tso_env/Cargo.toml

6. Benchmark : Environment trait × 3 implantations (GridWorld, Minigrid, random)
   Mesurer : µs/step, µs/reset, bandwidth (octets/step)
   → verify: cargo run --bin bench_env 2>&1 | grep -E 'GridWorld|Minigrid'
```

## Decision Document
- `Environment` trait : `step(action: usize) → (obs: Vec<f64>, reward: f64, done: bool)`
- `reset() → Vec<f64>` (observation initiale)
- `action_space() → usize`, `observation_dim() → usize`
- `Box<dyn Environment>` dans `TsoEngine` (serde skip)
- Encoder trait + Environment trait sont indépendants (pas de couplage entre eux)
- MinigridEnv utilise PyO3 (dépendance Python runtime)

## Testing Decisions
- Tests unitaires de l'Environment trait sur GridWorld (reset, step, actions valides)
- Test de regression : GridWorld via Environment doit produire les mêmes résultats qu'avant
- Les tests de scaling mesurent le throughput, pas la justesse

## Out of Scope
- Habitat integration (nécessite PyO3 + images, futur)
- Environment multi-agent
- Render (sortie visuelle, pas nécessaire pour le moteur)
