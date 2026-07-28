# BUG-001 : Le cervelet n'apprend pas — rl_signal nul par défaut

**Statut :** Confirmé
**Date :** 2026-10-10
**Source :** diagnose-root cross benchmark (bench_rl_scaling, bench_cognitive_modules, diag_rl_ceiling, diag_bfs_vs_none)

## Reproduction

Tout benchmark qui appelle `engine.step(&obs, 0.0, None, &[])` avec `use_stationary_reward=true` :
- bench_zigzag.rs
- bench_vae_joint.rs
- bench_rl_scaling.rs
- bench_cognitive_modules.rs
- spike_count_curiosity.rs

**Steps :**
1. Lancer `cargo run --release --bin diag_rl_ceiling`
2. Observer `|δ|_avg=0.000000` et les poids du cervelet qui ne bougent pas
3. Résultat : succès ~36-38% = marche aléatoire (4 actions, 10×10)

## Cause racine

`tso_engine.rs:step()` ligne ~675-685 :

```rust
let rl_signal = if self.use_stationary_reward {
    let bfs_shaping = match (self.prev_bfs_value, bfs_value) {
        (Some(prev_bv), Some(cur_bv)) => 0.99 * cur_bv - prev_bv,
        _ => 0.0,
    };
    self.prev_bfs_value = bfs_value;
    reward + bfs_shaping
} else { total_reward };
```

Quand `bfs_value=None` (le cas de tous les benchmarks sauf 1), le shaping vaut 0, `reward=0.0` (pas de reward externe), donc `rl_signal=0.0`. Le `reinforce_td(0.0, 0.99)` fait return immédiatement.

**Les 8 benchmarks précédents (e01 à e08 + les 4 spikes d'aujourd'hui) mesuraient tous la marche aléatoire, pas l'apprentissage.**

## Impact

- Tous les benchmarks "attention", "VAE conjoint", "RL scaling", "modules cognitifs" sont invalides — ils comparent du bruit aléatoire.
- Les résultats du rapport Minigrid (12-24%) sont aussi de la marche aléatoire sur 147D avec 7 actions → taux de base ~14%.
- Les 100% sur GridWorld 5×5 viennent du shaping BFS potentiel (via hydra/bfs_value passé correctement) — ce benchmark est le seul valide.

## Correction nécessaire

1. Passer `bfs_value` dans les benchmarks qui veulent tester l'apprentissage
2. Ou désactiver `use_stationary_reward` (utiliser `total_reward` qui intègre tous les signaux)
3. Ou modifier `step()` pour que le signal RL par défaut ne soit pas 0

## Recommandation

La correction est triviale (passer la bonne valeur bfs ou désactiver stationary_reward),
mais la conséquence est que **tous les résultats précédents sont à rejeter.**
