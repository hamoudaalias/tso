# Plan Tests — Significativité statistique des variantes TSO

## Risk Matrix

| Variante | Risque | Justification |
|----------|--------|---------------|
| TSO attracteur vs linear AC | P0 | Claim principal du papier |
| TSO VAE vs TSO attracteur | P1 | VAE sensé améliorer |
| TSO FPI vs TSO attracteur | P1 | FPI sensé améliorer |
| Φ gating ON vs OFF | P1 | Mécanisme signature, jamais isolé |
| TSO vs DQN | P0 | Baseline forte, pas surpasse |
| Effet de la taille (7×7 vs 13×13 vs 19×19) | P1 | Plafond de navigation |

## Stratégie statistique

| Métrique | Seuil | Test |
|----------|-------|------|
| Cohen's d | > 0.2 (small), > 0.5 (medium), > 0.8 (large) | Effect size |
| Welch t-test | p < 0.05 | Inégalité des variances (toujours le cas en RL) |
| IC 95% | Non chevauchement | Bootstrap ou normal |
| Pass@k | k=30 seeds, pass = IC ne chevauche pas zéro | Eval-driven |

## Plan d'implémentation

1. **MultiSeedRunner** struct qui encapsule :
   - N seeds (défaut 30, configurable)
   - Moyenne, σ, IC 95%, Cohen's d vs baseline
   - Export JSON/YAML
2. **Upgrade multi_seed_bisect** :
   - Ajouter configurations MiniGrid 147D (pas seulement RotatingT 4D)
   - Ajouter DQN/MLP/Linear comme baselines
   - Output format tabulaire + JSON
3. **Benchmark unifié** :
   - `cargo run --release --bin bench_all` qui run tout et imprime tableau
   - Résultats dans `specs/benchmarks/`

## Scénarios

| ID | Scénario | N seeds | Test |
|----|----------|---------|------|
| ST-01 | TSO attracteur vs linear AC sur MiniGrid 7×7 | 30 | Welch, Cohen's d |
| ST-02 | TSO VAE vs linear AC sur MiniGrid 7×7 | 30 | Welch, Cohen's d |
| ST-03 | TSO FPI vs TSO attracteur | 30 | Welch apparié |
| ST-04 | Φ gating ON vs OFF | 30 | Welch |
| ST-05 | TSO vs DQN (147D, 64 hidden) | 30 | Welch |
| ST-06 | TSO attracteur 7×7 vs 13×13 vs 19×19 | 10×3 | ANOVA à 1 facteur |
