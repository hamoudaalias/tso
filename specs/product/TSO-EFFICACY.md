# TSO-EFFICACY : Résultats validés

> Ce document résume ce qui a été mesuré, sur quels benchmarks,
> et quelles sont les limites actuelles.

## 1. Résultat principal — MiniGrid DoorKey 7×7 (147D)

30 seeds, 100 épisodes, Cohen's d.

| Config | Reward | d vs Linear | d vs Attracteur |
|--------|--------|-------------|-----------------|
| A0: Linear AC | 1.03 ± 0.17 | 0.00 | — |
| **A1: TSO attracteur** | **2.11 ± 0.36** | **2.59** | **0.00** |
| A2: TSO + VAE | 2.10 ± 0.35 | 2.58 | −0.02 |
| A3: TSO full (tous modules) | 2.13 ± 0.34 | 2.60 | +0.06 |
| A4: TSO + Φ gating | 2.05 ± 0.33 | 2.50 | −0.17 |
| A5: TSO sans attracteur | 0.41 ± 0.22 | −1.77 | −2.89 |

**Conclusion :** L'AttractorField est le seul composant avec un gain
mesurable. Sans lui, TSO régresse de −0.91σ.

## 2. Φ gating — Optimisation runtime

Benchmark bench_phi_gating_v2, 5 seeds, 70 ep après 30 warm-up.

| Métrique | Passif | Φ gating (corrigé) | Δ |
|----------|--------|-------------------|---|
| Wall-clock | 172 ms | 16.2 ms | **−90%** |
| Reward | −4.41 ± 0.25 | −3.92 ± 0.38 | +0.49* |

*Non significatif (σ élevée). La dégradation d=−0.70 (v0.1) est éliminée.

**Correction v0.2 :** L'early return du Φ gating sautait l'AttractorField.
Désormais Φ ne contrôle que la résolution du graphe et les transitions.

## 3. Non-apports mesurés

| Composant | d vs Attracteur | Conclusion |
|-----------|----------------|------------|
| VAE (A2) | d=0.02 | Aucun bénéfice. Retiré du code v0.2 |
| Φ gating (A4, v0.1) | d=−0.70 | Dégradait l'AttractorField. Corrigé v0.2 |
| Hypothalamus, épisodique, attention | — | Pas de gain mesurable (pas de benchmark dédié) |

## 4. Passage à l'échelle — non résolu

| Dimension | TSO | Linear | Δ |
|-----------|-----|--------|---|
| MiniGrid 7×7 (147D) | 0.59 | 0.10 | +0.49 |
| MiniGrid 13×13 (507D) | 0.06 | −0.10 | +0.16 |
| MiniGrid 19×19 (1083D) | −0.02 | −0.10 | +0.08 |
| RotatingT 5×5 (4D) | 2.51 | 3.82 | **−1.31** |

L'avantage s'érode avec la taille de grille. TSO n'est pas compétitif
sur les faibles dimensions.

## 5. Complexité

- **AttractorField** : O(n × d) mémoire, O(n × k × d) prédiction
- **Graphe** : O(|E| × batchs × iters) résolution
- **Cerebellum** : O(d × n_actions) forward, O(d) reinforce_td
- **Limite pratique** : n ≤ 50 prototypes, |E| ≤ 10³ arêtes, CPU only
