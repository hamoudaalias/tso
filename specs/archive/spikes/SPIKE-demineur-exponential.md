# Spike: Démineur sans suppression — décroissance exponentielle (×0.95/tick)

## Question
Le Weakness Game produit-il un PROOF SCORE ≥ 90 sans suppression brutale
d'arêtes (flag_edge) ? La décroissance exponentielle du poids (×0.95/tick)
suffit-elle à résoudre les conflits ?

## Résultat
**Oui. PROOF SCORE = 100.0 pour les deux stratégies.** La décroissance
exponentielle converge aussi bien que la suppression instantanée.

## Comparaison

| Métrique | flag_edge (instantané) | exponential ×0.95/tick |
|----------|----------------------|------------------------|
| PROOF SCORE | 100.0 | 100.0 |
| Φ final | 0.0000 | 0.0000 |
| Arêtes restantes | 0 | 0 |
| Temps (5 seeds) | ~2.8 µs | ~1.7 µs |
| Suppression | ✅ Instantanée | ✅ Quand |weight|<0.5 |
| Biologiquement plausible | ❌ | ✅ (décroissance continue) |

## Analyse
La décroissance exponentielle ×0.95 multiplie le poids de l'arête la plus
violée à chaque tick. Un poids de ±1 passe sous 0.5 en 14 ticks
(1 × 0.95^14 ≈ 0.49), un poids ±2 en 27 ticks. Le temps de convergence
total est linéaire en `max(|weight|) / (1 - factor)`, ici ~30 ticks max.
Le PROOF SCORE reste parfait (100.0) car le Φ éliminé est le même —
seule la vitesse change.

## Implications
- flag_edge peut être remplacé par exponential_decay_sweep sans perte
  de score de preuve.
- La version exponentielle est biologiquement plus plausible (LTD).
- La vitesse est acceptable (< 1 ms pour 200 arêtes).
- Pour la production, utiliser `exp_decay` en mode normal,
  `flag_edge` en mode debug/urgence.

## Code
- `exp_decay_edge_weight(from, to, factor)` sur Graph
- `exponential_decay_sweep(graph, tol, factor)` (module core)
- `TsoEngine::exponential_decay_sweep(tol, factor)`
