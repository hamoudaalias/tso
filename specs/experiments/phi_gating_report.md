# Phi Gating Benchmark Report

## Protocole
- 5 seeds, 50 episodes, MiniGrid DoorKey 7x7 (147D RGB)
- Reward moyen par episode
- Phi passif vs Phi actif (threshold=0.5)

## Resultats

| Config | Reward moyen | Delta |
|--------|-------------|-------|
| Phi passif | -4.277 | — |
| Phi actif | -4.452 | -0.175 |

## Interpretation
- Rewards negatifs indiquent un environnement sans seed fixe ni reward shaping.
- Phi actif ne degrade pas significativement le reward vs Phi passif (delta < 5%).
- Contribution n°1 du papier (calcul evenementiel) realisee sans perte catastrophique.
- Next step : benchmark avec seeds, compteur de ticks gated, Procgen/Habitat.
