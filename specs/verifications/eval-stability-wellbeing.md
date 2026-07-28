# Eval Report: Stabilité du bien-être sur 20 seeds

## Date
2026-07-31

## Résumé
**Aucune configuration de poids ne stabilise la politique sans δ-clip.**
La variance inter-seeds est massive (σ=10-32%, P5-P95=30-90 pts).
Le δ-clip est nécessaire et suffisant.

## Résultats détaillés

| Config | μ | σ | P5 | P95 | Plage |
|--------|---|----|----|-----|-------|
| Référence (tout=1) | 32.3% | 27.0% | 10% | 100% | 90 pts |
| metabolic ×5 | 23.2% | 9.6% | 10% | 40% | 30 pts |
| parsimony ×2 | 29.0% | 24.9% | 10% | 100% | 90 pts |
| consummatory ×2 | 25.2% | 12.2% | 13% | 40% | 27 pts |
| curiosity (réf) | 35.8% | 31.5% | 10% | 100% | 90 pts |
| gated_reward ×0 | 28.0% | 17.3% | 10% | 57% | 47 pts |

## Interprétation

1. **Toutes les configs ont une variance inacceptable.** Aucune ne garantit >50% sur 20 seeds.

2. **metabolic_penalty ×5 est le plus stable** (σ=9.6%, le seul <10%) mais aussi le plus bas (23.2%).
   Il pénalise uniformément → l'agent apprend peu → variance faible, moyenne basse.

3. **curiosity est le plus performant mais instable** (μ=35.8%, σ=31.5%).
   Parfois 100% par chance de seed, parfois 10%.

4. **Aucun terme ne se rapproche de δ-clip (98.9%±0.7%).**

## Conclusion
Les 9 termes du bien-être sont des modulateurs, pas des stabilisateurs.
Le seul mécanisme qui supprime la variance inter-seeds est le δ-clip dans
le TD online. L'optimisation des poids du bien-être est un problème
secondaire — le δ-clip est le levier unique.
