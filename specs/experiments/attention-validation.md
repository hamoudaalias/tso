# Attention spatiale — rapport de validation

## Résumé

L'attention spatiale (gain multiplicatif par erreur de prédiction épisodique)
a été évaluée sur 3 benchmarks. **Conclusion : l'attention ne dégrade pas,
mais n'apporte pas de gain significatif sur les environnements actuels.**

## Benchmarks

| Environnement | Sans attention | Avec attention | Gain | Interprétation |
|---------------|---------------|----------------|------|----------------|
| Empty room 5×5 (10 seeds) | 98.7% ± 0.8 | 98.9% ± 0.8 | +0.2% | Plafonné — trop facile |
| Terrarium 7×7 (3 seeds) | 98.5% ± 0.4 | 98.7% ± 0.5 | +0.2% | Plafonné — replay + δ-clip suffisent |
| Zigzag 10×10 (10 seeds) | 36.5% ± 3.0 | 36.8% ± 2.5 | +0.3% | ↔ Pas de différence significative |

## Analyse

Le gain attentionnel est nul sur tous les environnements testés. Causes
probables :

1. **Aliasing non résolu par l'attention seule** — l'attention amplifie des
   dimensions de moustaches, mais si deux positions ont des moustaches
   identiques (aliasing parfait), aucune amplification ne peut les distinguer.
2. **δ-clip + replay buffer dominent** — les correctifs e03 stabilisent
   tellement l'apprentissage que l'attention n'a plus de marge d'amélioration.
3. **Cervelet linéaire** — l'attention produit un proto-concept modulé, mais
   le cervelet linéaire ne peut pas exploiter cette modulation subtile.

## Conclusion

L'epic e04 est validée : l'attention spatiale est fonctionnelle, ne dégrade
pas, et n'interfère avec aucun sous-système. Son gain potentiel ne pourra
être mesuré que sur des environnements où :
- L'aliasing est résolu à 50% par l'attention (eg. vision partielle bruitée)
- Le cervelet est assez profond pour exploiter le gating attentionnel
