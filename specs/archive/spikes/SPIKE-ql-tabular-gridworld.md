# Spike: Q-learning tabulaire sur GridWorld

## Question
Un Q-learning tabulaire (sans graphe, sans attracteur, sans curiosité, sans sommeil)
atteint-il le même taux de succès que TSO sur GridWorld 5×5 ?

## Result
Oui — 100% sur 10 seeds, σ=0.0%. Mais l'état est la **position absolue (x,y)**,
ce qui donne une observabilité parfaite que TSO n'a pas (4 moustaches seulement).

## Findings
1. Q-learning tabulaire avec table 25×4 : 100% en 200 épisodes, convergence instantanée.
2. Sur GridWorld 5×5 empty room, le problème est trop simple pour discriminer.
3. L'avantage informatif de la position (25 états discrets vs 4 whiskers continus)
   est écrasant sur cette grille.
4. Temps d'exécution : 4.3ms pour 10 seeds (vs ~1s pour TSO complet).

## Evidence
```
Résultat : μ = 100.0%, σ = 0.0% [4.3ms]
TSO pur (curiosity=0) : 100% (benchmark précédent)
Q-learning tabulaire   : 100.0%
```

## Implications for the plan
- GridWorld 5×5 empty room est **inutile comme benchmark** — les deux approches
  plafonnent. Il faut des environnements avec aliasing perceptuel (Terrarium 7×7)
  ou récompenses rares pour voir une différence.
- Pour une baseline honnête, le Q-learning devrait utiliser les mêmes observations
  que TSO (4 whiskers) au lieu de la position. Mais alors l'état n'est plus tabulaire
  (espace continu). Solution : Q-learning with function approximation (tilings) ou
  discretiser les whiskers en buckets.

## What was NOT explored
- Terrarium 7×7 (récompenses rares, aliasing)
- Q-learning avec discrétisation des whiskers (même espace d'observation que TSO)
- Nombre d'épisodes plus faible (50 suffisent probablement)

## Recommendation
Ne pas déployer le Q-learning tabulaire comme baseline sur 5×5 — plafonné.
Le Terrarium 7×7 ou un GridWorld avec aliasing (random walls) sera plus discriminant.
Le code est jetable (supprimé).
