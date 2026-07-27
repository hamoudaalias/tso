# ADR-005 : Résolution de contraintes par recuit simulé

## statut
Accepté

## contexte
Le graphe sémantique accumule des contradictions (arêtes violées) qui produisent Φ. L'agent a besoin d'un mécanisme pour résoudre ces conflits en modifiant les vecteurs nœuds. Les modifications atomiques possibles sont :
- Inverser un vecteur (v → −v)
- Aligner deux vecteurs (les fusionner vers la moyenne)
- Repousser deux vecteurs (les séparer sur la sphère)

## décision
Utiliser le recuit simulé avec :
- Température initiale T₀ = 0.2 (heartbeat) / 0.3 (sleep)
- Refroidissement ×0.85 par itération
- Boltzmann selection des actions (poids exp(−ΔΦ/T))
- Détection d'oscillation : si Φ alterne de direction ≥3 fois en 6 itérations → mode greedy
- Actor RL local pour apprendre quelle action fonctionne pour chaque type de conflit

## parallélisation
Ajout de `resolve_parallel` pour distribuer les batchs d'arêtes indépendantes sur N threads.

## conséquences
- + Recuit simulé évite les minima locaux
- + Détection d'oscillation brise les cycles stériles Repel↔Align
- + Parallélisation essentielle pour |E| > 500
- − 15 itérations par heartbeat peuvent être insuffisantes pour les graphes très conflictuels
- − Le sommeil (80 itérations) est nécessaire pour une résolution en profondeur
