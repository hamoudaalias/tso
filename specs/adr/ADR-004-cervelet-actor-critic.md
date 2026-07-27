# ADR-004 : Actor-Critic avec TD(λ) et replay buffer (Cervelet)

## statut
Accepté (modifié depuis Phase 1)

## contexte
L'agent doit apprendre à sélectionner des actions pour maximiser son bien-être à long terme. Le problème est un POMDP avec un signal de récompense composite (9 termes). L'apprentissage doit être stable et capable d'exploitation pure (ε=0).

## décision initiale (Phase 1)
Cervelet Actor-Critic MLP avec TD(λ), eligibility traces, REINFORCE, learning rates asymétriques.

## modification post-Phase 1
Ajout d'un **replay buffer** (capacité 10 000) avec mini-batch TD pour stabiliser l'apprentissage, suite au constat que le cycle cognitif complet produisait un signal de récompense non-stationnaire (20% vs 98% en exploitation pure).

## conséquences
- + Le Cervelet seul atteint 98% sur 5×5, 66.5% sur 7×7
- + Le replay buffer stabilise l'apprentissage (100% exploitation pure atteignable)
- − Le MLP coûte 2× plus cher métaboliquement que le linéaire
- − Le well-being non-stationnaire reste un problème ouvert pour le TSO complet
- − L'apprentissage asymétrique (positif 5× plus vite que négatif) peut biaiser la politique
