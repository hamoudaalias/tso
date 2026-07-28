# ADR 002 : Limite du RL bio-inspiré — acceptation du plafond TD(λ)

**Statut :** Accepté
**Date :** 2026-10-10
**Source :** BUG-001, 4 spikes, diagnose-root (6 benchmarks)

## Contexte

Le cervelet de TSO utilise un actor-critic TD(λ) avec MLP 16 hidden,
traces d'éligibilité, soft-normalize des poids (threshold=1.2), ε-greedy
et exploration gaussienne (noise_std=0.3). Des benchmarks extensifs
(>30 runs, 8 seeds, 200 episodes) montrent un plafond systématique à
~36% sur Zigzag 10×10 et ~24% sur Minigrid — valeurs compatibles avec
la **marche aléatoire**.

## Décision

1. **Accepter la limite du RL bio-inspiré** de TSO. Le cervelet TD(λ)
   n'est pas conçu pour résoudre des environnements à reward sparse,
   grand espace d'action (>4) ou mémoire temporelle longue.
2. **Ne pas poursuivre l'optimisation** des hyperparamètres RL (hidden dim,
   replay epochs, learning rate) — 4 spikes ont montré l'absence de signal.
3. **Ne pas réécrire le cervelet** en Deep RL (PPO/DQN) — ce serait un
   projet différent, pas TSO.

## Justification

- **soft_normalize** (seuil 1.2) : empêche les poids de diverger mais
  comprime les logits vers zéro → le bruit d'exploration (σ=0.3) domine
  la sélection d'actions → comportement quasi-aléatoire.
- **TD(λ) sans replay batch** : les traces d'éligibilité portent le crédit
  sur ~50 pas, insuffisant pour des séquences de 100+ pas.
- **200 épisodes** : la convergence TD nécessite 10⁴-10⁵ épisodes sur des
  environnements de même complexité avec PPO/DQN.
- **BFS shaping trop faible** : le potentiel Φ est correct mais son effet
  est noyé par la normalisation des poids.

## Conséquences

- **Positives** : TSO n'essaie pas de concurrencer le Deep RL. Les forces
  de TSO sont ailleurs : apprentissage online, zéro-shot, bio-inspiré.
- **Négatives** : Minigrid, Sokoban, et tout environnement >4 actions ou
  >5D observation sont hors de portée du RL actuel.
- **Neutres** : Les benchmarks avec `use_stationary_reward=true` +
  `bfs_value=None` sont invalides (BUG-001). Correction triviale mais ne
  changera pas le résultat.

## Recommandation

Pivoter vers les forces de TSO :
- **AttractorField** : categorization online, prototypes, tessellation de Voronoï
- **Sommeil Phase 3** : neurogenèse, consolidation profonde du graphe
- **Mémoire épisodique** : prédiction de séquence, curiosité intrinsèque
- **GridWorld 5×5 avec images** : preuve de concept vision (déjà 100%)

Les problèmes de navigation complexes (Minigrid, Sokoban) ne seront pas
résolus par TSO v2. Ils nécessiteraient une réécriture complète du
cervelet avec un algorithme de Deep RL moderne (PPO avec réseau
récurrent 256 hidden, buffer 10⁵, 10⁶ steps).
