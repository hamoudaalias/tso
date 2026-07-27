# BUG-2025-08-03T120000: Cycle cognitif TSO interfère avec l'apprentissage du Cerebellum en 5×5

## Problem

Le TSO complet (TsoEngine::step) obtient ~25% de succès au test ε=0 sur l'environnement 5×5,
quel que soit le signal RL utilisé (well-being original ou reward stationnaire),
même avec l'hypothalamus gelé. Le même Cerebellum, entraîné sans le cycle cognitif
(Phase 1 #8), atteignait 98%.

- **Comportement observé** : Toutes les configs e03 donnent 20-32% sur 5×5.
  L'entraînement voit 100% de succès (ε-greedy), mais le test ε=0 s'effondre.
- **Comportement attendu** : Le correctif e03 (reward stationnaire) devait restaurer ~90%.
- **Reproduction** : `cargo run --bin experiment_e03`
- **Régression** : `cargo run --bin phase1b_fix3` donnait 98% avant le commit `abd3bf0`,
  donne maintenant 22% après refactoring.

## Root Cause Analysis

Le refactoring (commit `abd3bf0`) a intégré le cycle cognitif complet dans `TsoEngine::step()` :
attention spatiale, catégorisation des concepts, mémoire épisodique, résolution de graphe Φ,
curiosité, coût métabolique, sommeil. Chacun de ces sous-systèmes modifie le signal
d'entraînement du Cerebellum via des mécanismes parasites :

1. **Attention gating** : Le `gated` passé au Cerebellum (quand `use_stationary_reward=false`)
   est une version filtrée de la perception, pas la perception brute. Cela peut supprimer
   des dimensions pertinentes pour la décision.

2. **Création de concepts** : Chaque perception inédite crée un nouveau concept prototype.
   Cela dilue l'espace d'état et augmente la complexité du graphe Φ.

3. **Curiosité intrinsèque** : Ajoutée au well-being même en test ε=0
   (modifié par `total_reward`). Modifie la distribution du signal RL.

4. **Coût métabolique** : `metabolic_penalty = -total_cost * 20.0` biaise négativement
   le well-being de ~ -0.02/step, cumulé à ~ -3.0/épisode.

5. **Résolution de graphe Φ** : Tous les 50 steps, `resolve_with_anneal` modifie le graphe,
   ce qui change `phi_delta` et `chronic_tension` dans le well-being.

L'effet cumulé est que le Cerebellum reçoit un signal RL non-stationnaire même avec
`use_stationary_reward=true`, parce que l'état de décision (perception brute ou gated)
passe par un espace d'état conceptuel qui ne correspond pas à l'espace perceptuel simple
attendu par le Cerebellum. La politique apprise pendant l'entraînement ε-greedy ne
généralise pas à ε=0 car les trajectoires conceptuelles diffèrent.

- **Modules concernés** : TsoEngine::step(), l'ensemble du cycle cognitif
- **Risque** : Medium — la regression bloque l'epic e03 et masque l'impact réel des correctifs
- **Sécurité** : Aucun impact de sécurité identifié

## TDD Fix Plan

La solution est de permettre un mode "lean" où le cycle cognitif est minimal pendant
l'entraînement du Cerebellum, puis réactivé progressivement. Mais la solution la plus
immédiate pour valider e03 est de restaurer un mode Cerebellum-only (sans cycle cognitif)
dans le binaire d'expérience.

1. **RED**: `cargo test --test cerebellum_lean -- --nocapture 2>&1 | grep -E 'test.*PASS'`
   Teste que `TsoEngine` avec un flag `cognitive_cycle: false` désactive les sous-systèmes
   non-essentiels et obtient >80% au test ε=0 sur 5×5.
   **GREEN**: Ajouter un flag `cognitive_cycle: bool` à `TsoEngine` qui, quand `false`,
   ne désactive que les traitements redondants : pas d'attention, pas de catégorisation,
   pas de graphe, pas de curiosité, pas de métabolisme. Le Cerebellum est entraîné
   directement sur la perception brute avec `reward + bfs_shaping` (stationnaire).
   **verify**: `cargo test --test cerebellum_lean`

2. **RED**: Test qui vérifie que le mode cognitif complet redonne le même résultat qu'avant
   (régression test).
   **GREEN**: Le flag `cognitive_cycle` par défaut à `true` — pas de changement de
   comportement pour les autres binaires.
   **verify**: `cargo test --test cognitive_cycle_regression`

3. **RED**: Test qui valide que e03s04 en mode lean donne >80% sur 3 seeds différentes.
   **GREEN**: Aucun changement — les correctifs e03 (normalisation, dual critic, reward
   stationnaire) fonctionnent mais étaient masqués par le cycle cognitif.
   **verify**: `cargo test --test e03s04_comparison`

**REFACTOR**: Documenter les flags de mode dans la doc de `TsoEngine`. Créer un struct
`CognitiveConfig` pour contrôler finement les sous-systèmes activés.

## Acceptance Criteria

- [ ] Mode lean (cognitive_cycle=false) atteint >80% exploitation ε=0 sur 5×5
- [ ] Mode complet (cognitive_cycle=true) conserve le comportement actuel
- [ ] Tous les binaires existants compilent et tournent sans changement
- [ ] e03s04 valide 3 seeds avec >80%

## Resolution

<!-- filled in by validate-fix -->
