# BUG-2025-01-15T103000: Actor-critic TD(λ) collapse en ligne dans TSO S1

## Problem

L'entraînement du Cerebellum (actor-critic TD(λ) avec MLP hidden_dim=4) converge vers une politique gloutonne sous-optimale (19% de succès au test ε=0) quand `replay_only=false` (online TD actif) dans l'environnement TSO complet avec `use_stationary_reward=true`.

Le même Cerebellum, avec les mêmes hyperparamètres, atteint 99% de succès dans Phase 1 #8 (cervelet seul, même environnement 5×5, shaping BFS, replay). Le passage à `replay_only=true` (seul le replay entraîne) fait remonter le test à 46%.

La politique collapsée (toujours la même action, « Est ») émerge parce que le online TD(λ) amplifie de manière incontrôlée une action préférée via des δ non-bornés (jusqu'à ~10 pour une transition terminale eau) avec un learning rate élevé (0.30), tandis que `soft_normalize_row` (seuil 1.2, taux 0.01) ne suffit pas à réguler l'emballement.

- **Comportement observé** : Test ε=0 → 19% de succès, politique toujours la même action pour toutes les positions.
- **Comportement attendu** : >90% de succès, politique diverse (plusieurs actions selon la position).
- **Reproduction** : `cargo run --bin phase1c` (config S1).

## Root Cause Analysis

La cause racine est une **instabilité de l'actor-critic TD(λ) avec fonction approximée (MLP) en régime de double mise à jour (online + replay)**. Spécifiquement :

1. **Le online TD utilise `|δ|` comme magnitude d'update** : `step_a = lr * |δ|`. Un δ de ~10 (transition eau) applique un `step_a` de ~3.0 aux poids de la ligne d'action — soit 100× plus qu'un pas normal (δ ~0.3). Cela pousse la norme de `w2[action_choisie]` très au-dessus des autres.

2. **`soft_normalize_row` (seuil 1.2, taux 0.01) est trop doux** pour ce régime : à norme=50, l'échelle est ~0.67, ce qui n'empêche pas l'explosion.

3. **Le replay `replay_train` utilise `replay_lr * δ * 0.5`** (sans le `|δ|` de l'online) → les mises à jour replay sont ~60× plus petites que les mises à jour online terminales. Le replay ne peut pas corriger le biais introduit par le online.

4. **L'ordre des appels n'est pas en cause** — l'audit montre que `forward_logits` → `reinforce_td` → `decay_trace` → `mark` produit le même δ que Phase 1 #8 (qui fait `forward` → `mark` → `reinforce_td` → `decay`). La seule différence est que `mark` est appelé après `reinforce_td` dans TSO, ce qui fait que `v_prev` du premier step de chaque épisode est 0 → δ spurié de `0.99·V(s₀)`. Mais les traces sont initialement nulles, donc cet δ n'a aucun effet.

5. **Ce n'est pas un problème du TSO cognitif** — le même collapse se produit potentiellement dans Phase 1 #8 avec une autre seed aléatoire. La preuve : `replay_only=true` (qui désactive le online TD) fait monter le test de 19% à 46%, même avec tous les modules TSO actifs.

La **différence entre Phase 1 #8 (99%) et S1 (19%)** s'explique par le conditionnement initial : les poids aléatoires initiaux favorisent une action différente. Phase 1 #8 a eu la chance que l'action favorisée au départ soit une bonne action multi-position. S1 a favorisé East, qui est catastrophique (mène systématiquement au mur à droite).

## TDD Fix Plan

1. **RED**: Write a test that trains a Cerebellum (dim=6, hd=4) with online TD + replay for 500 episodes on 5×5 grid (BFS shaping, stationnaire) and verifies test ε=0 > 50%. Run 3 seeds; at least 2 must pass.
   **GREEN**: Reduce actor learning rate `lr` from 0.30 to 0.10 and set `replay_only=true` (désactive online TD, l'apprentissage se fait uniquement via replay).
   **verify**: `cargo test --test cerebellum_stability`

2. **RED**: Test that `replay_only=false` (online TD actif) with lr=0.10 still converges >50%.
   **GREEN**: Pas de changement ; le lr réduit stabilise l'online TD.
   **verify**: `cargo test --test cerebellum_stability -- replay_online`

3. **RED**: Test that `soft_normalize_row` empêche toute norme `w2[a]` de dépasser 5.0.
   **GREEN**: Augmenter le seuil de `soft_normalize_row` à 2.0 et le taux à 0.05.
   **verify**: `cargo test --test cerebellum_weight_regulation`

**REFACTOR**: After tests pass, document la recommandation de configuration dans le code de phase1c.rs : `replay_only=true` est plus robuste que l'apprentissage en ligne pour ce setup.

## Acceptance Criteria

- [ ] S1 avec `replay_only=true` atteint ≥80% au test ε=0 sur 3 seeds différentes
- [ ] Les poids `w2` restent bornés (norme < 5) après 500 épisodes
- [ ] Tous les tests existants passent
- [ ] La politique apprise est diverse (au moins 3 actions différentes utilisées dans la grille)

## Résolution

<!-- filled in by validate-fix -->

## Security impact

NONE — recherche en simulation, aucun impact sur sécurité ou données.
