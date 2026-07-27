# ADR-006 : Problème du signal de récompense non-stationnaire et δ-clip

## statut
Résolu — le δ-clip résout la régression (cause unique identifiée)

## contexte
La Phase 1 a révélé un problème fondamental : le TSO complet (avec attracteur, graphe Φ, attention, well-being à 9 termes) obtient seulement 20% en exploitation pure sur un environnement 5×5 où le Cervelet seul obtient 98%.

## analyse
La cause racine initialement suspectée (non-stationnarité du well-being à 9 termes) s'est avérée incorrecte. La régression 98% → 22% dans le cycle TSO complet était due à **l'absence de clip sur |δ| dans la mise à jour de l'acteur TD en ligne** (`step_a = lr · |δ|`). L'instabilité TD en ligne, pas la non-stationnarité du well-being, provoquait l'effondrement de la politique.

Le cycle cognitif complet (attracteur, graphe Φ, attention, curiosité, métabolisme, hypothalamus) est **compatible** avec le δ-clip et n'introduit pas d'interférence supplémentaire.

Une matrice multi-seeds (8 configurations cognitives × 10 seeds) a démontré :
- Config 0 (pas de clip) : 32.4% ± 22% (instable, haute variance)
- Config 1 (δ-clip 5.0) : 98.9% ± 0.7% (stable, variance quasi nulle)
- Configs 2-7 (tous sous-systèmes + clip) : >98% (compatibles)

La variance inter-seeds de 22% explique pourquoi certaines runs (Phase 1 #8) semblaient immunisées — le TD non-clippé fonctionne par chance de seed ou s'effondre selon l'initialisation aléatoire des poids.

## solutions tentées
| Solution | Résultat |
|----------|----------|
| Replay buffer (10 000 transitions) | ✅ Stabilise le Cervelet seul |
| Signal stationnaire (R_ext seul dans le replay) | ❌ Pas suffisant seul |
| Attention spatiale | À tester |
| **δ-clip (delta_clip_max=5.0)** | **✅ 98.9% ± 0.7% (10 seeds) — valide** |
| **CognitiveConfig (6 flags)** | **✅ Infra de bissection — tous les sous-systèmes compatibles** |

## décision
- Ajouter `CognitiveConfig` (6 flags + `delta_clip_max`) à `TsoEngine`
- Définir `delta_clip_max = 5.0` par défaut (appliqué à `Cerebellum.delta_clip` à chaque step)
- Propager le clip dans `reinforce_td` : `step_a = lr · min(|δ|, delta_clip_max)`
- La non-stationnarité du well-being n'est plus un obstacle bloquant — les expériences e03 avec well-being original + δ-clip donnent 100% d'exploitation pure

## conséquences
- + La régression 98% → 22% est résolue (cause unique : absence de δ-clip)
- + Le cycle cognitif complet est compatible avec le δ-clip (aucune dégradation mesurée)
- + Une infrastructure de bissection (`CognitiveConfig`) est disponible pour le diagnostic futur
- − Le replay buffer utilise encore `replay_lr · δ · 0.5` sans clip — chantier futur si le replay est réactivé
