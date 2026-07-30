# Adaptation v2 — Stories

## Story s01: Environment adaptatif local (bench_adapt_v2)
- **s01t01** — Créer un environnement local (pas rotating_t.rs) :
  - switch=10, max_steps=10, 8 goals, 4 aliasing pairs
  - reset() et step() compatibles Array1<f64>
- **s01t02** — Courbe de reward par épisode (200 épisodes)

## Story s02: Métriques d'adaptation
- **s02t01** — Reward moyen par phase de 10 épisodes
- **s02t02** — Pente intra-phase : régression linéaire sur les 10 épisodes
- **s02t03** — Drop au switch : reward(1er épisode après switch)
- **s02t04** — Forgetting : différence entre 1ère et 2ème occurrence du même but

## Story s03: Run + analyse
- **s03t01** — 30 seeds × 4 configs
- **s03t02** — Tableaux par métrique
