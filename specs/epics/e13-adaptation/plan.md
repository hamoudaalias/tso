# Adaptation au but tournant — Stories

## Story s01: bench_adapt — métriques par phase
- **s01t01** — Créer `bin/bench_adapt.rs` :
  - Run 4 configs (linear, TSO attracteur, TSO full, TSO+épisodique)
  - Pour chaque config, stocker reward par épisode sur 200 épisodes (4 phases de 50)
  - Détecter les switchs (phase_count change)
  - Calculer reward moyen pour "avant switch" (ep 40-49) et "après switch" (ep 50-59)
  - Calculer convergence : épisode où reward > 80% du max des 10 derniers épisodes

## Story s02: bench_adapt — analyse multi-seed
- **s02t01** — Wrapper multi-seed : 30 seeds, moyenne ± σ par métrique
- **s02t02** — Tableau : "épisodes pour converger après switch" (moyenne ± σ)
- **s02t03** — Tableau : "reward avant vs après switch" avec Cohen's d intra-phase
- **s02t04** — Détection de transfert : le reward post-switch est-il meilleur que l'initial ?

## Story s03: Visualisation
- **s03t01** — Export CSV des courbes pour plot externe
- **s03t02** — (Optionnel) texte pour le papier
