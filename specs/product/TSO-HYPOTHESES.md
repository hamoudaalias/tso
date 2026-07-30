# TSO-HYPOTHESES : Extensions futures

> Ce document liste les modules théoriques de TSO qui n'ont pas été
> validés expérimentalement. Ils font partie de l'architecture B mais
> leur utilité n'est pas démontrée sur les benchmarks actuels.
> La charge de la preuve est sur chacun d'eux.

## H1 — Dual-LIF (WorkingMemory)

**Théorie.**  Mémoire de travail à deux dynamiques (rapide α_slow, lente
α_fast) qui intègre les perceptions par couplage.  Prédit une dynamique
de type integrate-and-fire avec reset.

**État.**  Implémenté dans `working_memory.rs`.  API publique :
`observe()`, `recall()`, `reset()`, `membrane_potential()`,
`spike_rate()`.  Getters ajoutés — la dynamique duale est observable.

**Validé partiellement.** 5 tests unitaires ajoutés (juillet 2026) :
- `test_pulse_response_fast_dominates` — pulse → fast > slow (vérifié)
- `test_decay_rates_differ` — après 10 pas de zéro, slow > fast (vérifié)
- `test_spike_rate_after_pulse` — spike_rate > 0, fast > slow (vérifié)
- `test_observe_and_recall` — lookup associatif correct
- `test_reset_clears_state` — reset → tout à zéro

**Restant.**  Benchmark de l'impact sur des séquences temporelles (pas
de benchmark RL temporel dans la suite actuelle).

## H2 — GridCells

**Théorie.**  Encodage de position absolue par cellules de grille,
désambiguïsant l'aliasing perceptuel dans les grilles > 6×6.

**État.**  Implémenté dans `grid_cells.rs`.  `auto_configure()`
calcule `extra_dim()` en fonction de la taille de grille.  Activé
uniquement par `configure_for_grid()`.

**Ce qu'il faudrait.**  (1) Benchmark avec aliasing contrôlé.
(2) Vérifier que extra_dim améliore la classification sur grilles
> 6×6.  (3) Quantifier la réduction d'aliasing.

## H3 — R-STDP (plasticity)

**Théorie.**  Plasticité spike-timing hebbienne : renforcement des
poids pre-post suivis d'une récompense, affaiblissement si non
récompensé.

**État.**  Implémenté dans `plasticity.rs`.  `rstdp_enabled: false`
par défaut.  API incompatible avec le reste du code (utilise
`Vec<Vec<f64>>` au lieu de `Array2`).

**Ce qu'il faudrait.**  (1) Adapter l'API à ndarray.
(2) Benchmark sur un RL simple (GridWorld 5×5) avec et sans R-STDP.
(3) Vérifier que l'apprentissage converge plus vite.

## H4 — Neurogenesis

**Théorie.**  Naissance de nouveaux concepts (prototypes), période
critique de maturation, pruning des concepts inactifs.

**État.**  Implémenté dans `neurogenesis.rs`.  `sleep_neurogenesis_rate:
0.0` par défaut.  Code non testé en benchmark.

**Ce qu'il faudrait.**  (1) Définir un taux de neurogenèse non nul.
(2) Benchmark sur MiniGrid avec suivi du nombre de concepts.
(3) Détecter le « concept drift » (concepts qui deviennent inutiles).

## H5 — FPI / EFE / Inférence active

**Théorie.**  Inférence variationnelle par FPI (Fixed-Point Iteration)
sur un modèle génératif discret.  Expected Free Energy comme politique.

**État.**  Implémenté dans `fpi.rs`, `efe.rs`, `inference.rs`,
`learning.rs`, `model.rs`.  Feature-gaté sous `active-inference`.
Nécessite pymdp pour l'interopérabilité Python.  Non utilisé dans les
benchmarks RL.

**Ce qu'il faudrait.**  (1) Benchmark autonome sur un environnement
d'inférence active (pymdp).  (2) Quantifier la différence entre la
politique RL et la politique EFE.  (3) Fusionner les deux dans un
cycle hybride.

## H6 — AssociativeMemory

**Théorie.**  Lookup associatif rapide vecteur → ID.  Infrastructure
pour WorkingMemory.

**État.**  Implémenté dans `memory.rs`.  Utilisé par WorkingMemory.
Pas une contribution cognitive en soi.

## Résumé

| Hypothèse | Code | Testé | Priorité |
|-----------|------|-------|----------|
| H1 Dual-LIF | ✓ | 5 tests unitaires PASS | P2 — manque benchmark temporel |
| H2 GridCells | ✓ | Non | P3 |
| H3 R-STDP | ✓ | Non, API incompatible | **P1** — refactor Vec→Array2 |
| H4 Neurogenesis | ✓ | Non | P2 — nécessite env. ressources |
| H5 FPI/EFE | ✓ | Non | P2 — nécessite Phase 1-3 |
| H6 AssociativeMemory | ✓ | Infrastructure | P4 |

**Priorisation 2026-07** (cf. `TSO-ORGANISM-ROADMAP.md`) :

| Priorité | Action | Dépend de |
|----------|--------|-----------|
| P0 | Environnement avec ressources périssables | Rien |
| P1 | R-STDP : refactor `Vec<Vec<f64>>` → `Array2<f64>` | Rien |
| P1 | Couplage Φ × hypothalamus (`Φ_total = Φ_graph + homeostatic_drift`) | P0 |
| P2 | Neurogenèse active (taux non nul, benchmark) | P0 |
| P2 | PerceptualBelt synchrone (fusion Dual-LIF + attention) | Rien |
| P3 | Inférence active dans le cycle step() | P1, P2 |
| P4 | Environnement continu 2D/3D | P0 |

## Références

- `specs/product/TSO-ORGANISM-ROADMAP.md` — feuille de route TSO → organisme
- `specs/product/TSO-CORE.md` — architecture complète
- `specs/product/TSO-EFFICACY.md` — résultats validés
