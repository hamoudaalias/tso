# Test Design: eXX — Ablations systématiques du bien-être

## 1. Risk Matrix & Scenarios

### Risk assessment
| Risque | Niveau | Justification |
|--------|--------|--------------|
| Combinatoire | P0 | 9 termes × 3 niveaux × 10 seeds = 270 runs (~22 min) |
| Variance seed | P1 | RL non-déterministe : seed unique = bruit |
| Plafond δ-clip | P1 | Avec δ-clip, toutes les configs font 100% → pas de signal |

### Scénarios (3 phases, coût progressif)

**Phase 1 — Screening (45 runs, ~4 min)**

| ID | Description | Risque | Niveau | Runs |
|----|-------------|--------|--------|------|
| SC-P0-01 | Référence (poids=1.0), 10 seeds | P0 | E2E | 10 |
| SC-P0-02 | Chaque terme balayé {0, 0.5, 1, 2, 5}, 1 seed | P0 | E2E | 45 |
| SC-P1-03 | Top 3 termes (metabolic, parsimony, consummatory) × {0, 1, 5} × 5 seeds | P1 | E2E | 45 |

**Phase 2 — Confirmation multi-seeds (135 runs, ~11 min)**

| ID | Description | Risque | Niveau | Runs |
|----|-------------|--------|--------|------|
| SC-P1-04 | 9 termes × {0, 1} × 5 seeds | P1 | E2E | 90 |
| SC-P1-05 | Top 1 terme (metabolic) × {0, 0.5, 1, 2, 5} × 10 seeds | P1 | E2E | 50 |

**Phase 3 — Régimes homéostatiques (135 runs, ~11 min)**

| ID | Description | Risque | Niveau | Runs |
|----|-------------|--------|--------|------|
| SC-P2-06 | 5 régimes × 9 termes au poids par défaut × 3 seeds | P2 | E2E | 135 |

### Niveaux de test
- **Unit** : 0 (les termes sont testés via la formule dans step(), pas unitairement)
- **Integration** : 0
- **E2E** : toutes les runs (le comportement émergent du TSO complet est l'objet de mesure)

## 2. Fixture Architecture

- **Environnement** : GridWorld 5×5 (identique phase1b)
- **Agent** : TSO complet sans δ-clip (delta_clip_max=0.0)
- **Entraînement** : 200 eps, ε 0.8→0, noise 0.3→0
- **Test** : 50 eps ε=0
- **Métrique** : taux de succès exploitation pure

Niveau fixture : `config_utils.rs` avec `struct SensitivityConfig { weights: [f64; 9], seed: u64 }`.

## 3. NFR Verification

| NFR | Requis | Vérification |
|-----|--------|-------------|
| Temps Phase 1 | < 5 min | `time cargo run --bin sensitivity` |
| Temps Phase 2 | < 15 min | `time cargo run --bin sensitivity -- --phase2` |
| Temps Phase 3 | < 15 min | `time cargo run --bin sensitivity -- --phase3` |

## 4. Out of Scope

- Ablation avec δ-clip actif (bruit nul, plafond 100%)
- Tests unitaires des termes individuels du bien-être (formule triviale)
- Analyse de corrélation inter-termes (9×9 matrix)
