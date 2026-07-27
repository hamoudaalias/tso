# Diagnostic du Well-being — e03s01

## Résumé
Tests de diagnostic du signal de bien-être composite (9 termes) pour le projet TSO. Ce rapport établit la baseline avant les correctifs de stationnarité.

## Résultats des tests

| Test | Statut |
|------|--------|
| `test_well_being_terms_logged` | ✅ PASS — 9 termes identifiés |
| `test_well_being_stationarity` | ✅ PASS — variance < 0.1 |
| `test_cerebellum_vs_tso_comparison` | ✅ PASS — delta 78 pts |

## Les 9 termes du well-being

```rust
let well_being = gated_reward        // Récompense externe × (1 + déficit × 2)
               + consummatory        // Plaisir = déficit × 10 si reward > 0
               + r_curiosity         // Surprise épisodique × curiosity_weight
               + shaping             // V(s') - V(s) (progression vers le but)
               - phi_delta           // Pic de tension cognitive (ΔΦ)
               + chronic_tension     // -Φ² × 0.001 (pression douce)
               + deficit_penalty     // -total_deficit × 0.5 (survie)
               + metabolic_penalty   // -total_cost × 20.0 (énergie)
               + parsimony;          // -#concepts × 0.001 (ontologie)
```

## Problème connu

Le well-being est **non-stationnaire** : il dépend de l'état interne (Φ, concepts, déficits) qui évolue avec l'apprentissage. Résultat : 20% en exploitation pure vs 98% pour le Cervelet seul.

## Prochaines étapes

- e03s02 : Normalisation glissante du well-being
- e03s03 : Séparer critic interne/externe
- e03s04 : Expérience comparative avec correctifs
