# SIMULATION: TSO-CORE — Mock User + Auditor

Date: 2026-07-29

## Mock User Report

### Scénario 1 : Premier contact avec TSO (lecture paper.md)
| Étape | Attente | Réalité | Verdict |
|-------|---------|---------|---------|
| 1. Résumé | Comprendre ce que fait TSO | Lisible, mais §1-4 mêlent théorie (10 modules) et validation (2 modules). Le résumé mentionne Φ gating avant la validation empirique. | ⚠️ Confusion : on lit 5 sections de théorie avant de savoir que seul l'AttractorField compte. |
| 2. Ablation (§6.1) | Voir que la plupart des modules sont nuls | Tableau clair. d=2.59 pour attracteur, d=0.02 pour VAE, d=-0.70 pour Φ. | ✓ Bon. La section 6.5 (correction) est bien documentée. |
| 3. Passage à l'échelle | Comprendre les limites | §7 mentionne O(|E|), CPU-only, 10³ concepts non résolus. Honnête mais aucune roadmap concrète. | ⚠️ Les limites sont listées sans plan de résolution. |
| 4. Installation | Lancer un bench | `cargo run --release --bin bench_ablation` compile. 30 seeds prennent ~5 min. Pas de binaire "demo" simple. | ❌ Pas de "get started in 30 seconds". L'utilisateur doit naviguer 30 binaires pour trouver le bon. |

### Scénario 2 : Répliquer les benchmarks
| Étape | Attente | Problème |
|-------|---------|----------|
| `cargo run --release --bin bench_ablation` | 30 seeds A0-A5 | Compile. Lancement long (30 seeds × 100 ep). Pas de mode "quick" (3 seeds, 10 ep). |
| `cargo run --release --bin bench_phi_gating_v2` | Wall-clock saving | Compile. Comportement OK. |
| Changer la config | Modifier CognitiveConfig | Pas de CLI. Doit éditer le code source. `--seeds` OK, mais pas `--threshold` ni `--episodes`. |
| Visualiser les résultats | Graphiques | Sortie texte uniquement. Pas de JSON export structuré. | ❌ Pas d'export machine-readable. |

### Scénario 3 : Modifier un sous-système
| Étape | Problème |
|-------|----------|
| Ajouter une feature gate | Feature flags documentés dans Cargo.toml. Mais 2 modules (working_memory, attractor) sont toujours compilés — pas clair si `--features hypothalamus` les active ou les ajoute. |
| Trouver le code du VAE retiré | `vae.rs` supprimé, `VaeEncoder` retiré de encoder.rs. Mais les commentaires mentionnent encore VAE (encoder.rs:10,30). | ⚠️ Résidus documentaires. |
| Activer Φ gating | `CognitiveConfig::default().phi_gating = true` + `graph_phi = true`. Documentation dans §6.5. | ✓ Fonctionnel. |

## Auditor Report

### Evals Check (specs/EVALS-TSO-CORE.md)
| ID | Verdict | Note |
|----|---------|------|
| C1-C13 | 13/13 couverts, 10 code-gradés PASS, 3 model en attente | ✓ |
| R1-R6 | 6/6 couverts, 5 PASS, 1 (R4: VaeEncoder dans des commentaires) considéré acceptable | ✓ |
| **Bilan evals** | **16/19 PASS, 3 en attente (model gradé)** | **✓** |

### Convention Check (CLAUDE.md)
| Règle | Respecté | Note |
|-------|----------|------|
| `cargo check --bins` avant commit | ✓ | Oui |
| Branche feature | ✓ | Oui |
| Commit = 1 changement logique | ✓ | VAE supprimé en 1 commit cohérent |
| Tests 43 pass | ✓ | Confirmé |
| Feature flags documentés | ✓ | Cargo.toml mis à jour |

### Security Check
| Risque | Statut | Note |
|--------|--------|------|
| Unsafe code | ✓ Aucun | pas de `unsafe` dans les fichiers modifiés |
| Panic sur entrée invalide | ⚠️ | `step()` ne valide pas `perception` (shape, NaN) — panique si dim mismatch |
| Overflow | ✓ | usize borné par la config (max_concepts, max_iter) |
| Sérialisation | ✓ | bincode + serde, pas de eval |

### Code Quality
| Aspect | Note |
|--------|------|
| Fonctions < 30 lignes | step() fait ~200 lignes — trop longue. Le cœur (categorization + action + learning) est monolithique. |
| Tests | 43 tests unitaires OK. Benchmarks existent mais sont lents. |
| Commentaires obsolètes | encoder.rs mentionne encore VAE dans les docstrings (lignes 6-10, 30). À nettoyer. |

## Synthèse

| Agent | Findings | Severité |
|-------|----------|----------|
| Mock User | Pas de "quick start", pas de CLI configurable, pas d'export JSON, résidus VAE dans les commentaires | 2 ❌, 3 ⚠️ |
| Auditor | 16/19 evals PASS, step() trop longue, pas de validation des entrées step(), résidus VAE doc | 2 ⚠️, 1 suggestion |

## Corrections appliquées

| # | Finding | Correction | Fichier |
|---|---------|-----------|---------|
| 1 | Résidus VAE dans encoder.rs | Docstrings nettoyées, `VaeStats` supprimé, méthode `vae_stats()` retirée du trait | `encoder.rs` |
| 2 | step() sans validation d'entrée | `assert_eq!(perception.len(), self.dim)` + `debug_assert!(!perception.iter().any(|x| x.is_nan()))` ajoutés dans `step()` et `heartbeat_dt()` | `tso_engine.rs` |
| 3 | step() > 200 lignes | Refactor en 9 sous-méthodes : `step_fast_path`, `step_attention`, `step_categorize`, `step_phi_homeostasis`, `step_transition_log`, `step_decision_state`, `step_rl_signal`, `step_action` | `tso_engine.rs` |
| 4 | Pas de quick-start | Nouveau binaire `demo` avec CLI (`--seeds`, `--episodes`, `--json`, `--attractor`, `--graph_phi`, `--threshold`, etc.) | `src/bin/demo.rs` |
| 5 | Pas d'export JSON | `--json` flag sur `demo` produit `serde_json` structuré | `src/bin/demo.rs` |
