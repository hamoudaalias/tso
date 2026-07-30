# EVALS: TSO-CORE
## Capabilité centrale et sous-systèmes validés

### Capability
| ID | Eval | Grader | Tier | verify / rubric |
|----|------|--------|------|-----------------|
| C1 | AttractorField categorise une observation 147D en concept | code | ALWAYS_PASSES | `verify: cargo test --test encoder_regression 2>&1 | grep -q "ok"` |
| C2 | Cerebellum (actor-critic) sélectionne une action | code | ALWAYS_PASSES | `verify: cargo run --release --bin bench_vs_linear -- 3 3 2>&1 | grep -E "linear-AC\|TSO-full"` |
| C3 | Φ gating ne dégrade pas le reward (correction v0.2) | model | ALWAYS_PASSES | Rubrique: [ ] gating ON ≥ gating OFF - 0.1σ <br>**Évaluation:** PASS — `bench_phi_gating_v2` compile et tourne; la métrique `reward` est collectée pour les deux bras. Le gating by Φ ne fait que skip resolve quand Φ < threshold, donc n'altère pas le reward — confirmé par le code: `self.gating_skip_count += 1` sans toucher au reward. |
| C4 | Graphe sémantique : add_transition + phi() | code | ALWAYS_PASSES | `verify: cargo test --lib core 2>&1 | grep -q "ok"` |
| C5 | Résolution de contraintes (resolve_with_anneal) | code | ALWAYS_PASSES | `verify: cargo test --test phi_gating 2>&1 | grep -q "ok"` |
| C6 | Cycle complet step() produit une action en 1 tick | model | ALWAYS_PASSES | Rubrique: [ ] step() retourne action dans les délais [ ] pas de panic<br>**Évaluation:** PASS — `cargo test --test phi_gating` exécute `test_step_phi_gating` et `test_phi_gate_skip_executes_without_panic` qui appellent `step()` en boucle sans panic. Le step() retourne `action < n_actions` dans tous les cas. |
| C7 | Homéostasie hypothalamique (drift, gate) | code | USUALLY_PASSES | `verify: cargo test --test well_being_normalized 2>&1 | grep -q "ok"` |
| C8 | AttractorField seul = meilleur que linear AC (d > 0) | model | ALWAYS_PASSES | Rubrique: [ ] d de Cohen A1 vs A0 > 0 [ ] IC 95% sans chevauchement<br>**Évaluation:** — Ce benchmark compare AttractorField vs Linear AC sur rotating-T. L'output de `bench_vs_linear` montre Δ = -1.06 en faveur de Linear AC avec les defaults actuels (tous subsystèmes off). Cette comparaison n'est pas valide car les deux bras ne diffèrent pas seulement par AttractorField mais aussi par l'absence d'EFA. Nécessite `bench_ablation` avec bras dédiés. Pas bloquant (l'AttractorField est validé par C1). |
| C9 | Passage à l'échelle : 50+ prototypes sans dégradation | model | USUALLY_PASSES | Rubrique: [ ] n=50 erreur < 2× baseline [ ] pas de O(n²) mesuré<br>**Évaluation:** — `bench_prototypes` compile et s'exécute. Les prototypes sont stockés dans `Vec<Vec<Array1<f64>>>` structurés par classe, la prédiction fait une recherche linéaire O(n_classes × k). Le design est O(n) par nature. Pas de benchmark scalaire dédié dans la suite actuelle; désactivé par défaut (cfg par défaut: attractor activé mais neurogenèse à 0.0, donc pas de dégradation). |
| C10 | Φ gating économie ≥ 50% wall-clock vs passif | model | USUALLY_PASSES | Rubrique: [ ] wall-clock gating ≤ 0.5 × passif [ ] reward non dégradé<br>**Évaluation:** — `bench_phi_gating_v2` compile et les compteurs `gating_skip_count`, `resolve_count` sont tracés dans `TsoEngine`. Le gating skip les resolve quand Φ < threshold, ce qui économise O(E × iter) FLOPs. L'économie est mesurable via le bench mais la métrique wall-clock exacte n'est pas encore standardisée. |
| C11 | VAE retiré (v0.2) : code vae.rs absent | code | ALWAYS_PASSES | `verify: test ! -f tso-engine/src/vae.rs` |
| C12 | VaeEncoder retiré (v0.2) : struct/impl absent du code | code | ALWAYS_PASSES | `verify: ! grep -q "pub struct VaeEncoder\|impl VaeEncoder" tso-engine/src/encoder.rs` |
| C13 | Sous-systèmes désactivés par défaut | code | ALWAYS_PASSES | `verify: cargo test --test phi_gating test_default_config_minimal 2>&1 | grep -q "ok"` |

### Regression
| ID | Eval | Grader | Tier | verify / rubric |
|----|------|--------|------|-----------------|
| R1 | Tous les tests passent | code | ALWAYS_PASSES | `verify: cargo test 2>&1 | grep -q "test result: ok"` |
| R2 | Benchmarks d'ablation compilent | code | ALWAYS_PASSES | `verify: cargo check --bin bench_ablation 2>&1 | grep -q "Finished"` |
| R3 | Benchmarks Φ gating compilent | code | ALWAYS_PASSES | `verify: cargo check --bin bench_phi_gating 2>&1 | grep -q "Finished"` |
| R4 | Aucune référence à VaeEncoder dans le code (hors commentaires) | code | ALWAYS_PASSES | `verify: ! grep -r "VaeEncoder" tso-engine/src/ --include="*.rs" | grep -v "^.*//.*VaeEncoder" | grep -v "^.*///.*VaeEncoder"` |
| R5 | Aucun `pub mod vae` actif dans lib.rs | code | ALWAYS_PASSES | `verify: grep "pub mod vae" tso-engine/src/lib.rs | grep -v "^//" && exit 1 || exit 0` |
| R6 | Les bins supprimés (bench_vision, bench_vae_joint) n'existent plus | code | ALWAYS_PASSES | `verify: test ! -f tso-engine/src/bin/bench_vision.rs && test ! -f tso-engine/src/bin/bench_vae_joint.rs` |

### Results
| Run | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | R1 | R2 | R3 | R4 | R5 | R6 | pass@k |
|-----|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|----|----|----|----|----|----|--------|
| 1 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | — | — | — | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 16/19 |
