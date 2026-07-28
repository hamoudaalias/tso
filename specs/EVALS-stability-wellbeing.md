# EVALS: Stabilité du bien-être sur 100 seeds

## Capability

| ID | Eval | Grader | Tier | Vérification |
|----|------|--------|------|--------------|
| C1 | Référence (poids=1.0, δ-clip off) : succès moyen sur 100 seeds | code | USUALLY_PASSES | `cargo run --bin eval_stability -- --ref` |
| C2 | metabolic_penalty ×5 : delta > +10 pts vs référence | code | EXPERIMENTAL | `cargo run --bin eval_stability -- --metabolic-5` |
| C3 | parsimony ×2 : delta > +10 pts vs référence | code | EXPERIMENTAL | `cargo run --bin eval_stability -- --parsimony-2` |
| C4 | gated_reward ×0 : delta < +5 pts (doit être neutre) | code | EXPERIMENTAL | `cargo run --bin eval_stability -- --gate-0` |
| C5 | Variance inter-seeds : σ < 20% pour chaque config | code | USUALLY_PASSES | `cargo run --bin eval_stability -- --variance` |

## Regression

| ID | Eval | Grader | Tier | Vérification |
|----|------|--------|------|--------------|
| R1 | 0 warnings, tous les tests passent | code | ALWAYS_PASSES | `cargo check --bins && cargo test` |

## Protocol

Chaque run : env 5×5, TSO complet sans δ-clip, 200 train + 50 test ε=0.
Résultat : moyenne ± écart-type sur 100 seeds, heatmap de densité.
