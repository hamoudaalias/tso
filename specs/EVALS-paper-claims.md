# EVALS — Claims du papier TSO

## Capability Evals

| ID | Eval | Grader | Tier | Vérification |
|----|------|--------|------|-------------|
| C1 | Phi gating: step count diffère avec/sans | code | ALWAYS_PASSES | cargo test --test phi_gating test_gate_skip |
| C2 | Phi gating bench: reward Δ (ON - OFF) < 0.5σ | code | USUALLY_PASSES | cargo run --bin bench_phi_gating -- --seeds 10 |
| C3 | +105% vs linéaire: delta ≥ 0.8 sur 10 seeds | code | USUALLY_PASSES | cargo run --bin multi_seed_bisect -- --baseline linear --config tso |
| C4 | +105% vs linéaire: delta ≥ 0.3 sur 30 seeds | code | EXPERIMENTAL | cargo run --bin multi_seed_bisect -- --seeds 30 |
| C5 | TSO vs MLP: delta > 0 | code | EXPERIMENTAL | cargo run --bin bench_vs_mlp -- --seeds 10 |
| C6 | Attracteur vs K-means: delta < 0.5σ | code | EXPERIMENTAL | cargo run --bin bench_prototypes -- --compare kmeans |
| C7 | Gradient coupé: VAE weights inchangés après TD | code | ALWAYS_PASSES | cargo test --test encoder_unit test_gradient_frozen |
| C8 | VAE encode->reparam->decode: ELBO fini | code | ALWAYS_PASSES | cargo test --test vae_unit |
| C9 | 45+ tests passent: cargo test --lib --tests | code | ALWAYS_PASSES | cargo test --lib --tests 2>&1 | tail -5 |

## Regression Evals

| ID | Eval | Grader | Tier | Vérification |
|----|------|--------|------|-------------|
| R1 | MiniGrid 10 seeds reproductible (reward > 2.0) | code | ALWAYS_PASSES | cargo run --bin bench_minigrid -- --seeds 10 --check 2.0 |
| R2 | Aucun test existant cassé | code | ALWAYS_PASSES | cargo test 2>&1 | grep -E "FAILED|test result" |
| R3 | Phi compute unchanged | code | USUALLY_PASSES | cargo test --test phi_gating |

## pass@k notation

Chaque eval rapporte: [pass/total] [tier]

Exemple:
C3: +105% vs lineaire -> 3/3 USUALLY_PASSES
C4: +105% vs lineaire 30 seeds -> 1/3 EXPERIMENTAL
C5: TSO vs MLP -> 0/3 EXPERIMENTAL

ALWAYS_PASSES doit etre 3/3 pour release.
USUALLY_PASSES doit etre >= 2/3.
EXPERIMENTAL est informatif.