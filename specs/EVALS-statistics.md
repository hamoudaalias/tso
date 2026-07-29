# EVALS — Significativité TSO

## Capability Evals

| ID | Eval | Grader | Tier | Vérification |
|----|------|--------|------|-------------|
| S1 | TSO attracteur > linear AC : Cohen's d > 0.5 sur MiniGrid 7×7 30 seeds | code | USUALLY_PASSES | cargo run --release --bin multi_seed_bisect -- --env minigrid7 --n 30 --stat cohen |
| S2 | TSO attracteur > linear AC : IC 95% ne chevauche pas zéro | code | USUALLY_PASSES | cargo run --release --bin multi_seed_bisect -- --env minigrid7 --n 30 --stat ci95 |
| S3 | Φ gating effect : |Δ| < 0.5σ (neutre ou négatif acceptable) | code | EXPERIMENTAL | cargo run --release --bin bench_phi_gating -- 30 |
| S4 | TSO VAE vs attracteur : |Δ| < 0.3σ (équivalents) | code | EXPERIMENTAL | cargo run --release --bin bench_minigrid -- --vae |
| S5 | Scale 7×7 vs 13×13 : Δ TSO - linear décroît | code | EXPERIMENTAL | cargo run --release --bin bench_scale |

## Metrics précises

Chaque benchmark rapporte :
- N seeds
- Moyenne ± σ
- IC 95% (moyenne ± 1.96 × σ/√N)
- Cohen's d = (m₁ - m₂) / σ_pooled
- Welch t = (m₁ - m₂) / √(σ₁²/n₁ + σ₂²/n₂)
