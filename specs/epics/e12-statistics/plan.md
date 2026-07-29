# Statistics & Multi-Seed Benchmarking

## Story s01: MultiSeedRunner struct with statistics
- **s01t01** — Create `src/baselines/multi_seed.rs`:
  - `pub struct MultiSeedRunner { seeds: usize, episodes: usize }`
  - `run(agent_fn, env_fn) -> SeedResults { mean, std, ci95, cohens_d, welch_p }`
- **s01t02** — Implémenter les fonctions stats :
  - `mean(v)`, `std(v)`, `ci95(v)` (normal approximation)
  - `cohens_d(m1, m2, s1, s2)`
  - `welch_t(m1, m2, s1, s2, n1, n2)` + approximation de Welch-Satterthwaite

## Story s02: Upgrade multi_seed_bisect
- **s02t01** — Ajouter env MiniGrid 7×7 aux configs existantes
- **s02t02** — Ajouter DQN comme baseline dans la matrice
- **s02t03** — Formater output avec IC 95%, Cohen's d
- **s02t04** — Sauvegarder résultats dans `specs/benchmarks/`

## Story s03: bench_all unified
- **s03t01** — `bin/bench_all.rs` qui run (A0-A7) × (E1-E4) et imprime tableau markdown
- **s03t02** — Export JSON des résultats
