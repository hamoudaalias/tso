# Test Plan — Claims du papier TSO

## Risk Matrix

| Claim | Risque | Justification |
|-------|--------|---------------|
| Φ gating réduit le calcul | P0 | Claim principal du papier, jamais isolé |
| +105% vs régresseur linéaire | P0 | Baseline minimal, pas de test statistique |
| Catégorisation par prototypes | P1 | Comparé à rien (raw->linear), pas à d'autres méthodes |
| VAE Gumbel-STE (hors-ligne) | P1 | VAE existe, mais "online" était faux |
| Gradient coupé VAE<->Cerebellum | P2 | Détail architectural, pas un claim |
| 45 tests couvrent les composants | P2 | Vérification CI |
| Dual-LIF (alpha lent/rapide) | P3 | Non benchmarké, descriptif |

## Stratégie par niveau

| Niveau | Scénarios | Coût |
|--------|-----------|------|
| Unit | Phi compute, VAE forward, gradient freeze, gate skip | < 1s |
| Integration | Cycle complet avec/sans gating, VAE+attracteur | < 10s |
| Benchmark | MiniGrid 10 seeds, ablation composant | ~5 min |
| Statistique | Welch, CI, power analysis | hors-ligne |

## Scénarios

### P0 — Phi gating réduit le calcul
**SC-P0-01** Phi = 0.0 en sortie de reset (existe: phi_gating.rs)
**SC-P0-02** gate skip: nb étapes cognitives avec/sans gating diffère
**SC-P0-03** bench Phi: reward moyen gating ON vs OFF (mêmes seeds)
**SC-P0-04** bench Phi: FLOPs ou pas CPU avec/sans gating

### P0 — +105% vs régresseur linéaire
**SC-P0-05** bench: TSO vs régresseur linéaire, 30 seeds (pas 10)
**SC-P0-06** bench: TSO vs MLP 1 couche cachée (147->64->4)
**SC-P0-07** bench: TSO vs DQN simple (147->256->4, replay buffer)
**SC-P0-08** stats: Welch t-test, IC 95% sur le delta

### P1 — Catégorisation par prototypes
**SC-P1-01** bench: TSO + attracteur vs TSO + K-means
**SC-P1-02** bench: TSO + attracteur vs TSO + GMM
**SC-P1-03** bench: TSO + attracteur vs TSO sans catégorisation

### P1 — VAE Gumbel-STE
**SC-P1-04** unit: VAE encode->reparam->decode (existe)
**SC-P1-05** unit: ELBO loss décroît (existe)
**SC-P1-06** bench: VAE latents vs raw 147D vs PCA 16D

### P2 — Gradient coupé
**SC-P2-01** unit: snap VAE weights avant reinforce_td, vérifier après

### P2 — CI
**SC-P2-02** cargo test --lib --tests passe toujours

## Fixtures
- MultiSeedRunner: seeds {0..29} -> moyenne, sigma, IC
- BenchComparison: 2 configs -> delta, t-stat, p-value
- AblationSuite: 1 composant OFF -> delta vs full TSO

## Budget
Phase 1 (P0, unit): ~5 min CI
Phase 2 (P0, bench): ~30 min CI
Phase 3 (P1): ~15 min CI