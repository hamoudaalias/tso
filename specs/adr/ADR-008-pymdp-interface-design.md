# ADR-008 : Interface Design des 7 modules pymdp

## Statut
Accepté — e11/efe-integration, merge sur master.

## Contexte
TSO utilise des mécanismes ad-hoc non-probabilistes (resolve_with_anneal,
reinforce_td, attractor hebbien) pour l'inférence, le contrôle et
l'apprentissage. Le formalisme pymdp (FPI, EFE, Dirichlet) apporte
une fondation bayésienne unifiée. L'import direct est impossible (JAX ≠ ndarray).

## Décision
Extraire le noyau mathématique de chaque module pymdp et le porter en ndarray Rust.

Modules portés :
| Module pymdp | Fichier Rust | Sémantique |
|---|---|---|
| Modèle génératif (A, B, C, D, E) | model.rs | Struct GenerativeModel + vérification des dimensions |
| FPI (Fixed Point Iteration) | fpi.rs | run_vanilla_fpi, run_factorized_fpi |
| VFE (Variational Free Energy) | inference.rs | calc_vfe, infer_states |
| EFE / Scoring (Expected Free Energy) | efe.rs | expected_utility, info_gain, score_policy |
| Dirichlet A update | learning.rs | update_obs_likelihood_dirichlet |
| Dirichlet B update | learning.rs | update_state_transition_dirichlet |
| Modulateur | modulator.rs | Policy precision + action selection |

Chaque module est optionnel via des flags dans CognitiveConfig (default=false).

## Conséquences
- Positif : fondation bayésienne pour les décisions futures
- Positif : rétrocompatibilité totale
- Négatif : 5 nouveaux fichiers sources, complexité cognitive accrue
- Risque : les modules ne sont pas testés en intégration continue avec le TSO complet
