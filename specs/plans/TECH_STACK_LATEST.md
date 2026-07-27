# TECH_STACK — TSO

## Langage
| Couche | Technologie | Version |
|--------|------------|---------|
| Langage | Rust | 2024 edition |
| Build | Cargo | stable |

## Dépendances principales
| Crate | Version | Usage |
|-------|---------|-------|
| ndarray | 0.16 (features: serde) | Vecteurs, matrices, algèbre linéaire |
| serde | 1.x (derive) | Sérialisation des modèles |
| bincode | 1.3 | Encodage binaire pour la sauvegarde |
| rand | 0.8 | RNG pour exploration et recuit simulé |
| ctrlc | 3.4 | Gestion du signal d'arrêt |

## Structure du projet
```
tso-engine/
├── Cargo.toml
└── src/
    ├── lib.rs              # Module declarations
    ├── main.rs             # Entry point CLI
    ├── tso_engine.rs       # Cycle cognitif complet
    ├── core.rs             # Graphe sémantique + Φ + résolution
    ├── attractor.rs        # AttractorField (classification)
    ├── cerebellum.rs       # Actor-Critic RL
    ├── hypothalamus.rs     # Régulation homéostatique
    ├── episodic.rs         # Mémoire épisodique
    ├── working_memory.rs   # DualLIF + associative
    ├── attention.rs        # Attention spatiale
    ├── memory.rs           # Mémoire associative vectorielle
    ├── neurons.rs          # Dynamique neuronale LIF
    ├── action.rs           # Actions de l'agent
    ├── replay_buffer.rs    # Experience replay
    ├── grid_world.rs       # Environnement grille
    ├── grid_cells.rs       # Cellules de grille spatiales
    ├── multi_grid_cells.rs # Grid cells multi-modules
    ├── sokoban.rs          # Environnement Sokoban
    ├── terrarium.rs        # Environnement Terrarium
    └── bin/                # 18 binaires expérimentaux
```

## Déploiement
Pas de déploiement serveur pour le moment — binaire autonome.

## CI
Pas encore configurée (prévue via wire-ci).

## Dépendances futures
- [tracing](https://crates.io/crates/tracing) — observabilité structurée (wire-observability)
- [rayon](https://crates.io/crates/rayon) — parallélisation si resolve_parallel passe en work-stealing
