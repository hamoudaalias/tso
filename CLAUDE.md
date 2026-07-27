# TSO — Tension-Solving Organism

## Stack
- Rust 2024 edition
- ndarray 0.16 (avec serde)
- serde + bincode (sérialisation)
- rand 0.8

## Commandes essentielles
```bash
# Vérifier la lib
cd tso-engine && cargo check --lib

# Vérifier tous les binaires
cd tso-engine && cargo check --bins

# Build release
cd tso-engine && cargo build --release

# Lancer un binaire spécifique
cd tso-engine && cargo run --bin debug_rl

# Tester
cd tso-engine && cargo test
```

## Architecture
Moteur cognitif bio-inspiré en Rust :
- `core.rs` — Graphe sémantique + Φ (tension cognitive) + résolution de contraintes
- `attractor.rs` — Catégoriseur à prototypes (AttractorField)
- `cerebellum.rs` — Actor-critic (TD(λ) + replay buffer)
- `hypothalamus.rs` — Régulation homéostatique (énergie, hydratation, température)
- `episodic.rs` — Mémoire épisodique (séquences, prédiction par suffixe)
- `working_memory.rs` — DualLIF + mémoire associative
- `attention.rs` — Attention spatiale (gain par erreur de prédiction)
- `tso_engine.rs` — Cycle cognitif principal (4 étapes/heartbeat)
- `grid_world.rs` / `grid_cells.rs` — Environnement + cellules de grille

## Conventions
- Toujours lancer `cargo check --bins` avant un commit
- Toujours travailler sur une branche feature (cf. `kickoff-branch`)
- Un commit = un changement logique (pas de "fix + refactor + doc" dans le même commit)
- Les specs vont dans `specs/`, l'architecture dans `paper.md`
