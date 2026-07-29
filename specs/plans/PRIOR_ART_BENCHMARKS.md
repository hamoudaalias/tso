# Prior Art — Rust grid benchmarks & scale

## Registres

### crates.io — environnements grille Rust
- `mini-grid` (2022) — MiniGrid-like en Rust pur. Obs 5×5×3. Pas de DoorKey.
- `gym-rs` (2023) — bindings Rust pour Gym. Dépendance Python.
- `griddly-headless` (2024) — Rust bindings pour Griddly (C++). Supporte
  grilles, sprites, règles personnalisées. Prometteur mais nécessite C++.
- `ale-rs` (2023) — bindings Rust pour Arcade Learning Environment.
  84×84×4 = 28K dimensions. Trop pour ndarray sans CNN.

### Repos existants dans le projet
- `tso-engine/src/minigrid_env.rs` — 7×7 DoorKey, 147D RGB. Fait main.
- `tso-engine/src/grid_world.rs` — 5×5 open room, 4 whiskers.
- `tso-engine/src/rotating_t.rs` — 5×5 rotating goal, 4D one-hot.
- `tso-engine/src/zigzag_grid.rs` — pas utilisé.

## Littérature / Benchmarks standards

### MiniGrid (Chevalier-Boisvert, 2018–2023)
- Environnements : DoorKey (7×7, 16×16), MultiRoom (N rooms), RedBlueDoors.
- Observations : RGB partiel (7×7×3) ou directionnel (agent view).
- Benchmark standard pour RL visuel procédural.
- Implémentation officielle : Python. Aucun port Rust complet connu.

### Procgen (Cobbe et al., 2020)
- 16 environnements, 200 niveaux chacun.
- Benchmark standard pour généralisation RL.
- Pas de port Rust.

### Griddly (Bamford, 2020–2024)
- Moteur de jeux grille procéduraux en C++.
- Rust bindings via griddly-rs.
- Supporte : n'importe quelle taille de grille, sprites, règles RL.

## Recommandation
- **Court terme** : MiniGrid paramétrique Rust (7×7, 13×13, 19×19).
  Faisable aujourd'hui, zéro dépendance.
- **Moyen terme** : Griddly-rs si validation externe nécessaire.
- **Long terme** : Benchmark TSO sur DoorKey 16×16 (Python → Rust port).
