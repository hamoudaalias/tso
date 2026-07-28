## Prior Art — Scalabilité GridWorld → Minigrid → Habitat

### Candidates

| Solution | Source | Verdict | Raison |
|----------|--------|---------|--------|
| **`gymnasium`** (crates.io v0.0.1) | FFI bindings Python Gymnasium | ⚠️ **extend** | Dépendance Python runtime, doc minimale, version précoce |
| **`gymnasia`** (crates.io) | Pure Rust Gymnasium API | ❌ **compose** | Réimplémentation partielle, pas de MiniGrid natif |
| **PyO3 Minigrid wrapper** | 40 lignes de binding Rust → Python | ✅ **build** | Accès à TOUS les environnements Python sans réimplémentation |
| **Réimplémentation TSO GridWorld** | `tso-engine/src/grid_world.rs` | ✅ **extend** | GridWorld existant, extensible en complexité (murs, pièges, portes) |
| **Habitat via PyO3** | Python 3D simulator | ⚠️ **build** | Possible mais lourd : PyO3 + rendering → latence |

### Recommandation

1. **Court terme** : étendre `GridWorld` TSO (murs internes, téléporteurs, ports, clés).
   Le module `grid_world.rs` a déjà `empty_room()`, `corridor()`, `random()`.
   Ajouter une config `maze_custom(w, h, walls, goals, teleports)` supporte
   des scénarios MiniGrid-like sans dépendance externe.

2. **Moyen terme** : wrapper PyO3 `tso_env` qui expose TSOGridWorld comme
   environnement Gymnasium (step() / reset() / render()) → interop Python.
   Inverse : appeler Python Minigrid depuis Rust via PyO3.

3. **Long terme** : bridge Habitat via PyO3 asynchrone (images 3D → VAE encoder).
