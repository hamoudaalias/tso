# Deepen Architecture — Pipeline encodeur + renderer

## Résultat du research-first

**Aucun renderer GridWorld en Rust existant.** Le seul renderer est un script
Python (`scripts/collect_gridworld_frames.py`) qui exporte des one-hot 25D en `.bin`.

## Opportunité de deepening

### 1. Renderer GridWorld natif (✅ fait)

**Problème :** renderer en Python → dépendance pyo3, pas de visualisation en Rust.
**Solution :** `render_ascii()` + `render_png()` (feature `viz`, dépendance `image` 0.25).
**Bénéfice :** visualisation locale, pas de dépendance Python, utilisable dans les benches.

Module Depth score : **2 → 4** (interface : 2 méthodes, impl : ~50 lignes + `image` crate)

### 2. Encoder trait + deux adaptateurs (identifié, non fait)

**Problème :** `AttractorEncoder` et `VaeEncoder` sont deux adaptateurs du même trait
`Encoder`, mais `VaeEncoder` n'est pas sur master (branche e09). Le trait `Encoder`
a 4 méthodes, dont certaines optionnelles (`adapt`, `vae_stats`).

**Opportunité :** simplifier le trait en 2 méthodes obligatoires (`encode_raw`, `encode_continuous`).
Le reste (`adapt`, `vae_stats`) en méthodes par défaut.

Module Depth score : **3** (déjà correct, peut être meilleur)

### 3. Config double source (identifié, accepté)

**Problème :** `CognitiveConfig` + `NeurogenesisConfig` sont deux structs avec les mêmes champs.
La sync est manuelle dans `sleep_cycle()`.

**Solution :** déprécier les champs `sleep_*` de `CognitiveConfig`, utiliser uniquement
`NeurogenesisConfig`. Break les anciens benchmarks → nécessite une migration.

Module Depth score : **2** (shallow — deux interfaces pour le même concept)

### Recommandation

- **Garder** le renderer ajouté
- **Migrer** la config double quand on touchera CognitiveConfig pour autre chose
- **Reporter** la simplification de l'Encoder trait à e09
