# TSO-CORE : Architecture

> Ce document définit l'architecture TSO complète (groupe B).
> Les sous-ensembles validés et hypothétiques sont distingués par section.

## 1. Modules

### 1.1 Cœur validé (A — AttractorField + Cerebellum)

| Module | Rôle | Statut | Défaut |
|--------|------|--------|--------|
| AttractorField | Catégorisation par prototypes, similarité cosinus | Validé (d=2.59 sur MiniGrid 7×7) | Activé |
| Cerebellum | Actor-critic TD(λ), linear ou MLP | Validé (indispensable) | Activé |
| ActionMotor | Sélection et exécution d'action | Validé (trivial) | Activé |
| TsoEngine | Cycle step() + heartbeat | Validé (43 tests) | Activé |

### 1.2 Extensions (B — modules théoriques)

Extensions classées par état de validation. Toutes désactivées par défaut
dans CognitiveConfig sauf mention contraire.

#### B1 — Implémentés, impact nul sur MiniGrid

| Module | Rôle | Feature flag | Statut |
|--------|------|-------------|--------|
| Hypothalamus | Régulation homéostatique (énergie, hydratation, température) | `hypothalamus` | Pas de gain mesuré |
| Attention spatiale | Gain par erreur de prédiction | `attention` | Pas de gain mesuré |
| Mémoire épisodique | Prédiction par suffixe, curiosité intrinsèque | `episodic` | Pas de gain mesuré |
| Graphe Φ (résolution) | Satisfaction de contraintes sur sphère, Φ gating (skip résolution) | `graph_phi` | Pas de gain mesuré sur MiniGrid ; non-dégradant après correction v0.2 |

#### B2 — Destinés à l'inférence active (non RL)

| Module | Rôle | Feature flag |
|--------|------|-------------|
| FPI | Fixed-point iteration, mise à jour de croyances | `active-inference` |
| EFE | Expected Free Energy scoring | `active-inference` |
| Inference | Free-energy / VFE calculation | `active-inference` |
| Learning | Dirichlet updates (A, B matrices) | `active-inference` |
| Model | Generative model structure | `active-inference` |

#### B3 — Feature-gatés (spéculatifs)

| Module | Rôle | Feature flag | Statut |
|--------|------|-------------|--------|
| R-STDP (plasticity) | Plasticité hebbienne spike-timing | `rstdp` | rstdp_enabled: false par défaut |
| Interop (PyO3) | Bridge Python | `interop` | Infrastructure |

#### B4 — Toujours compilés (pas de feature flag)

| Module | Rôle | Statut |
|--------|------|--------|
| WorkingMemory (Dual-LIF) | Mémoire de travail associative à deux dynamiques | Spéculatif — getters ajoutés (membrane_potential, spike_rate) |
| GridCells | Encodage de position, désambiguïsation d'aliasing | Non testé sur benchmark |
| Neurogenesis | Naissance/maturation/pruning de concepts | sleep_neurogenesis_rate: 0 par défaut |
| AssociativeMemory (memory) | Lookup associatif (infrastructure) | Infrastructure, non cognitif |
| ReplayBuffer | Stockage de transitions pour Cerebellum MLP | Infrastructure |
| PerceptualBelt | Pipeline perception → concept | Infrastructure |
| Encoder | Trait commun AttractorEncoder (VaeStats retiré) | Infrastructure |
| Neurons | Types neuronaux (DualLIFState) | Infrastructure |

## 2. Flux de données (cycle step())

```
Perception (Array1<f64>)
    ↓
[Attention] — optionnel (feature gate)
    ↓
WorkingMemory.observe()
    ↓
Catégorisation (AttractorField) — TOUJOURS
    ↓
[Épisodique] — optionnel
    ↓
[Graphe Φ] — optionnel
    ├─ resolve (conditionnel: Φ ≥ threshold)
    ├─ add_transition (conditionnel: Φ ≥ threshold)
    └─ periodic pruning
    ↓
Cerebellum.forward_logits()
    ↓
RL signal → reinforce_td → mark(action)
    ↓
Action (usize)
```

## 3. Configuration par défaut

```rust
CognitiveConfig {
    attractor: true,        // cœur validé (A)
    graph_phi: true,        // cœur validé (A) — résolution de contraintes
    phi_gating: false,      // extension B1 — pas de gain mesuré
    attention: false,       // extension B1
    episodic_curiosity: false,
    hypothalamus: false,
    use_fpi: false,         // extension B2
    efe_weight: 0.0,
    delta_clip_max: 5.0,
    rstdp_enabled: false,   // extension B3
    sleep_neurogenesis_rate: 0.0,
}
```

## 4. Feature flags Cargo

```toml
[features]
default = ["cognitive-cycle"]
cognitive-cycle = []        # cœur A — feature marqueur (pas de cfg dans le code)
active-inference = []       # B2 — FPI, EFE, inference
hypothalamus = []           # B1
attention = []              # B1
episodic = []               # B1
graph_phi = []              # B1 — Φ gating + résolution
rstdp = []                 # B3 — R-STDP plasticity
interop = []               # B3 — PyO3 bridge Python
```

## 5. Références

- `paper.md` — publication scientifique complète
- `specs/tech-architecture/cdt-formal.md` — preuves formelles (esquisses)
- `specs/product/TSO-EFFICACY.md` — résultats de validation
- `specs/product/TSO-HYPOTHESES.md` — extensions futures
- `specs/EVALS-TSO-CORE.md` — evals de capability et régression
