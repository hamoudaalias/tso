# TSO — Topographic Stabilization Operator

**Une architecture neuromorphique où la friction topographique (Φ) remplace l'attention des Transformers. 100% Rust, zéro GPU, zéro gradient.**

---

## Résultats clés

| Propriété | Résultat | vs Transformer |
|-----------|----------|----------------|
| SNLI test (classification) | **57.03%** (V14 DeepTSO 2L, 20D, 30 clusters) | +0.34% vs V13, hiérarchie validée |
| Temps d'entraînement | **~2 minutes** (CPU 28 cœurs, R-STDP inter-couches) | vs ~30 min sur GPU |
| Apprentissage continu | **Δ = 0.00%** (oubli catastrophique vaincu) | Impossible sans EWC/replay |
| Scalabilité | **V=10⁶ en 95s** (SVD randomisée, 800 MB) | vs ~3 Go pour embeddings Transformer |
| Génération auto-régressive | **Dérive sémantique émergente** (sans backprop) | Aucun équivalent |

---

## Utilisation

```bash
# SNLI classification (pipeline complet)
cargo run --release --bin tso-bench eval data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_test.jsonl 5 20 100 checkpoints_snli

# Apprentissage continu SNLI → MultiNLI
cargo run --release --bin tso-bench continual data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl data/multinli_1.0/multinli_1.0_train.jsonl

# Génération auto-régressive (V6 Dual-LIF)
cargo run --release --bin tso-bench generate checkpoints_snli "a man is"

# DeepTSO validation (V14) — 1 couche
cargo run --release --bin tso-bench deval data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl 30 1

# DeepTSO validation (V14) — 2 couches + R-STDP inter-couches
cargo run --release --bin tso-bench deval data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl 30 2
```

---

## Génération : La dérive sémantique en action

Trois prompts, trois trajectoires dans l'espace latent — sans probabilités, sans softmax, sans gradient :

| Prompt | V6.0 Dual-LIF Génératif |
|--------|------------------------|
| `a man is` | `man is a the on in and wearing shirt red white black with blue jacket green boy young girl little` |
| `the dog ran` | `dog brown running grass runs across the is a on man in and with black shirt white red wearing blue` |
| `a woman sits` | `woman a is man the in and wearing red shirt white black jacket blue hat with on sitting bench sits` |
| `a man is` (V7 anchored) | `man is wearing a red shirt and blue jeans with black jacket white hat on the grass` (35% anchor recall à t=7, dérive plafonnée) |
| `the dog ran` (V9 Triple-LIF) | `dog runs across the grass in the park with a ball and a boy playing fetch` (ancre téléportée à t=18 du chien → parc → ballon) |

Le système navigue le graphe de friction par **Inverse Motor** : $w_{t+1} = \arg\max \langle S_{slow} + \eta_m \cdot S_{medium} + \eta_f \cdot S_{fast}, e(w) \rangle$. L'arrêt est homéostatique — quand l'état cesse de changer (Φ < seuil), le système se tait.

---

## Architecture

| Composant Transformer | Équivalent TSO | Module |
|----------------------|----------------|--------|
| Self-Attention | Friction topographique Φ | `friction.rs` |
| Feed-Forward | Réservoir LIF (Leaky Integrate-and-Fire) | `neurons.rs` |
| Multi-Head Attention | Triple-LIF (α=0.9 lent / α=0.7 moyen / α=0.5 rapide) | `decoder.rs` |
| Backpropagation | R-STDP (plasticité locale, zéro gradient) | `plasticity.rs` |
| Tête de classification | AttractorField (k-means + LVQ1) | `attractor.rs` |
| Embedding / Projection | Double Mapping / Inverse Motor (intersection dot) | `operators.rs`, `decoder.rs` |
| Padding global de l'expansion | Expansion Asynchrone (V10, dimensions variables) | `decoder.rs` (Vec<Array1>, ensure_dim) |
| Positional Encoding | Trace temporelle LIF | `neurons.rs` |
| Critic (évaluation globale) | Onde de Choc Locale (V8) | `friction.rs` (LocalWaveCritic) |
| Apprentissage lent des exclusions | Cicatrice Morphologique Volatile (V9.1) → Instinct Endogène (V11) | `decoder.rs` (EndogenousInversionDetector) |
| Oscillation infinie du Critic | Coupe-Circuit de Fatigue (V13, isolement temporaire) | `friction.rs` (FatigueTracker) |
| Absence de hiérarchie | DeepTSO (V14, cycle cortical à 2 phases, Φ inter-couche) | `deep.rs` (DeepTSO, DeepConfig) |

---

## État du projet (v14.1)

- [x] Classification SNLI (57.03% test, V14 DeepTSO 2L + R-STDP, ~2min CPU)
- [x] Dual-LIF multi-échelle (mémoire lente + rapide)
- [x] Apprentissage continu (oubli catastrophique vaincu structuralement)
- [x] Scalabilité jusqu'à V=10⁶ (95s, 800 MB, pas de GPU)
- [x] Génération auto-régressive (Inverse Motor + Φ homeostasis)
- [x] Dual-LIF Génératif (syntaxe améliorée par état prédictif composé)
- [x] Anchored Decoder V7 (mémoire épisodique, dérive contrôlée)
- [x] LocalWaveCritic V8 (Critic local asynchrone sans évaluation globale)
- [x] Triple-LIF V9 (α=0.7 medium, ancre dynamique téléportable)
- [x] Expansion Asynchrone V10 (`Vec<Array1<f64>>`, dimensions variables, plus de padding global)
- [x] Instinct Endogène V11 (apprentissage de la négation par la friction, plus de code en dur)
- [ ] **Remodelage Synaptique V12** — pruning sous friction pour restructuration profonde (concept)
- [x] **Coupe-Circuit de Fatigue V13** — isolement temporaire des nœuds pour briser les boucles paradoxales
- [x] **DeepTSO V14** — cycle cortical à 2 phases, Φ inter-couche, modulation top-down, R-STDP inter-couches
- [ ] Benchmark énergétique RAPL (nécessite machine Linux native)

---

## Publication

Le [`paper.md`](paper.md) détaille la théorie (CDT), l'architecture, les 7 opérateurs cognitifs, et l'ensemble des résultats expérimentaux.

Hamouda ALIAS, Juillet 2026.
