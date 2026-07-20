# TSO — Topographic Stabilization Operator

**Une architecture neuromorphique où la friction topographique (Φ) remplace l'attention des Transformers. 100% Rust, zéro GPU, zéro gradient.**

---

## Résultats clés

| Propriété | Résultat | vs Transformer |
|-----------|----------|----------------|
| SNLI test (classification) | **56.69%** (17D Dual-LIF) | +1.84% au-delà du plafond sac-de-mots |
| Temps d'entraînement | **~20 secondes** (CPU 8 cœurs) | vs ~30 min sur GPU |
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

Le système navigue le graphe de friction par **Inverse Motor** : $w_{t+1} = \arg\max \langle S_{slow} + \eta \cdot S_{fast}, e(w) \rangle$. L'arrêt est homéostatique — quand l'état cesse de changer (Φ < seuil), le système se tait.

---

## Architecture

| Composant Transformer | Équivalent TSO | Module |
|----------------------|----------------|--------|
| Self-Attention | Friction topographique Φ | `friction.rs` |
| Feed-Forward | Réservoir LIF (Leaky Integrate-and-Fire) | `neurons.rs` |
| Multi-Head Attention | Dual-LIF (α=0.9 lent / α=0.5 rapide) | `distributional.rs` |
| Backpropagation | R-STDP (plasticité locale, zéro gradient) | `plasticity.rs` |
| Tête de classification | AttractorField (k-means + LVQ1) | `attractor.rs` |
| Embedding / Projection | Double Mapping / Inverse Motor | `operators.rs`, `decoder.rs` |
| Positional Encoding | Trace temporelle LIF | `neurons.rs` |
| Critic (évaluation globale) | Onde de Choc Locale (V8) | `friction.rs` (LocalWaveCritic) |

---

## État du projet (v8.0)

- [x] Classification SNLI (56.69% test, ~20s CPU)
- [x] Dual-LIF multi-échelle (mémoire lente + rapide)
- [x] Apprentissage continu (oubli catastrophique vaincu structuralement)
- [x] Scalabilité jusqu'à V=10⁶ (95s, 800 MB, pas de GPU)
- [x] Génération auto-régressive (Inverse Motor + Φ homeostasis)
- [x] Dual-LIF Génératif (syntaxe améliorée par état prédictif composé)
- [x] Anchored Decoder V7 (mémoire épisodique, dérive contrôlée)
- [x] LocalWaveCritic V8 (Critic local asynchrone sans évaluation globale)
- [ ] Benchmark énergétique RAPL (nécessite machine Linux native)

---

## Publication

Le [`paper.md`](paper.md) détaille la théorie (CDT), l'architecture, les 7 opérateurs cognitifs, et l'ensemble des résultats expérimentaux.

Hamouda ALIAS, Juillet 2026.
