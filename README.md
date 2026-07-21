# TSO — Topographic Stabilization Operator

**Une architecture neuromorphique où la friction topographique (Φ) remplace l'attention des Transformers. 100% Rust, zéro GPU, zéro gradient, zéro SVD.**

---

## Résultats clés

| Propriété | Résultat | vs Transformer |
|-----------|----------|----------------|
| SNLI test (classification) | **56.31%** (V16 Cold Start — projections aléatoires, 4L DeepTSO, R-STDP pur) | Apprentissage endogène *ex nihilo* |
| SNLI test (Warm Start SVD) | **56.50%** (V16 Warm — départ SVD, Δ=0.19% vs Cold) | Écart nul : SVD non nécessaire |
| SNLI dev (optimal) | **57.15%** (V15 — 4L DeepTSO, WTA 94% sparsity, SVD) | +0.46% vs V13, parcimonie garantie |
| Temps d'entraînement | **~10 minutes** (CPU 28 cœurs, WTA + pré-entraînement R-STDP) | vs ~30 min sur GPU |
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

# DeepTSO V15 — 4 couches, WTA 5%, pré-entraînement non supervisé
cargo run --release --bin tso-bench deval data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl 50 4

# DeepTSO V16 — Cold Start : projections aléatoires, zéro SVD
cargo run --release --bin tso-bench deval data/snli_1.0/snli_1.0/snli_1.0_train.jsonl \
  data/snli_1.0/snli_1.0/snli_1.0_test.jsonl 50 4 --cold-start
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
| Embedding / Projection | Double Mapping / Inverse Motor (intersection dot) / **WordProjector (V16, R-STDP)** | `operators.rs`, `decoder.rs`, `projector.rs` |
| Padding global de l'expansion | Expansion Asynchrone (V10, dimensions variables) | `decoder.rs` (Vec<Array1>, ensure_dim) |
| Positional Encoding | Trace temporelle LIF | `neurons.rs` |
| Critic (évaluation globale) | Onde de Choc Locale (V8) | `friction.rs` (LocalWaveCritic) |
| Apprentissage lent des exclusions | Cicatrice Morphologique Volatile (V9.1) → Instinct Endogène (V11) | `decoder.rs` (EndogenousInversionDetector) |
| Oscillation infinie du Critic | Coupe-Circuit de Fatigue (V13, isolement temporaire) | `friction.rs` (FatigueTracker) |
| Absence de hiérarchie | DeepTSO (V14, cycle cortical à 2 phases, Φ inter-couche) | `deep.rs` (DeepTSO, DeepConfig) |

---

## État du projet (v16.0 — Tabula Rasa)

- [x] Classification SNLI (57.15% dev V15, 56.31% test V16 Cold Start — 4L DeepTSO + WTA + pré-entraînement, ~10min CPU)
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
- [x] **DeepTSO V15** — WTA (94% sparsity garantie), pré-entraînement non supervisé 11M mots, input scaling LIF
- [x] **V16 Tabula Rasa** — WordProjector appris par R-STDP, Cold Start 56.31% (écart 0.19% vs SVD), système 100% autonome, zéro SVD
- [ ] Benchmark énergétique RAPL (nécessite machine Linux native)

---

## Publication

Le [`paper.md`](paper.md) détaille la théorie (CDT), l'architecture, les 7 opérateurs cognitifs, et l'ensemble des résultats expérimentaux.

Hamouda ALIAS, Juillet 2026.
