# TSO : Topographic Stabilization Operator
## Architecture neuromorphique à friction topographique

**Auteur :** Hamouda ALIAS
**Date :** Juillet 2026

---

## Résumé

Cet article présente **TSO**, une architecture neuromorphique Rust qui
intègre catégorisation par prototypes (AttractorField), apprentissage
TD(λ), et gating par friction topographique (Φ).

Sur MiniGrid DoorKey 7×7 (observations RGB 147D), l'AttractorField
seul atteint **+0.49 de reward moyen** par rapport à un régresseur
linéaire (d de Cohen = 2.59, IC 95% sans chevauchement). Le VAE et
les sous-systèmes supplémentaires (Φ, hypothalamus, épisodique)
n'apportent pas de gain significatif sur ce benchmark. Le Φ gating
dégrade les performances.

TSO n'est pas compétitif sur les environnements de faible dimension
(RotatingT 4D) où un Linear AC standard domine (d = −3.89).

---

## 1. Introduction

Les modèles denses d'IA déploient les mêmes ressources par token. TSO
propose une alternative basée sur une friction géométrique Φ mesurant
les contradictions internes. L'architecture complète comprend six
sous-systèmes (AttractorField, VAE, Cerebellum, hypothalamus, mémoire
épisodique, attention spatiale). L'ablation systématique montre que
l'AttractorField est le seul composant déterminant sur MiniGrid.

## 2. Travaux connexes

TSO emprunte au calcul adaptatif (ACT, PonderNet), aux SNN (Dual-LIF),
et à l'inférence active (Friston). Il s'en distingue par son déclencheur
géométrique.

## 3. Friction topographique (Φ)

L'état du système à l'instant t est `X_t = (G_t, S_t, W_t)`. La
friction globale mesure la violation des contraintes géométriques :

- **Contrainte d'Implication** (w_ij=1) : violation = max(0, γ − ⟨z_i, z_j⟩)
- **Contrainte d'Exclusion** (w_ij=−1) : violation = max(0, ⟨z_i, z_j⟩ − ε)

Φ(G_t) = Σ_{(i,j)∈E} Violation_ij.

Note d'ablation : le Φ gating (étape 4 du heartbeat) dégrade les
performances de −0.70σ vs attracteur seul sur MiniGrid 7×7.

## 4. Architecture TSO

Le heartbeat s'exécute en 4 étapes à chaque tick :

1. **Catégorisation** — AttractorField (prototypes) ou VAE ou FPI
2. **Action** — Cerebellum (actor-critic TD(λ))
3. **Apprentissage** — reinforce_td
4. **Gating par Φ** — optionnel, dégrade les performances mesurées

Le gradient TD ne remonte pas dans le VAE.

### 4.1 AttractorField

Classe chaque entrée par similarité cosinus au prototype le plus proche.
Composant clé de TSO : sans lui, les performances chutent de −0.91σ.
Avec lui, TSO surpasse le linéaire de d=2.59.

## 5. Implémentation Rust

Kernel Rust (crate tso-engine, ndarray), zéro dépendance Python. 45 tests
unitaires couvrent les composants. DQN et MLP disponibles comme baselines.

## 6. Expériences

### 6.1 Ablation systématique — MiniGrid 7×7

Benchmark sur MiniGrid DoorKey 7×7 (147D RGB). 30 seeds, 100 épisodes.
Cohen's d vs Linear AC (A0) et vs Attracteur seul (A1).

| Config | Mean | σ | IC 95% | d vs A0 | d vs A1 |
|--------|------|---|--------|---------|---------|
| A0: Linear AC | 0.10 | 0.13 | [0.05, 0.14] | 0.00 | — |
| **A1: TSO attracteur** | **0.59** | 0.24 | **[0.51, 0.68]** | **2.59** | **0.00** |
| A2: TSO + VAE | 0.60 | 0.20 | [0.53, 0.67] | 2.98 | 0.02 |
| A3: TSO full | 0.68 | 0.29 | [0.58, 0.79] | 2.56 | 0.33 |
| A4: TSO + Φ gating | 0.43 | 0.23 | [0.35, 0.51] | 1.76 | −0.70 |
| A5: TSO sans attract. | 0.39 | 0.20 | [0.32, 0.47] | 1.77 | −0.91 |

**Résultats clés :**
- **Attracteur vs linear : d=2.59** — effet large, IC sans chevauchement
- **VAE vs attracteur : d=0.02** — aucun bénéfice mesurable
- **Φ gating : d=−0.70** — dégrade les performances
- **Sans attracteur : d=−0.91** — l'attracteur est indispensable

### 6.2 Analyse multi-échelle

Sur RotatingT 5×5 (4D), TSO perd contre toutes les baselines :

| Agent | Mean | σ | IC 95% | d vs Linear |
|-------|------|---|--------|-------------|
| Linear AC | 3.82 | 0.38 | [3.69, 3.96] | — |
| MLP AC | 3.67 | 1.31 | [3.20, 4.14] | −0.16 |
| DQN | 2.36 | 1.43 | [1.85, 2.87] | −1.40 |
| TSO attracteur | 2.51 | 0.30 | [2.40, 2.61] | −3.89 |

TSO a une variance plus faible que DQN/MLP mais une moyenne inférieure.
L'avantage de l'AttractorField ne compense pas l'overhead sur 4D.

### 6.3 Scale MiniGrid

| Taille | Dim | Linear | TSO attracteur | Δ |
|--------|-----|--------|----------------|---|
| 7×7 | 147D | 0.10 | **0.59** | +0.49 |
| 13×13 | 507D | −0.10 | **0.06** | +0.16 |
| 19×19 | 1083D | −0.10 | **−0.02** | +0.08 |

L'avantage se réduit avec la taille de grille : la navigation devient
le facteur limitant, pas la perception.

## 7. Discussion

**Apport.** L'AttractorField est un mécanisme de réduction dimensionnelle
efficace sur les observations visuelles de haute dimension (147D). Il
surpasse le régresseur linéaire avec un effet large (d=2.59).

**Non-apports.** Le VAE n'améliore pas l'AttractorField. Le Φ gating
dégrade les performances. Les sous-systèmes additionnels (hypothalamus,
épisodique, attention) n'apportent pas de gain sur ce benchmark.

**Limites.** TSO n'est pas compétitif sur les environnements de faible
dimension. L'avantage se réduit avec la complexité de navigation
(grilles plus grandes). Variance seed élevée (σ=0.20–0.29).

## 8. Travaux futurs

Validation sur Procgen, amélioration du Φ gating, apprentissage conjoint
VAE↔Cerebellum (gradient partagé).

## 9. Conclusion

TSO montre qu'un AttractorField suffit à surpasser un régresseur linéaire
sur MiniGrid 7×7 (d=2.59). Les autres mécanismes (Φ, VAE, hypothalamus)
n'apportent pas de gain mesurable. L'approche ne se généralise pas aux
faibles dimensions ni aux grandes grilles.
