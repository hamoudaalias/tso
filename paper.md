# TSO : Topographic Stabilization Operator
## Architecture neuromorphique à friction topographique

**Auteur :** Hamouda ALIAS  
**Date :** Juillet 2026  
**Discipline :** Architectures d'IA, Systèmes Neuromorphiques, Systèmes Dynamiques

---

## Résumé

Les architectures d'IA actuelles maintiennent une relation fixe entre l'information
entrante et le calcul : une quantité constante d'opérations est exécutée par token,
indépendamment de la complexité cognitive de l'entrée. Cet article propose une
alternative architecturale, **TSO (Topographic Stabilization Operator)**, où le calcul
devient une conséquence d'une mesure interne d'instabilité plutôt qu'une obligation
liée à l'arrivée d'une donnée. Fondée sur la Théorie de la Dissipation Cognitive (CDT),
l'architecture TSO modélise l'activité neuronale comme un processus de minimisation
d'une énergie de friction Φ, formellement définie comme une contrainte géométrique
calculable sur un graphe conceptuel émergent.

**Contributions clés :**
1. **Friction topographique (Φ)** : le calcul est événementiel, proportionnel
   à la complexité de l'entrée, pas à sa longueur.
2. **AttractorField** : catégorisation par prototypes compétitifs, validée par
   ablation (d de Cohen = 2.59 vs linéaire). Remplace le VAE après démonstration
   de sa supériorité (§6.1).
3. **Φ gating** : le déclenchement du calcul est conditionné par un seuil de
   tension cognitive, avec compteurs de skip et de résolution intégrés au cycle.
4. **Benchmark MiniGrid** (147D RGB) : TSO + attracteur bat le linéaire de +105%,
   démontrant l'avantage de la catégorisation par prototypes sur les entrées
   visuelles de haute dimension.

Le kernel de référence est implémenté en Rust (ndarray), sans dépendances Python,
PyTorch ou CUDA. Tests validés : 51 (all-features) / 28 (default).

---

## 1. Introduction

La capacité des modèles de langage à grande échelle (LLMs) a progressé grâce à
les modèles denses présentent une caractéristique
limitative : la relation entre l'information et le calcul est statique. Un modèle dense
déploie la même quantité de ressources par token, qu'il traite un mot trivial ou une
équation complexe. Cette densité entraîne un coût énergétique élevé et rend
l'apprentissage continu vulnérable à l'oubli catastrophique.

Nous introduisons **TSO (Topographic Stabilization Operator)**, une architecture
événementielle où le calcul est déclenché par la nécessité de maintenir l'homéostasie
interne. La thèse fondatrice est que le calcul devient une conséquence d'une mesure
interne d'instabilité plutôt qu'une obligation liée à l'arrivée d'une donnée. Dans TSO,
l'activité neuronale émerge comme une réponse à une friction cognitive, et le système
ne calcule que lorsqu'une contradiction perturbe son état d'équilibre.

TSO est architecturé autour d'un **kernel unique** (Rust, ndarray). La validation
repose sur la navigation visuelle (MiniGrid, §6). Les principes de friction
topographique et de catégorisation par prototypes sont génériques et non spécifiques
à un domaine.

---

## 2. Travaux connexes

TSO s'inscrit à la confluence de plusieurs domaines :

**Calcul Adaptatif :** Des méthodes comme Adaptive Computation Time (ACT) ou
PonderNet visent à ajuster le calcul à la difficulté de l'entrée. TSO s'en distingue
par un déclenchement endogène (Φ) plutôt qu'une modulation de profondeur.

**Catégorisation par prototypes :** Les AttractorFields de TSO sont proches des
GLVQ (Generalized Learning Vector Quantization) et des cartes de Kohonen, mais
intégrés dans un cycle cognitif complet avec résolution de contraintes.

**Inférence Active :** Issus de Friston, ces cadres modélisent la cognition comme une
minimisation de la surprise. TSO s'en distingue en modélisant la friction comme
une contradiction structurelle interne plutôt qu'une erreur de prédiction externe.

---

## 3. Théorie de la Dissipation Cognitive (CDT)

Le fondement de TSO repose sur la modélisation des contradictions comme une grandeur
énergétique calculable.

**Définition 1 (État Cognitif).** L'état du système à l'instant t est
X_t = (G_t, S_t, W_t). G_t est le graphe sémantique (nœuds = concepts, arêtes =
contraintes), S_t l'activité neuronale, W_t les poids synaptiques.

**Définition 2 (Friction Φ).** La friction globale mesure la violation des contraintes
géométriques du graphe G_t. Pour deux vecteurs d'activation z_i, z_j ∈ R^d :

- Contrainte d'Implication (w_ij=1) : les vecteurs doivent être alignés.
  Violation = max(0, γ − ⟨z_i, z_j⟩)
- Contrainte d'Exclusion (w_ij=−1) : les vecteurs doivent être opposés.
  Violation = max(0, ⟨z_i, z_j⟩ − ε)

La friction globale : Φ(G_t) = Σ_{(i,j)∈E} Violation_ij.

**Dynamique.** L'évolution temporelle suit trois lois :
- G_{t+1} = G(G_t, Φ_t) — mise à jour topologique (resolve_with_anneal)
- S_{t+1} = f(S_t, I_t, W_t) — dynamique neuronale LIF (perceptual_belt)
- W_{t+1} = W_t + ΔW_t — plasticité locale (R-STDP, prévu)

### 3.1 Dual-LIF : mémoire multi-échelle

Le Dual-LIF utilise deux réservoirs LIF parallèles :

- **Mémoire lente** (α=0.9) : contexte global (sujet, agent, thème)
- **Mémoire rapide** (α=0.5) : syntaxe locale (2-3 derniers mots, négations)

Chaque mot met à jour les deux mémoires simultanément. Les features résultantes
(6D : cos, distance euclidienne, ratio des normes) remplacent les 3D du mono-LIF.

### 3.2 PerceptualBelt : pipeline de représentation unifié

Le PerceptualBelt est le pipeline d'entrée qui fusionne :
- Attention spatiale (gain par erreur de prédiction)
- Dual-LIF (intégration temporelle)
- AttractorField (catégorisation par prototypes)

Le belt est implémenté comme module distinct. L'intégration complète dans step()
est en cours.

### 3.3 Opérateurs cognitifs

Pour dissiper Φ, le système dispose d'opérateurs géométriques implémentés
dans le graphe via `Constraint::apply_to_graph()` :

- **Inversion** : z_i → −z_i (contradiction directe)
- **Alignement** : moyenne de deux vecteurs, normalisation sur la sphère unité
- **Répulsion** : gradient tangent projectif (séparation de nœuds)

---

## 4. Architecture TSO

### 4.1 Cycle cognitif

Le cycle heartbeat en 4 étapes :
1. **Catégorisation** : AttractorField → concept_id
2. **Action** : sélection par Cerebellum (TD(λ))
3. **Apprentissage** : reinforce_td (TD(λ) avec trace de eligibility)
4. **Φ gating** : résolution périodique du graphe (step % 50), skip quand Φ < threshold

Le Φ gating est implémenté avec compteurs de performance : `gating_skip_count`,
`resolve_count`, `resolve_total_iters` — mesurables dans les benchmarks.

### 4.2 Configuration minimale par défaut

Depuis la validation par ablation, la configuration par défaut (C13) active
uniquement les sous-systèmes validés :

| Sous-système | Défaut | Validation |
|-------------|--------|-----------|
| AttractorField | true | d=2.59 vs linéaire |
| Graphe Φ | true | Résolution de contraintes |
| Attention | false | Non validé sur MiniGrid |
| Mémoire épisodique | false | Non validé |
| Hypothalamus | false | Non validé |
| Coût métabolique | false | Non validé |

---

## 5. Implémentation Rust

Le kernel TSO est implémenté en Rust pur (crate `tso-engine`), sans dépendances
Python, PyTorch ou CUDA.

**Modules du kernel :**

| Module | Rôle | Feature gate |
|--------|------|-------------|
| `neurons.rs` | Clusters LIF + Dual-LIF | — |
| `core.rs` | Graphe Φ + `resolve_with_anneal` + contraintes | — |
| `attractor.rs` | AttractorField (prototypes compétitifs) | — |
| `cerebellum.rs` | Actor-critic TD(λ) + trace de eligibility | — |
| `tso_engine.rs` | Cycle cognitif principal + CognitiveConfig | — |
| `encoder.rs` | Trait Encoder unifié (AttractorEncoder) | — |
| `perceptual_belt.rs` | Pipeline de représentation unifié | — |
| `hypothalamus.rs` | Régulation homéostatique | `hypothalamus` |
| `episodic.rs` | Mémoire épisodique | `episodic` |
| `working_memory.rs` | Dual-LIF + mémoire associative | — |
| `attention.rs` | Attention spatiale | `attention` |
| `plasticity.rs` | Plasticité R-STDP | `rstdp` |
| `inference.rs` | Inférence variationnelle (FPI) | `active-inference` |
| `fpi.rs` + `efe.rs` | Active inference | `active-inference` |
| `grid_world.rs` / `grid_cells.rs` | Environnement + cellules de grille | — |
| `minigrid_env.rs` | MiniGrid Rust 7×7 147D | — |
| `replay_buffer.rs` | Buffer d'expérience pour TD | — |
| `neurogenesis.rs` | Naissance et maturation de concepts | — |

**Modules retirés (v0.2) :**
| Module | Raison |
|--------|--------|
| `vae.rs` | Ablation : d=0.02 vs attracteur seul (§6.1) |
| `VaeEncoder` | Idem — redondant avec AttractorField |
| `VaeStats` | Résidu documentaire dans `encoder.rs` — supprimé |

**Propriétés :**
- Zéro dépendance Python — compilation native
- Parcimonie explicite — les sous-systèmes inactifs sont ignorés
- Compilation conditionnelle : `active-inference` (FPI/EFE/inference), `rstdp` (plasticité R-STDP), `hypothalamus`, `episodic`, `attention`, `graph_phi` sont des Cargo features
- Validation des entrées : `step()` et `heartbeat_dt()` lèvent une assertion si `perception.len() != dim` ou contient des NaN
- `step()` refactorée en 9 sous-méthodes nommées (< 50 lignes chacune)
- Pas de rétropropagation globale — apprentissage local (TD + attracteur)
- 51 tests (all-features) / 28 tests (default)

**Binaires livrés :**
| Binaire | Usage |
|---------|-------|
| `demo` | Quick-start 30s : `cargo run --release --bin demo` |
| `demo --json` | Export structuré JSON des scores |
| `demo -- --attractor --graph_phi --threshold 0.3` | CLI complète pour CognitiveConfig |

---

## 6. Expériences

### 6.1 Benchmark MiniGrid — Navigation visuelle

Pour démontrer la généralité de TSO au-delà du NLP, un benchmark de navigation
visuelle sur MiniGrid DoorKey (grille 7×7, observation RGB 147D).

**Protocole :** 10 seeds, 100 épisodes, but tournant tous les 50 épisodes.

| Condition | Moyenne | σ | Δ vs linéaire |
|-----------|---------|---|---------------|
| Actor-critic linéaire (147D) | 1.03 | 0.17 |
| TSO + attracteur (147D) | 2.11 | 0.36 | +1.08 |
| TSO + attracteur + hypothal. | 2.06 | 0.34 | +1.03 |
| TSO + attracteur + épisodique | **2.13** | 0.34 | +1.10 |

TSO surpasse le linéaire de +105% en configuration attracteur. Le VAE a été
retiré après démonstration que son apport est négligeable (d=0.02, cf. ablation
ci-dessous).

**Ablation complète :** Chaque sous-système a été testé individuellement.
L'AttractorField est le seul composant avec un effet significatif (d=2.59).
Les autres (hypothalamus : d=0.12, épisodique : d=0.08, VAE : d=0.02) n'ont
pas justifié leur coût computationnel — ils sont désactivés par défaut (C13).

### 6.2 Analyse de dimension

Plus l'observation est riche, plus l'écart TSO vs linéaire se creuse :

| Dimension | Benchmark | Gain |
|-----------|-----------|------|
| 5D (whiskers) | Terrarium 7×7 | = (48.5% vs 49.5%) |
| 25D (grille) | GridWorld 5×5 | +24% |
| 147D (RGB) | MiniGrid 7×7 | +105% |

La catégorisation par prototypes devient rentable là où le nombre de dimensions
dépasse la capacité d'un mélange linéaire.

---

## 7. Discussion

TSO propose un changement de paradigme : passer d'une exécution systématique à
une cybernétique de survie active. En assujettissant le calcul à une friction
géométriquement calculable, TSO aligne l'efficacité computationnelle sur la
complexité réelle du problème.

L'ablation systématique a permis d'isoler l'AttractorField comme le seul composant
à impact significatif. Les autres sous-systèmes (VAE, hypothalamus, épisodique,
attention) sont conservés dans le code mais désactivés par défaut — disponibles
pour validation future sur des environnements plus complexes.

Le Φ gating (résolution conditionnelle du graphe) est instrumenté avec des
compteurs de performance : à chaque step où Φ < threshold, `gating_skip_count`
est incrémenté et la résolution est sautée, économisant O(|E| × iter) opérations.

---

## 8. Limites et travaux futurs

**Théorie encore floue (Φ → calcul).** Le lien entre friction Φ et déclenchement
effectif du calcul est défini qualitativement (Φ > seuil → stabilisation active),
mais pas quantifié : combien d'opérations sont économisées quand Φ est bas ?
Un modèle formel liant ||E||, sparsité α et nombre d'itérations de résolution
reste à établir.

**Φ gaging partiel.** Le gating par Φ ne contrôle que la résolution du graphe
(tous les 50 steps), pas le cycle complet. La promesse d'un calcul événementiel
intégral n'est que partiellement tenue.

**Échelle limitée.** MiniGrid 7×7 (147D RGB) est un test de principe.
Les benchmarks Procgen (64×64, 16 jeux) et Habitat (3D, ego-vision) sont
identifiés comme prochaines étapes mais non réalisés.

**Complexité O(|E|).** Le graphe Φ a une complexité O(|E|) par tick de
résolution. Le déminage et l'élagage maintiennent |E| borné sur les grilles
5×7, mais sur des environnements 3D (>10³ concepts) une parallélisation GPU
(wgpu, burn) ou une approximation parcimonieuse plus agressive sera nécessaire.

## 9. Conclusion

Nous proposons que la friction topographique (Φ) explore une direction où
le calcul est conditionné par une dynamique interne de stabilisation, et non
par une obligation liée au flux de données.
La validation sur MiniGrid (TSO 2.13 vs linéaire 1.03, +105%) confirme
que la catégorisation par prototypes et la friction topographique apportent
un avantage mesurable sur les entrées visuelles de haute dimension.
L'ablation systématique a isolé l'AttractorField comme le moteur principal
de cette performance, menant au retrait du VAE (d=0.02) et à la désactivation
des sous-systèmes non validés par défaut (C13).
## 10. Validation expérimentale — 17 scénarios

Le plan de test `e00-TEST_PLAN_TSO_CORE.md` couvre 17 scénarios en 4 phases
de risque. Résultats au 29 juillet 2026 :

| Phase | Scénarios | PASS | Note |
|-------|-----------|------|------|
| P0 — Cœur validé | 6 | 6/6 | AttractorField d=2.68 vs linear ; Φ gating non-dégradant |
| P1 — Stabilité Φ | 4 | 3/4 | 1 bench à calibrer (wall-clock) |
| P2 — Feature gates | 4 | 4/4 | hypothalamus, épisodique, compile gates OK |
| P3 — Scaling | 3 | 3/3 | TSO tient du 7×7 au 19×19 |
| **Total** | **17** | **16/17** | |

Résultat clé : **d de Cohen = 2.68** (A1 vs A0, bench_ablation), confirmant
que l'AttractorField seul explique la performance TSO — cohérent avec
l'ablation qui a mené au retrait du VAE (d=0.02).

Le kernel Rust, les 51 tests (all-features), et les benchmarks reproductibles sont
disponibles en open source.
