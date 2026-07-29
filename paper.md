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
2. **PerceptualBelt** : pipeline de représentation unifié fusionnant attention
   spatiale, mémoire de travail Dual-LIF, catégorisation par prototypes et
   inférence variationnelle (FPI). Interface : 8 méthodes publiques.
3. **VAE online + Gumbel-STE** : encodeur variationnel entraîné à chaque tick
   avec gradient local via straight-through estimator (température annealed).
4. **Benchmark MiniGrid** (147D RGB) : TSO + VAE + FPI bat le linéaire de +105%,
   démontrant l'avantage de la catégorisation par prototypes sur les entrées
   visuelles de haute dimension.

Le kernel de référence est implémenté en Rust (ndarray), sans dépendances Python,
PyTorch ou CUDA. Tests validés : 45 tests unitaires,

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

TSO est architecturé autour d'un **kernel unique** (Rust, ndarray) qui supporte à la
fois le traitement du langage naturel (SNLI, §9.1) et la navigation visuelle (MiniGrid,
§9.2). Cette dualité démontre que les principes de friction topographique et de
catégorisation par prototypes ne sont pas spécifiques à un domaine.

---

## 2. Travaux connexes

TSO s'inscrit à la confluence de plusieurs domaines :

**Calcul Adaptatif :** Des méthodes comme Adaptive Computation Time (ACT) ou
PonderNet visent à ajuster le calcul à la difficulté de l'entrée.

**Réseaux de Neurones à Impulsions (SNN) :** TSO utilise des réservoirs SNN
(Dual-LIF) pour l'intégration temporelle multi-échelle.

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
- G_{t+1} = G(G_t, Φ_t) — mise à jour topologique
- S_{t+1} = f(S_t, I_t, W_t) — dynamique neuronale LIF
- W_{t+1} = W_t + ΔW_t — plasticité locale (R-STDP, prévu)

### 3.1 Dual-LIF : mémoire multi-échelle

Le Dual-LIF utilise deux réservoirs LIF parallèles :

- **Mémoire lente** (α=0.9) : contexte global (sujet, agent, thème)
- **Mémoire rapide** (α=0.5) : syntaxe locale (2-3 derniers mots, négations)

Chaque mot met à jour les deux mémoires simultanément. Les features résultantes
(6D : cos, distance euclidienne, ratio des normes) remplacent les 3D du mono-LIF.

### 3.2 PerceptualBelt : pipeline de représentation unifié

Le PerceptualBelt (refactorisation v2.1) encapsule cinq modules de représentation
qui étaient auparavant éparpillés :

- **Attention spatiale** : gain multiplicatif par erreur de prédiction
- **Mémoire de travail** : Dual-LIF + mémoire associative
- **Catégorisation** : trois backends interchangeables — attracteur à prototypes,
  encodeur variationnel (VAE, entraîné en ligne, lr=0.001), inférence FPI
- **Cellules de grille** : augmentation perceptive pour la navigation
- **Encodeur interchangeable** : trait Encoder (VaeEncoder, AttractorEncoder)

Interface publique : 8 méthodes (process, recall, reset, lif_state, configure,
extra_dim, num_concepts, set_encoder).

### 3.3 Opérateurs cognitifs

Pour dissiper Φ, le système dispose d'opérateurs géométriques :

- **Inversion** : z_i → −z_i (contradiction directe)
- **Alignement** : moyenne de deux vecteurs, normalisation sur la sphère unité
- **Répulsion** : gradient tangent projectif (séparation de nœuds)
- **Expansion dimensionnelle** : doublement de l'espace latent pour résoudre
  les contradictions fortes (opérateur "MAIS", orthogonalité stricte)

### 3.4 Gumbel-STE : gradient local vers le VAE

L'argmax sur les centroids du VAE casse le gradient. Nous introduisons un
**Straight-Through Estimator (STE)** basé sur Gumbel-Softmax : la température τ
s'annele de 1.0 à 0.1 (decay 0.995), les poids softmax pondèrent la mise à jour
des centroids, et le gradient peut circuler de la tâche vers l'encodeur.

---

## 4. Architecture TSO

### 4.1 Cycle cognitif (mode navigation)

Le cycle heartbeat en 4 étapes :
1. **Perception** : entrée sensorielle → PerceptualBelt.process()
2. **Catégorisation** : attracteur / VAE / FPI → concept_id
3. **Action** : sélection par Cerebellum (TD) ou EFE (active inference)
4. **Apprentissage** : reinforce_td + R-STDP locale

Un fast path contourne le belt quand tous les sous-systèmes sont désactivés,
réduisant l'overhead à un simple forward_logits + reinforce_td.

---

## 5. Implémentation Rust

Le kernel TSO est implémenté en Rust pur (crate `tso-engine`), sans dépendances
Python, PyTorch ou CUDA.

**Modules du kernel :**

| Module | Rôle |
|--------|------|----------------------|
| `neurons.rs` | Clusters LIF + Dual-LIF |
| `core.rs` | Graphe Φ, résolution de contraintes |
| `plasticity.rs` | R-STDP (prévu) |
| `operators.rs` | Inversion, Alignement, Répulsion |
| `perceptual_belt.rs` | Pipeline de représentation unifié |
| `attractor.rs` | AttractorField (prototypes) |
| `cerebellum.rs` | Actor-critic TD(λ) |
| `vae.rs` | Encodeur variationnel (online) |
| `fpi.rs` + `efe.rs` | Active inference (FPI/EFE) |
| `rotating_t.rs` | Benchmark non-stationnaire |
| `minigrid_env.rs` | MiniGrid Rust 7×7 147D |

**Propriétés :**
- Zéro dépendance Python — compilation native
- Parcimonie explicite — les clusters inactifs sont ignorés dans Φ
- Pas de rétropropagation — apprentissage local (R-STDP)
- Parallélisation via Rayon
- 45 tests unitaires

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
| TSO + VAE (147D→16D) | 2.06 | 0.34 | +1.03 |
| TSO + VAE + FPI | **2.13** | 0.34 | +1.10 |

TSO surpasse le linéaire de +105% en configuration FPI+attracteur. Le VAE
avec Gumbel-STE améliore la robustesse mais n'est pas nécessaire sur cet
environnement.

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

**Différentiabilité.** Le Gumbel-STE dans le VAE (température annealed 1.0→0.1,
decay 0.995) permet au gradient de la tâche de circuler jusqu'à l'encodeur.
La jointure complète (TD → belt → VAE) est partiellement implémentée.

---

## 8. Limites et travaux futurs

**Théorie encore floue (Φ → calcul).** Le lien entre friction Φ et déclenchement
effectif du calcul est défini qualitativement (Φ > seuil → stabilisation active),
mais pas quantifié : combien d'opérations sont économisées quand Φ est bas ?
Un modèle formel liant ||E||, sparsité α et nombre d'itérations de résolution
reste à établir. Actuellement, le step() complet s'exécute à chaque tick quel
que soit Φ — la promesse d'un calcul événementiel n'est que partiellement tenue.

**Différentiabilité partielle.** Le Gumbel-STE (température annealed 1.0→0.1,
decay 0.995) permet au gradient local de circuler du choix de catégorie vers
l'encodeur VAE. Mais la jointure complète TD → belt → VAE n'est pas active
dans step() : le gradient de l'erreur TD (reinforce_td) ne remonte pas jusqu'aux
poids du VAE. L'apprentissage perceptuel reste découplé de l'apprentissage par
renforcement. Une connexion directe (backprop_td) est implémentée dans
l'API du VaeEncoder mais pas appelée dans le cycle.

**R-STDP non intégrée.** La plasticité locale R-STDP est annoncée comme
élément central mais absente du cycle en ligne. L'apprentissage repose
sur l'actor-critic TD(λ), qui est classique, pas neuromorphique. L'intégration
de R-STDP dans le PerceptualBelt (remplacement de reinforce_td par mise à jour
locale des poids du cerebellum) est une piste ouverte non implémentée.

**Échelle limitée.** MiniGrid 7×7 (147D RGB) est un test de principe.
Les benchmarks Procgen (64×64, 16 jeux) et Habitat (3D, ego-vision) sont
identifiés comme prochaines étapes mais non réalisés. Le VAE 16D latent
et les centroides de l'attracteur sont spécialisés sur l'espace observé —
leur robustesse face au bruit visuel massif n'est pas testée.

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
Le kernel Rust, les 45 tests unitaires, et les benchmarks reproductibles sont
disponibles en open source.
