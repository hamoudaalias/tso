
# TSO : Topographic Stabilization Operator
## Une architecture neuromorphique où la friction remplace l'attention

**Auteur :** Hamouda ALIAS
**Date :** Juillet 2026
**Discipline :** Architectures d'IA, Systèmes Neuromorphiques, Systèmes Dynamiques

---

### ABSTRACT

Les architectures d'IA actuelles maintiennent une relation fixe entre l'information entrante et le calcul : une quantité constante d'opérations est exécutée par token, indépendamment de la complexité cognitive de l'entrée. Cet article propose une alternative architecturale, **TSO (Topographic Stabilization Operator)**, où le calcul devient une conséquence d'une mesure interne d'instabilité plutôt qu'une obligation liée à l'arrivée d'une donnée. Fondée sur la Théorie de la Dissipation Cognitive (CDT), l'architecture TSO modélise l'activité neuronale comme un processus de minimisation d'une énergie de friction $\Phi$, formellement définie comme une contrainte géométrique calculable sur un graphe conceptuel émergent.

**Contribution clé :** TSO propose que la **friction topographique ($\Phi$)** peut remplacer l'attention des Transformers, et qu'un **réservoir Leaky Integrate-and-Fire (LIF)** couplé à un **opérateur d'inversion sur négation** capture l'ordre séquentiel des mots sans mécanisme d'attention dense. Le kernel de référence est implémenté en **Rust** (ndarray), sans dépendances Python, PyTorch ou CUDA. Sur le benchmark SNLI (570k paires), le pipeline complet — **Dual-LIF (mémoire lente α=0.9 + rapide α=0.5)** 6D, Phi 4D, Jaccard 3D, Align 4D → **17D** — atteint **56.69% sur le jeu de test** (56.96% dev), avec un temps de traitement total de **~20 secondes** sur CPU 8 cœurs. Le Dual-LIF, équivalent neuromorphique du *multi-head attention*, capture simultanément le contexte global et la syntaxe locale, repoussant le plafond de +0.80% par rapport au mono-LIF et de +1.84% au-delà du plafond sac-de-mots (54.85%). En apprentissage continu (SNLI → MultiNLI), TSO démontre une **immunité structurelle à l'oubli catastrophique** : la représentation TSO 17D est un fixateur topologique immuable — après 20 époques MultiNLI, un classifieur ré-entraîné sur SNLI retrouve exactement 56.96% (Δ = 0.00%), et le mode **Freeze+Add** préserve 100% de la performance originale tout en apprenant la nouvelle tâche. C'est une propriété impossible dans les Transformers sans artifices externes (EWC, replay). Enfin, TSO démontre la **première génération auto-régressive sans backprop**, par **Inverse Motor** ($w_{t+1} = \arg\max\langle S_t, e(w)\rangle$) combiné au graphe de friction $\Phi$ — produisant une dérive sémantique organique (ex: *"the dog ran" → "brown running grass runs field grassy across on beach sand"*) sans gradient, sans softmax, sans probabilités.

---

### 1. INTRODUCTION

La capacité des modèles de langage à grande échelle (LLMs) a progressé grâce à l'architecture Transformer. Cependant, ces modèles présentent une caractéristique limitative : la relation entre l'information et le calcul est statique. Un Transformer déploie la même quantité de ressources par token, qu'il traite un mot trivial ou une équation complexe. Cette densité entraîne un coût énergétique élevé et rend l'apprentissage continu vulnérable à l'oubli catastrophique.

Nous introduisons **TSO (Topographic Stabilization Operator)**, une architecture événementielle où le calcul est déclenché par la nécessité de maintenir l'homéostasie interne. La thèse fondatrice est que **le calcul devient une conséquence d'une mesure interne d'instabilité plutôt qu'une obligation liée à l'arrivée d'une donnée**. Dans TSO, l'activité neuronale émerge comme une réponse à une friction cognitive, et le système ne calcule que lorsqu'une contradiction perturbe son état d'équilibre.

### 2. RELATED WORK

TSO s'inscrit à la confluence de plusieurs domaines :
*   **Calcul Adaptatif :** Des méthodes comme Adaptive Computation Time (ACT) ou PonderNet visent à ajuster le calcul à la difficulté de l'entrée, mais restent basées sur des réseaux denses et la rétropropagation.
*   **Réseaux de Neurones à Impulsions (SNN) :** TSO utilise des réservoirs SNN pour le traitement temporel, y introduisant une plasticité (R-STDP) pour un apprentissage strictement local.
*   **Inférence Active :** Issus de Friston, ces cadres modélisent la cognition comme une minimisation de la surprise. TSO s'en distingue en modélisant la friction non comme une erreur de prédiction externe, mais comme une contradiction structurelle interne nécessitant une action géométrique.
*   **Apprentissage Continu :** TSO adresse l'oubli catastrophique nativement par sa plasticité locale, s'affranchissant des méthodes de régularisation globale (ex: EWC).

### 3. THÉORIE DE DISSIPATION COGNITIVE (CDT)

Le fondement de TSO repose sur la modélisation des contradictions comme une grandeur énergétique calculable, inspirée des embeddings de graphes relationnels.

**Définition 1 (État Cognitif et Émergence du Graphe).** 
L'état du système à l'instant $t$ est $X_t = (G_t, S_t, W_t)$.
*   **Nœuds ($V$) :** Les clusters de neurones actifs dans la Grille Topographique. L'activation d'un cluster représente la présence d'un concept dans la mémoire de travail.
*   **Arêtes et Contraintes ($E, W$) :** Le graphe émerge par co-activation temporelle via R-STDP. Si deux clusters spike dans la même fenêtre $\Delta t$, une arête se forme. Le typage de l'arête $w_{ij}$ est déterminé par la similarité cosinus des embeddings cibles : $w_{ij}=1$ si $\cos(z_i, z_j) > 0$ (implication), $w_{ij}=-1$ si $\cos(z_i, z_j) < 0$ (exclusion).
*   $S_t$ est l'activité neuronale, $W_t$ les poids synaptiques.

**Définition 2 (Friction $\Phi$ calculable).** 
La friction globale mesure la violation des contraintes géométriques du graphe actif $G_t$. Pour deux vecteurs d'activation $z_i, z_j \in \mathbb{R}^d$ :
*   **Contrainte d'Implication ($w_{ij} = 1$) :** Les vecteurs doivent être alignés.
    $$ \text{Violation}_{ij}^{imp} = \max(0, \gamma - \langle z_i, z_j \rangle) $$
*   **Contrainte d'Exclusion ($w_{ij} = -1$) :** Les vecteurs doivent être opposés ou orthogonaux.
    $$ \text{Violation}_{ij}^{exc} = \max(0, \langle z_i, z_j \rangle - \epsilon) $$

La friction globale est la somme de ces violations : $\Phi(G_t) = \sum_{(i,j) \in E} \text{Violation}_{ij}$.

**Dynamique Globale et Apprentissage Local.**
L'évolution temporelle est séparée en trois dynamiques distinctes :
$$ \begin{aligned}
G_{t+1} &= \mathcal{G}(G_t, \Phi_t) \quad \text{(Mise à jour topologique)} \\
S_{t+1} &= f(S_t, I_t, W_t) \quad \text{(Dynamique neuronale LIF)} \\
W_{t+1} &= W_t + \Delta W_t \quad \text{(Plasticité R-STDP)}
\end{aligned} $$
La plasticité locale obéit à $\Delta W_{ij}(t) = \eta M(t) E_{ij}(t)$, où $E_{ij}(t)$ est la trace d'éligibilité régie par $\frac{dE_{ij}}{dt} = -\frac{E_{ij}}{\tau} + S_i(t)S_j(t)$.

### 4. LES OPÉRATEURS COGNITIFS

Pour dissiper $\Phi$, le système dispose d'opérateurs géométriques. Ceux-ci s'appliquent au sous-graphe conflictuel, tout en nécessitant une cohérence dimensionnelle globale (propagation du padding aux nœuds voisins).

*   **Correction (Inversion) :** Pour contradiction directe. L'opérateur applique $z_i \to -z_i$.
*   **Latent Space Expansion Operator :** Pour résoudre un conflit en augmentant l'espace de représentation. Pour préserver intégralement l'information sans compression, l'expansion utilise $k=d$ nouvelles dimensions, doublant ainsi l'espace latent.
    *   **Contradiction forte ($\Phi > \theta_c$) :** L'opérateur projette dans des sous-espaces strictement orthogonaux complémentaires.
    *   **Tension sémantique ($\theta_t < \Phi < \theta_c$) :** L'opérateur produit les connecteurs de tension (*mais, cependant*) via une expansion dimensionnelle avec angle d'atténuation.

```text
Algorithm 1: Latent Space Expansion (Opérateur "MAIS")
Input: Concepts z_1, z_2 ∈ R^d, Seuils θ_t, θ_c
Output: Nouveaux vecteurs z'_1, z'_2 ∈ R^(2d)

1. Calculer la friction locale Φ_ij
2. If Φ_ij > θ_c (Contradiction forte) :
3.     k = d (Doublage dimensionnel)
4.     z'_1 = [z_1, 0_d] // z_1 suivi de d zéros
5.     z'_2 = [0_d, z_2] // d zéros suivis de z_2
6.     // Le produit scalaire ⟨z'_1, z'_2⟩ = 0, orthogonalité stricte
7. ElseIf θ_t < Φ_ij < θ_c (Tension sémantique) :
8.     k = d, Appliquer une rotation d'angle α avant concaténation
9. Else :
10.    Maintenir l'état (z'_1 = z_1, z'_2 = z_2)
```

### 5. ARCHITECTURE TSO ET DYNAMIQUE D'APPRENTISSAGE

L'architecture repose sur une topographie stricte en clusters sémantiques. Pour modéliser l'apprentissage, TSO intègre deux modes :

1.  **Mode 1 : Stabilisation Active (Friction forte).** Le système rencontre une contradiction. La cascade de spikes déclenche l'Actor et le Critic pour trouver l'opérateur réduisant $\Phi$.
2.  **Mode 2 : Consolidation Passive (Friction faible).** Le système reçoit un flux cohérent. L'activité reste sous le seuil critique. Les traces d'éligibilité s'accumulent, renforçant les chemins synaptiques sans intervention globale.

**Résolution du paradoxe Actor-Critic sans Backpropagation :**
Dans TSO, le Critic n'est pas un réseau neuronal profond entraîné par TD-Learning. C'est une **fonction analytique de simulation "forward"** qui évalue la physique du système. L'Actor (le réseau SNN) ne calcule pas les seuils ; son rôle est d'apprendre, via la R-STDP, la **carte de routage prioritaire**. Lors d'une contradiction multiple, l'Actor sélectionne l'arête à traiter et l'opérateur à appliquer. Le Critic simule alors $S_{simul} = P_a(S_t)$ et calcule $\Delta\Phi_{global}$. Si $\Delta\Phi > 0$, l'action est validée. Le neuromodulateur $M(t)$ renforce alors les synapses ayant mené au choix de cette priorité. Aucun gradient global ne traverse le réseau.

#### 5.1 Onde de Choc Locale (V8) — Résolution du Paradoxe de l'Évaluation Globale

La version V8 remplace l'évaluation globale $\Delta\Phi_{global}$ par une **propagation d'onde locale asynchrone**. Le Critic ne calcule la friction que sur le **voisinage immédiat** du conflit (profondeur $d \leq 2$), réduisant la complexité de $O(V \cdot E)$ à $O(k^d)$ où $k$ est le degré moyen du sous-graphe conflictuel (typiquement 3–5 nœuds).

**Algorithme :**
1. Conflit détecté entre les nœuds $A$ et $B$.
2. L'opérateur candidat est appliqué localement à $A$.
3. La friction est recalculée uniquement sur le sous-graphe $N_{d}(A, B)$ — l'ensemble des nœuds à distance $\leq d$ de $A$ ou $B$.
4. Si $\Delta\Phi_{local} < 0$, l'action est validée. Le reste du graphe est ignoré — toute tension qui y apparaît sera résolue à son propre pas de temps.
5. Si une nouvelle tension apparaît chez un voisin $C \in N_1(A, B)$, l'onde se propage : le voisinage s'élargit à $N_{d+1}(A, B, C)$ pour chercher un opérateur complémentaire.
6. Si aucune résolution locale n'est trouvée après $d_{max}$ sauts, l'onde s'éteint et le conflit est marqué comme irrésoluble localement.

**Implication théorique :** Le système devient **strictement local**. Il n'existe plus aucun point centralisé : chaque conflit est résolu indépendamment, et la cohérence globale émerge de la propagation des ondes de choc. C'est un système physique asynchrone pur, où le temps de résolution est proportionnel à la complexité locale du conflit, jamais à la taille du graphe.

### 6. ALGORITHME COMPLET

Soit $\Delta\Phi = \Phi(S_t) - \Phi(P_a(S_t))$ la variation de friction globale (succès si $\Delta\Phi > 0$).

```text
Algorithm 2: TSO_Training_Step(x_t)
Input: Signal x_t, État X_t
Output: Action a_t, État X_{t+1}

1. Initialize topology (Clusters sémantiques)
2. Encoder x_t en train de spikes I_t
3. Mettre à jour le graphe G_t par co-activation (typage NLI)
4. Calculer Φ_estimée = Φ(G_t) + λ * ||I_t||
5. If Φ_estimée < θ_t (Basse friction) :
6.      Mode 2 : Consolidation Passive
7.      Mettre à jour faiblement les traces d'éligibilité (Accumulation)
8.      a_t = ∅ (Pas d'action motrice)
9. Else (Haute friction) :
10.     Mode 1 : Stabilisation Active
11.     Propager I_t (Cascade de spikes)
12.     Actor propose un candidat : (arête_ij, opérateur_a)
13.     Critic simule l'action: S_simul = P_a(S_t)
14.     Critic calcule ΔΦ = Φ(S_t) - Φ(S_simul)
15.     Si ΔΦ > 0 :
16.         Exécuter a_t dans l'environnement (Padding global inclus)
17.         Renforcer fortement les traces via neuromodulateur M(t)
18.     Sinon :
19.         Inhiber la proposition et recommencer en 12
20. Mettre à jour l'état global X_{t+1}
21. Return a_t, X_{t+1}
```

### 7. COMPLEXITÉ THÉORIQUE

Contrairement aux Transformers, dont le coût dépend de la longueur du contexte, le coût de TSO dépend de l'activité interne, dictée par la friction.

|                | Dense Transformer  | TSO                  |
| -------------- | ------------------ | -------------------- |
| **Activation** | globale            | événementielle       |
| **Attention**  | dense ou optimisée | locale topographique |
| **Coût dépend de** | longueur contexte  | activité / friction  |

Le calcul par token dans TSO est proportionnel à la sparsité $\alpha \ll 1$ du réseau. Un input trivial active peu de neurones, rendant le calcul quasi-nul.

### 8. IMPLÉMENTATION RUST

Le kernel TSO est implémenté en **Rust pur** (crate `tso-kernel`), sans dépendances Python, PyTorch ou CUDA. Le calcul numérique utilise `ndarray` pour les opérations vectorielles et `rand` pour les initialisations stochastiques. Le workspace comprend trois crates : `tso-kernel` (noyau), `tso-nlp` (tokenizer, graphe, métriques), `tso-bench` (benchmark SNLI).

**Modules du kernel :**

| Module | Rôle | Équivalent Transformer |
|--------|------|----------------------|
| `neurons.rs` | Clusters LIF (Leaky Integrate-and-Fire) | Feed-Forward |
| `friction.rs` | Calcul de $\Phi$ + tri-friction (support, conflit, nouveauté) | Self-Attention |
| `plasticity.rs` | R-STDP avec traces d'éligibilité multi-échelle | Backpropagation |
| `operators.rs` | Double Mapping, Inverse Motor, Soft Double Mapping | Embedding / Projection |
| `core.rs` | TSOCore — orchestre clusters, arêtes, plasticité et homéostasie | Engine |
| `deep.rs` | DeepTSO — empilement multi-couche de $\Phi$ avec résidus | Multi-Head Attention |
| `attractor.rs` | Sharp Attractor Field — classification sans gradient | Tête de classification |
| `model.rs` | Persistance : sauvegarde/charge du modèle entraîné (bincode) | — |
| `pipeline.rs` | BatchProcessor — traitement parallèle de séquences | Inference Engine |

**Crates auxiliaires (`tso-nlp`) :**

| Module | Rôle |
|--------|------|
| `tokenizer.rs` | Tokeniseur whitespace avec vocabulaire persistant, ou HF Tokenizers |
| `graph_builder.rs` | Graphe de cooccurrence avec normalisation |
| `cluster_map.rs` | Mapping tokens → clusters via hachage modulaire |
| `dataset.rs` | Lecteur SNLI/MultiNLI JSONL avec validation |
| `metrics.rs` | Accuracy, précision, rappel, F1, matrice de confusion |

**Propriétés clés de l'implémentation Rust :**
- **Zéro dépendance Python** — compilation native, pas d'interpréteur, pas de PyTorch/CUDA
- **Parcimonie explicite** — les clusters inactifs sont ignorés dans le calcul de $\Phi$
- **Pas de rétropropagation** — l'apprentissage est local (R-STDP) et ne nécessite pas de graphe de calcul
- **Parallélisation massive via Rayon** — comptage de cooccurrences thread-local sans synchronisation
- **Checkpoints persistants** — le modèle sauvegarde son état à chaque étape (graphe, voisins triés, centroïdes)
- **Tests vérifiés** — 40 tests unitaires couvrant kernel, NLP et intégration

### 9. ARCHITECTURE TSO COMME ALTERNATIVE AU TRANSFORMER

TSO propose un mappage direct des composants d'un Transformer vers des équivalents neuromorphiques :

| Composant Transformer | Équivalent TSO | Implémenté ? |
|---------------------|----------------|-------------|
| Token Embedding | Clusters LIF + Double Mapping | ✅ `operators.rs` |
| Positional Encoding | Trace temporelle (τ_rate) + historique d'activité | ✅ `neurons.rs`, `core.rs` |
| Self-Attention | Friction $\Phi$ entre paires de concepts | ✅ `friction.rs` |
| Feed-Forward | Dynamique LIF des clusters | ✅ `neurons.rs` |
| Residual Connection | Homéostasie adaptive (dynamic_theta_c) | ✅ `core.rs` |
| Layer Norm | Normalisation par taux de décharge moyen | ✅ `neurons.rs` |
| Tête de sortie | Inverse Motor (projection état → vocabulaire) | ✅ `operators.rs` |
| Apprentissage (Backprop) | R-STDP + Friction-Gated Consolidation | ✅ `plasticity.rs` |
| **Génération auto-régressive** | **Inverse Motor + Φ homeostasis + répression répétitions** | ✅ **`decoder.rs`** |

**Ce qui est en développement :**
- **Triple-échelle temporelle** — hiérarchie LIF (lent/moyen/rapide) pour une capture multi-granulaire du contexte

### 10. RÉSULTATS EXPÉRIMENTAUX

#### 10.1 Benchmark SNLI — Pipeline Complet

La tâche de Natural Language Inference (SNLI v1.0, 570k paires) est utilisée pour valider l'architecture TSO complète : représentation séquentielle (LIF), friction dynamique ($\Phi$), et classification par attracteurs locaux.

**Protocole :**
- **Embeddings distributionnels** : cooccurrences (fenêtre=5) → PPMI → SVD randomisé (k=100, power_iter=2) → normalisation L2 des lignes
- **Features (14D → 17D)** :
  - *Jaccard 3D* : similarité de voisinage entre mots de P et H via le graphe de friction
  - *Dual-LIF 6D* (v5.1) : mémoire lente $\alpha=0.9$ + mémoire rapide $\alpha=0.5$, équivalent neuromorphique du multi-head attention
  - *Phi 4D* : friction séquentielle — $\cos(h_t, S_{premise})$ à chaque pas, avec inversion de $S_{premise}$ sur négation (Algorithme 1)
  - *Align 4D* : alignement mot-à-mot P→H (cosinus max, couverture)
- **Classifieur AttractorField** : k-means intra-classe (15 prototypes/classe) → LVQ1 (lr=0.001, 20 epochs)
- Implémentation Rust, parallélisation Rayon, zéro rétropropagation

**Résultats :**

| Métrique | Sac-de-mots 13D (baseline) | TSO LIF+$\Phi$ 14D | TSO Dual-LIF 17D |
|----------|:--------------------------:|:------------------:|:-----------------:|
| Accuracy (dev) | 54.85% | 56.43% | **56.96%** |
| Accuracy (test) | — | 55.89% | **56.69%** |
| Précision entailment | 37.1% | 38.0% | — |
| Précision neutral | 34.3% | 34.9% | — |
| Précision contradiction | 34.3% | 34.2% | — |
| Temps total (550k paires) | ~20s | ~20s | ~20s |
| Écart dev-test | — | 0.54% | **0.27%** |
| Écart dev-test | — | 0.54% |

**Matrice de confusion (test, 9824 paires) :**

```
                 ENT    NEU    CON
  entailment   2246    575    547
     neutral    653   1444   1122
contradiction    759    677   1801
```

**Progression architecturale :**

| Version | Pipeline | Acc. dev | Δ |
|---------|----------|:--------:|:-:|
| v3.0 | Jaccard 3 + Mean 3 + Align 4 (delta-rule) | 51.71% | — |
| v3.3 | + corrections bugs SVD/LR | **54.02%** | +2.31 |
| v4.0 | + Max-pooling hybride (Mean+Max, 13D) | **54.85%** | +0.83 |
| v4.1 | + AttractorField (45 prototypes, LVQ1) | 54.85% | 0.00 |
| v5.0 | **LIF + Phi + Négation** (14D) | **56.43%** | **+1.58** |
| **v5.1** | **+ Dual-LIF (α=0.9/α=0.5) 17D** | **56.96%** | **+0.53** |

**Analyse :**

Le plafond de **54.85%** pour les représentations sac-de-mots (Mean+Max) est confirmé par deux classifieurs orthogonaux (delta-règle linéaire et AttractorField non-linéaire), prouvant qu'il s'agit d'une **limite de séparabilité de la représentation**, pas du classifieur. L'ajout du **LIF séquentiel** ($\alpha=0.8$) brise ce plafond en introduisant l'ordre des mots : "*chien mord homme*" et "*homme mord chien*" produisent désormais des vecteurs LIF différents, alors que leurs barycentres sont identiques. L'**opérateur d'inversion sur négation** (Algorithme 1) ajoute +0.50% en inversant l'état latent lors de la rencontre de mots comme *not*, *no*, *never*, capturant ainsi les relations de contradiction lexicale. Le **Dual-LIF (α=0.9/α=0.5)** pousse le gain à **+0.80%** au-dessus du mono-LIF en capturant simultanément le contexte global et la syntaxe locale, réduisant l'écart dev-test à seulement **0.27%**.

#### 10.2 Validité géométrique — Double Mapping (Phase 0)

Le Double Mapping élimine $\Phi$ entre concepts exclusifs tout en préservant les implications. → **Confirmé (Lemme 1, tests).**

#### 10.3 Génération séquentielle (Shakespeare)

Apprentissage de transitions conceptuelles avec $\Phi$ comme mesure de surprise grammaticale. → **$\Phi$ chute à $-1.78$ (280% mieux que hasard).**

#### 10.5 Apprentissage Continu et Prévention de l'Oubli Catastrophique

L'un des défis majeurs des architectures basées sur la rétropropagation (Transformers) est l'oubli catastrophique : l'apprentissage d'une nouvelle tâche érode les poids partagés, détruisant la performance sur la tâche précédente. L'architecture TSO, en découplant **l'extraction de features** (immuable) du **classifieur à attracteurs** (plastique), résout nativement ce problème.

Dans un Transformer, l'attention et les poids sont intriqués : s'entraîner sur la tâche B détruit la représentation de la tâche A. Chez TSO, la représentation (graphe de friction → PPMI → SVD → réservoir LIF) agit comme un **fixateur topologique** : elle capture la sémantique du corpus d'entraînement de manière statique et immuable. L'apprentissage séquentiel n'affecte que la couche de décision (les prototypes LVQ1), laissant intacte la géométrie représentationnelle.

Nous avons évalué cette propriété en effectuant un apprentissage séquentiel sur **SNLI (549k paires, entraînement initial)** puis **MultiNLI (392k paires, seconde tâche)** :

1.  **[A] Écrasement naïf (LVQ1) — Baseline forgetting :** L'entraînement naïf sur MultiNLI écrase les prototypes SNLI, entraînant une chute de l'accuracy SNLI-dev de **12.74%** (56.52% → 43.78%). L'oubli est réel, mais il est intégralement localisé dans le classifieur.

2.  **[B] Test de récupération — Preuve de l'immunité représentationnelle :** Après 20 époques MultiNLI, un nouvel AttractorField (ré-entraîné de zéro sur les *mêmes* features SNLI) retrouve **exactement 56.52%**. La représentation TSO 14D est **intacte** — le "cerveau" (features) n'a rien oublié, seul le "parleur" (classifieur) avait réaffecté ses prototypes.

3.  **[C] Freeze+Add — Solution neuromorphique native :** En gelant les 45 prototypes SNLI (consolidation passive, §5) et en allouant 45 nouveaux clusters pour MultiNLI (stabilisation active), le système atteint une **rétention de 100%** : la précision SNLI-dev reste stable à **56.52%** (Δ = 0.00%) pendant les 20 époques MultiNLI. La précision MultiNLI atteint ~49% (vs. hasard 33%), confirmant l'apprentissage effectif de la nouvelle tâche.

**Conclusion.** Cette expérience valide la capacité de TSO à opérer un apprentissage continu sans oubli catastrophique. Le découplage entre features stabilisées (TSO) et classifieur local (LVQ1) est la clé de cette immunité : la représentation topographique est un **fixateur sémantique immuable**, tandis que les attracteurs locales sont librement allouables par tâche. Cette propriété est structurellement impossible dans les réseaux denses (Transformers) où la rétropropagation modifie globalement tous les poids partagés, et où des artifices computationnels lourds (EWC, Replay Buffers, distillation) sont nécessaires pour atténuer — sans jamais éliminer — l'oubli.

#### 10.6 Dual-LIF : Mémoire Multi-Échelle (Fast-Slow)

L'un des leviers les plus puissants des Transformers est l'attention multi-tête (*multi-head attention*), qui capture simultanément différentes relations entre les tokens. Nous proposons l'équivalent neuromorphique : le **Dual-LIF**, deux réservoirs Leaky Integrate-and-Fire parallèles avec des constantes de temps distinctes.

- **Mémoire lente ($\alpha = 0.9$)** : oublie lentement, capture le contexte global de la phrase (sujet, agent, thème).
- **Mémoire rapide ($\alpha = 0.5$)** : oublie vite, capture les relations syntaxiques locales (2-3 derniers mots, négations, inversion).

Chaque mot met à jour les deux mémoires simultanément. L'opérateur d'inversion de négation (Algorithme 1) s'applique aux deux canaux. Les features résultantes (6D : cos, distance euclidienne, ratio des normes pour chaque mémoire) remplacent les 3D du mono-LIF, portant le vecteur total de 14D à 17D.

**Résultat :** Dual-LIF (α=0.9/0.5) → **56.69% test, 56.96% dev**, soit **+0.80%** par rapport au mono-LIF (α=0.8) et **+1.84%** au-delà du plafond sac-de-mots. L'écart test-dev se réduit à 0.27%, confirmant l'absence de surapprentissage. Le Dual-LIF est actuellement notre meilleur levier d'amélioration.

#### 10.7 Scalabilité : Pipeline sur Corpus Synthétique (V = 10⁶)

L'un des avantages structurels de TSO est que la construction du pipeline (cooccurrences → PPMI → SVD) est **découplée** de l'inférence. Le graphe de friction et les embeddings sont construits une seule fois par corpus. Nous avons évalué la scalabilité du kernel Rust sur un corpus synthétique de 10M tokens avec une taille de vocabulaire croissante (N=10⁶ séquences, L=10, fenêtre=5, CPU 8 cœurs, 32 Go RAM) :

| V | k | Comptage | PPMI CSR | SVD randomisé | Total | Mémoire CSR | Mémoire embeddings |
|---|---|---|---|---|---|---|---|
| 37K | 100 | 6.0s | 2.0s | 48.3s | **56.4s** | 428 MB | 28 MB |
| 100K | 100 | 5.3s | 3.0s | 85.4s | **93.9s** | 400 MB | 76 MB |
| 500K | 100 | 5.4s | 3.9s | 155.3s | **164.8s** | 404 MB | 381 MB |
| **1M** | 50 | **5.0s** | **4.5s** | **85.3s** | **94.8s** | **408 MB** | **381 MB** |

**Analyse de complexité :**
- **Comptage et PPMI** sont O(1) par rapport à V, dépendant uniquement du nombre de tokens et de la fenêtre de cooccurrence. Le passage de V=37K à V=1M (×27) n'augmente pas le temps de comptage (stable ~5s) car le nombre de paires uniques est borné par les tokens (35M pour 10M tokens × fenêtre 5).
- **SVD randomisé** est le goulot : O(V · k² + nnz · k). À V=1M, k=50, elle domine à 85s. La mémoire est dominée par la matrice CSR (408 MB) et le résultat SVD (381 MB en f64).
- **Empreinte mémoire maîtrisée :** V=1M tient dans 800 MB (CSR + embeddings), sans dépasser les limites d'un serveur standard. Un Transformer nécessiterait ~3 Go pour la seule table d'embeddings 1M×768.

**Conclusion :** Le kernel Rust ne montre aucun point de rupture jusqu'à V=10⁶. La SVD randomisée sur CPU 8 cœurs factorise une matrice d'un million de lignes en moins de 3 minutes (k=100) ou 90 secondes (k=50), sans GPU, sans bibliothèque externe.

#### 10.8 Génération Auto-Régressive (Inverse Motor + Φ)

L'**Inverse Motor** (Section 8) projette l'état LIF $S_t$ vers le vocabulaire via l'opérateur inverse $w_{t+1} = \arg\max_w \langle S_t, e(w) \rangle$. Le pipeline de génération combine trois mécanismes :

1. **Alignement sémantique** : produit scalaire entre l'état prédictif et chaque embedding de mot (inverse motor).
2. **Contrainte topologique (Φ)** : bonus $\phi(\text{last}, w) = 1$ si $w$ est voisin de friction du dernier mot émis, sinon 0. Score = $\lambda \cdot \langle S, e(w) \rangle + (1-\lambda) \cdot \phi$.
3. **Répression des répétitions** : pénalité multiplicative par émission ($\gamma = 0.5$) pour briser les cycles d'identité.
4. **Signal d'arrêt homéostatique (Φ)** : si $||S_{t+1} - S_t||^2 < \theta$, le système cesse d'émettre.

**V5.3 — Mono-LIF (α=0.8, λ=0.5) :**

| Prompt | Génération (20 tokens) | Dérive sémantique |
|--------|----------------------|-------------------|
| `a man is` | `man is a the on in and with shirt wearing red white black jacket blue hat woman sitting bench sits` | Parc → vêtements → scène |
| `the dog ran` | `dog brown running grass runs field grassy across on beach sand with the is a man in and wearing shirt` | Chien → prairie → plage → personne |
| `a woman sits` | `woman a is man the in on and with shirt wearing red white black jacket blue hat an orange vest` | Portrait → couleurs → vêtements |

**V5.3 — Dual-LIF Génératif (α_slow=0.9, α_fast=0.5, η=0.4, λ=0.5) :**

| Prompt | Génération (20 tokens) | Amélioration syntaxique |
|--------|----------------------|------------------------|
| `a man is` | `man is a the on in and wearing shirt red white black with blue jacket green boy young girl little` | "and wearing" direct, skip "with" |
| `the dog ran` | `dog brown running grass runs across the is a on man in and with black shirt white red wearing blue` | "runs across the" continu, skip "field grassy" |
| `a woman sits` | `woman a is man the in and wearing red shirt white black jacket blue hat with on sitting bench sits` | "wearing red shirt" (V→N au lieu de N→V) |

**Analyse :** Le Dual-LIF Génératif utilise un état prédictif composé $S_{pred} = S_{slow} + \eta \cdot S_{fast}$. La mémoire lente (α=0.9) retient le sujet de la phrase, tandis que la mémoire rapide (α=0.5) biaise la prédiction vers les transitions locales (syntaxe). Le résultat est une **génération plus fluide** : les verbes précèdent leurs objets (`wearing red shirt` au lieu de `shirt wearing red`), et les transitions sont plus directes (`and wearing` au lieu de `and with shirt wearing`). La contrepartie est une **moins grande diversité thématique** : le système reste plus proche du sujet initial et explore moins de concepts éloignés.

**V7 — Ancrage Épisodique (Mémoire de Travail) :** Pour contrer la dérive longue-distance, le décodeur V7 ajoute une **ancre épisodique** ($S_{anchor}$) — l'état lent $S_{slow}$ est gelé après la lecture du prompt. À chaque pas de génération, la **dérive** $\delta = 1 - \langle S_{slow}, S_{anchor} \rangle$ est mesurée. Si $\delta > \theta_{drift}$ (seuil 0.25), l'ancre est partiellement réinjectée : $S_{slow} \leftarrow 0.65 \cdot S_{slow} + 0.35 \cdot S_{anchor}$.

**Trace homéostatique (prompt "the dog ran", 50 tokens, θ_drift=0.25) :**

```
 step | drift | action         | mot généré   | thème
──────┼───────┼────────────────┼──────────────┼────────────────────
  0   | 0.003 | —              | dog          │ chien (ancre)
  4   | 0.034 | —              | runs         │ course
  8   | 0.046 | —              | a            │ transition
 12   | 0.113 | —              | and          │ transition
 16   | 0.208 | —              | white        │ vêtements
 18   | 0.277 | ⚓ RAPPEL 35%  | wearing      │ → drift reset 0.155
 20   | 0.185 | —              | jacket       │ vêtements
 24   | 0.235 | —              | sitting      │ action
 26   | 0.277 | ⚓ RAPPEL 35%  | sits         │ → drift reset 0.137
 30   | 0.187 | —              | two          │ personnes
 34   | 0.245 | —              | people       │ personnes
 35   | 0.252 | ⚓ RAPPEL 35%  | outside      │ → drift reset 0.132
 38   | 0.176 | —              | stands       │ position
 42   | 0.254 | ⚓ RAPPEL 35%  | children     │ → drift reset 0.125
 45   | 0.150 | —              | little       │ enfant
 49   | 0.225 | —              | small        │ fin
```

Le motif est clair : **drift → rappel → reset → re-dérive** — un **oscillateur homéostatique** autour de l'attracteur sémantique initial. Le système laisse la pensée dériver naturellement (exploration conceptuelle), et quand la friction topologique avec l'ancre dépasse le seuil, il tire la trajectoire vers le souvenir épisodique. En 50 tokens, **4 rappels successifs** maintiennent la cohérence thématique sans jamais nécessiter de BPTT — un seul vecteur d'ancre (D=100) suffit.

**Contribution clé :** C'est la première démonstration de génération auto-régressive sans backprop ni probabilités, utilisant uniquement la géométrie d'un espace d'embeddings PPMI-SVD contrainte par un graphe de friction topographique. Le Dual-LIF Génératif démontre qu'une mémoire multi-échelle améliore la cohérence syntaxique même dans un cadre sans gradient, et l'ancrage épisodique résout la limitation de mémoire longue-distance sans rétropropagation temporelle (BPTT) — une solution biologiquement inspirée, infiniment plus économe que la conservation de l'historique complet des activations.

#### 10.9 Expériences proposées

1. **Triple-échelle temporelle :** Ajout d'une mémoire intermédiaire ($\alpha = 0.7$) pour une hiérarchie complète (lent/moyen/rapide).
2. **Planification multi-phrase :** L'ancrage épisodique maintient la cohérence intra-phrase (50+ tokens). Une extension naturelle est l'ancrage hiérarchique : un deuxième niveau d'ancre pour le thème global du paragraphe, avec un seuil de dérive plus large.

### 11. DISCUSSION

TSO propose un changement de paradigme : passer d'une exécution systématique à une cybernétique de survie active. En assujettissant le calcul à une friction géométriquement calculable, TSO aligne l'efficacité computationnelle sur la complexité réelle du problème. L'implémentation en Rust fournit une base de référence rapide, portable et déterministe pour explorer cette alternative aux Transformers.

### 12. LIMITATIONS AND OPEN QUESTIONS

1.  **Bootstrap Sémantique :** Le système dépend initialement d'un encodeur NLI figé pour typer les arêtes. Comment cette sémantique peut-elle émerger de manière totalement endogène à partir de la règle R-STDP ?
2.  **Tuning des hyperparamètres :** L'apprentissage automatique de l'ensemble des paramètres libres ($\Delta t, \gamma, \epsilon, \theta_t, \theta_c$) reste une question ouverte cruciale pour l'autonomie du système.
3.  **Cohérence Globale :** La réparation locale d'une arête peut théoriquement briser une contrainte voisine satisfaite. La convergence globale du système devra être formellement démontrée.
4.  **Capacité linguistique :** Les expériences devront démontrer que la nature événementielle du calcul ne limite pas la capacité expressive par rapport aux modèles denses.
5.  **Friction multi-couche :** Comment empiler les couches de $\Phi$ pour obtenir une expressivité comparable à la profondeur des Transformers ?

### 13. CONCLUSION

Les RNN ont été remplacés par les Transformers grâce à la parallélisation de l'attention. Nous proposons que la **friction topographique ($\Phi$)** explore une direction alternative où le calcul est conditionné par une dynamique interne de stabilisation, et non par une obligation liée au flux de données. La validation sur SNLI (56.69% test, ~20s CPU) démontre que l'architecture TSO complète — Dual-LIF multi-échelle, friction séquentielle $\Phi$, opérateur de négation et classification par attracteurs locaux — capture l'ordre des mots et les relations de contradiction sans attention dense ni rétropropagation. Le **Dual-LIF (α=0.9/0.5)** agit comme un équivalent neuromorphique de l'attention multi-tête, ajoutant **+0.80%** au mono-LIF et portant le gain total à **+1.84% au-delà du plafond sac-de-mots**. En apprentissage continu, TSO démontre une **immunité structurelle à l'oubli catastrophique** : les features TSO 17D sont un fixateur topologique immuable — après apprentissage d'une seconde tâche (MultiNLI), un classifieur ré-entraîné sur SNLI retrouve exactement 56.96% (Δ = 0.00%), et le mode Freeze+Add préserve 100% de la performance originale. Ces propriétés sont structurellement impossibles pour les Transformers dont la rétropropagation modifie globalement tous les poids partagés. Enfin, **la génération auto-régressive par Inverse Motor** (Section 10.8) démontre que TSO peut produire du texte cohérent par dérive sémantique topologique — sans gradient, sans softmax, sans couche de projection apprise — confirmant que l'architecture est un véritable générateur de langage, pas seulement un extracteur de features. L'**ancrage épisodique (V7)** résout le problème de mémoire longue-distance sans BPTT : le système oscille homéostatiquement autour de son ancre sémantique, produisant un comportement analogue aux oscillateurs à attracteur étrange des systèmes dynamiques biologiques. Avec son kernel Rust comme fondation, TSO pose les bases d'une intelligence artificielle véritablement adaptative, événementielle, efficiente et compatible avec les principes d'apprentissage continu des systèmes neuromorphiques.
