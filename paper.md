# TSO : Topographic Stabilization Operator
## Architecture neuromorphique à friction topographique

**Auteur :** Hamouda ALIAS
**Date :** Juillet 2026
**Discipline :** Architectures d'IA, Systèmes Neuromorphiques, Systèmes Dynamiques

---

## Résumé

Les architectures d'IA actuelles exécutent une quantité fixe d'opérations
par token, quelle que soit la complexité cognitive de l'entrée.

Cet article propose **TSO (Topographic Stabilization Operator)**, une
architecture où le calcul est déclenché par une mesure interne d'instabilité,
non par l'arrivée d'une donnée.

Fondée sur la Théorie de la Dissipation Cognitive (CDT), TSO modélise
l'activité neuronale comme minimisation d'une énergie de friction Φ,
définie comme contrainte géométrique sur un graphe conceptuel émergent.

**Contributions clés :**
1. **Friction topographique (Φ)** : calcul événementiel, proportionnel à la
   complexité de l'entrée, pas à sa longueur.
2. **Catégorisation par prototypes** : attracteur, VAE online, FPI pour
   la généralisation sur entrées visuelles de haute dimension.
3. **VAE online + Gumbel-STE** : encodeur variationnel entraîné à chaque
   tick avec gradient local via straight-through estimator.
4. **Benchmark MiniGrid** (147D RGB) : TSO bat le linéaire de +105%,
   démontrant l'avantage de la catégorisation par prototypes.

Le kernel de référence est implémenté en Rust (ndarray), sans Python,
PyTorch ni CUDA. 45 tests unitaires valident le cycle cognitif.

---

## 1. Introduction

Les grands modèles de langage (LLMs) ont progressé grâce au passage à
l'échelle. Mais les modèles denses présentent une limite : la relation
entre information et calcul est statique.

Un modèle dense déploie les mêmes ressources par token, qu'il traite un
mot trivial ou une équation complexe. Cela entraîne un coût énergétique
élevé et rend l'apprentissage continu vulnérable à l'oubli catastrophique.

Nous introduisons **TSO**, une architecture événementielle où le calcul
est déclenché par le maintien de l'homéostasie interne.

La thèse fondatrice : le calcul devient une conséquence d'une mesure
interne d'instabilité, non une obligation liée au flux de données.
L'activité neuronale émerge comme réponse à une friction cognitive ;
le système ne calcule que lorsqu'une contradiction perturbe son équilibre.

TSO est architecturé autour d'un **kernel unique** (Rust, ndarray) qui
supporte la navigation visuelle (MiniGrid, §6) et s'étend au traitement
du langage naturel (SNLI, travaux futurs §8).

---

## 2. Travaux connexes

TSO s'inscrit à la confluence de plusieurs domaines :

**Calcul adaptatif.** Des méthodes comme Adaptive Computation Time (ACT)
ou PonderNet ajustent le calcul à la difficulté de l'entrée. TSO s'en
distingue par son fondement géométrique : la friction Φ émerge du graphe
conceptuel, non d'une heuristique de halte.

**Réseaux à impulsions (SNN).** TSO utilise des réservoirs Dual-LIF pour
l'intégration temporelle multi-échelle. Contrairement aux SNN purs,
l'apprentissage repose sur un gradient local (Gumbel-STE) combiné à la
plasticité R-STDP, sans rétropropagation globale.

**Inférence active.** Issu de Friston, ce cadre modélise la cognition
comme minimisation de la surprise. TSO s'en distingue en modélisant la
friction comme contradiction structurelle interne, non comme erreur de
prédiction externe.

---

## 3. Théorie de la Dissipation Cognitive (CDT)

Le fondement de TSO repose sur la modélisation des contradictions comme
une grandeur énergétique calculable.

**Définition 1 (État Cognitif).** L'état du système à l'instant t est
X_t = (G_t, S_t, W_t). G_t est le graphe sémantique (nœuds = concepts,
arêtes = contraintes), S_t l'activité neuronale, W_t les poids synaptiques.

**Définition 2 (Friction Φ).** La friction globale mesure la violation
des contraintes géométriques du graphe G_t. Pour deux vecteurs d'activation
z_i, z_j ∈ R^d :

- **Contrainte d'Implication** (w_ij=1) : les vecteurs doivent être alignés.
  Violation = max(0, γ − ⟨z_i, z_j⟩)
- **Contrainte d'Exclusion** (w_ij=−1) : les vecteurs doivent être opposés.
  Violation = max(0, ⟨z_i, z_j⟩ − ε)

La friction globale : Φ(G_t) = Σ_{(i,j)∈E} Violation_ij.

**Dynamique.** L'évolution temporelle suit trois lois :
- G_{t+1} = G(G_t, Φ_t) — mise à jour topologique (prévue)
- S_{t+1} = f(S_t, I_t, W_t) — dynamique neuronale LIF (implémentée)
- W_{t+1} = W_t + ΔW_t — plasticité locale (R-STDP, prévue)

### 3.1 Dual-LIF : mémoire multi-échelle

Le Dual-LIF utilise deux réservoirs LIF parallèles :

- **Mémoire lente** (α=0.9) : contexte global (sujet, agent, thème)
- **Mémoire rapide** (α=0.5) : syntaxe locale (2-3 derniers mots, négations)

Chaque mot met à jour les deux mémoires simultanément. Les features
résultantes (6D : cos, distance euclidienne, ratio des normes) remplacent
les 3D du mono-LIF.

### 3.2 PerceptualBelt : pipeline de représentation

Le PerceptualBelt fusionne quatre modules en une représentation unifiée :

- **Attention spatiale** : gain modulateur par erreur de prédiction
- **Dual-LIF** : intégration temporelle multi-échelle
- **AttractorField** : catégorisation par prototypes
- **VAE + FPI** : encoding variationnel et inférence active

Le belt transforme l'observation brute en vecteur latent stabilisé,
partagé entre le cortex (classification) et le cervelet (action).
Module distinct (perceptual_belt.rs) pas encore intégré au heartbeat —
voir §7.

### 3.3 Opérateurs cognitifs

Pour dissiper Φ, le système dispose d'opérateurs géométriques :

- **Inversion** : z_i → −z_i (contradiction directe)
- **Alignement** : moyenne de deux vecteurs, normalisation sur la sphère
- **Répulsion** : gradient tangent projectif (séparation de nœuds)
- **Expansion dimensionnelle** : doublement de l'espace latent pour les
  contradictions fortes (opérateur "MAIS", orthogonalité stricte)

### 3.4 Gumbel-STE : gradient local vers le VAE

L'argmax sur les centroids du VAE casse le gradient. Nous utilisons un
**Straight-Through Estimator (STE)** basé sur Gumbel-Softmax.

La température τ s'annele de 1.0 à 0.1 (decay 0.995). Les poids softmax
pondèrent la mise à jour des centroids, permettant au gradient de circuler
de la tâche vers l'encodeur.

---

## 4. Architecture TSO

### 4.1 Cycle cognitif

Le heartbeat s'exécute en 4 étapes à chaque tick :

1. **Catégorisation** : attracteur / VAE / FPI → concept_id
2. **Action** : sélection par Cerebellum (TD) ou EFE (active inference)
3. **Apprentissage** : reinforce_td (TD(λ))
4. **Calcul de Φ** : friction mesurée mais ne contrôle pas encore le cycle

À ce stade, le step() complet s'exécute à chaque tick quel que soit Φ.
L'intégration du PerceptualBelt et le gating par Φ sont la priorité
immédiate (voir §7).

---

## 5. Implémentation Rust

Le kernel TSO est implémenté en Rust pur (crate tso-engine), sans
Python, PyTorch ni CUDA.

**Modules du kernel :**

| Module | Rôle |
|--------|------|
| neurons.rs | Clusters LIF + Dual-LIF |
| core.rs | Graphe Φ, résolution de contraintes |
| plasticity.rs | R-STDP (prévu) |
| operators.rs | Inversion, Alignement, Répulsion |
| perceptual_belt.rs | Pipeline de représentation unifié |
| attractor.rs | AttractorField (prototypes) |
| cerebellum.rs | Actor-critic TD(λ) |
| vae.rs | Encodeur variationnel (online) |
| fpi.rs + efe.rs | Active inference (FPI/EFE) |
| rotating_t.rs | Benchmark non-stationnaire |
| minigrid_env.rs | MiniGrid Rust 7×7 147D |

**Propriétés :**
- Zéro dépendance Python — compilation native
- Parcimonie explicite — clusters inactifs ignorés dans Φ
- Pas de rétropropagation globale — apprentissage local (R-STDP prévu)
- Parallélisation via Rayon
- 45 tests unitaires validant le cycle cognitif

---

## 6. Expériences

### 6.1 Benchmark MiniGrid — Navigation visuelle

Benchmark sur MiniGrid DoorKey (grille 7×7, observation RGB 147D).

**Protocole :** 10 seeds, 100 épisodes, but tournant tous les 50 épisodes.
Métrique : reward moyen par épisode.

| Condition | Moyenne | σ | Δ vs linéaire |
|-----------|---------|---|---------------|
| Actor-critic linéaire (147D) | 1.03 | 0.17 | — |
| TSO + attracteur (147D) | 2.11 | 0.36 | +1.08 |
| TSO + VAE (147D→16D) | 2.06 | 0.34 | +1.03 |
| TSO + VAE + FPI | **2.13** | 0.34 | +1.10 |

TSO surpasse le linéaire de +105% en configuration FPI+attracteur.
Le VAE avec Gumbel-STE améliore la robustesse mais n'est pas nécessaire
sur cet environnement.

### 6.2 Analyse de dimension

Plus l'observation est riche, plus l'écart TSO vs linéaire se creuse :

| Dimension | Benchmark | Gain |
|-----------|-----------|------|
| 5D (whiskers) | Terrarium 7×7 | ≈ (48.5% vs 49.5%) |
| 25D (grille) | GridWorld 5×5 | +24% |
| 147D (RGB) | MiniGrid 7×7 | +105% |

La catégorisation par prototypes devient rentable quand le nombre de
dimensions dépasse la capacité d'un mélange linéaire.

---

## 7. Discussion et limites

### Apport conceptuel

TSO propose de passer d'une exécution systématique à une cybernétique de
survie active. Assujettir le calcul à une friction géométrique aligne
l'efficacité sur la complexité réelle du problème.

### Φ passif

La friction est calculée à chaque tick mais ne contrôle pas le
déclenchement du calcul. Le step() complet s'exécute systématiquement.
Le lien Φ > seuil → stabilisation active reste qualitatif.
Un modèle formel liant ||E||, sparsité α et itérations de résolution
reste à établir.

### Différentiabilité partielle

Le Gumbel-STE permet au gradient de circuler du choix de catégorie vers
le VAE. Mais la jointure complète TD → belt → VAE n'est pas active dans
step() : le gradient TD ne remonte pas jusqu'au VAE.
Une connexion backprop_td existe dans l'API mais n'est pas appelée.

### R-STDP non intégrée

L'apprentissage repose sur l'actor-critic TD(λ), classique, pas
neuromorphique. L'intégration de R-STDP (remplacement de reinforce_td
par mise à jour locale) est une piste ouverte.

### Échelle limitée

MiniGrid 7×7 (147D RGB) est un test de principe. Procgen (64×64, 16 jeux)
et Habitat (3D, ego-vision) sont les prochaines étapes identifiées.
Le VAE 16D latent reste à valider sur bruit visuel massif.

### Complexité O(|E|)

Le graphe Φ a une complexité O(|E|) par tick. Le déminage maintient |E|
borné sur les grilles 5×7. Pour des environnements 3D (>10³ concepts),
une parallélisation GPU (wgpu, burn) sera nécessaire.

---

## 8. Travaux futurs

Les axes prioritaires sont :

- **Gating par Φ** : rendre le calcul événementiel (contribution n°1)
- **Intégration R-STDP** : remplacer TD(λ) par plasticité locale
- **Jointure TD→VAE** : backprop_td dans le cycle step()
- **PerceptualBelt intégré** : brancher le belt dans le heartbeat
- **Benchmarks étendus** : Procgen, Habitat, plus de seeds, tests de
  significativité
- **Extension NLP** : évaluation sur SNLI prévue
- **Citations** : références bibliographiques à ajouter

---

## 9. Conclusion

Nous proposons que la friction topographique (Φ) explore une direction
où le calcul est conditionné par une dynamique interne de stabilisation,
non par une obligation liée au flux de données.

La validation sur MiniGrid (TSO 2.13 vs linéaire 1.03, +105%) confirme
que la catégorisation par prototypes apporte un avantage mesurable sur
les entrées visuelles de haute dimension.

Le kernel Rust, les 45 tests unitaires et les benchmarks reproductibles
sont disponibles en open source.


---

## Références

[Graves16] A. Graves. "Adaptive Computation Time for Recurrent Neural
Networks." arXiv:1603.08983, 2016.

[Banino21] A. Banino et al. "PonderNet: Learning to Ponder."
arXiv:2107.05407, 2021.

[Jang16] E. Jang, S. Gu, B. Poole. "Categorical Reparameterization with
Gumbel-Softmax." arXiv:1611.01144 (ICLR 2017), 2016.

[Friston09] K. Friston, J. Daunizeau, S. Kiebel. "Reinforcement Learning
or Active Inference?" PLOS ONE, 4(7):e6421, 2009.

[Friston10] K. Friston. "The free-energy principle: a unified brain
theory?" Nature Reviews Neuroscience, 11:127–138, 2010.

[Florian07] R. V. Florian. "Reinforcement Learning Through Modulation
of Spike-Timing-Dependent Synaptic Plasticity." Neural Computation,
19(6):1468–1502, 2007.

[Pfister06] J.-P. Pfister, T. Toyoizumi, D. Barber, W. Gerstner. "A
Learning Theory for Reward-Modulated Spike-Timing-Dependent Plasticity
with Application to Biofeedback." PLOS Computational Biology, 2(7):e85,
2006.

[Kingma13] D. P. Kingma, M. Welling. "Auto-Encoding Variational Bayes."
arXiv:1312.6114, 2013.

[Sutton88] R. S. Sutton. "Learning to Predict by the Methods of Temporal
Differences." Machine Learning, 3:9–44, 1988.

[Chevalier18] M. Chevalier-Boisvert, L. Willems, S. Pal. "Minimalistic
Gridworld Environment for OpenAI Gym." GitHub, 2018.

[Chevalier23] M. Chevalier-Boisvert et al. "Minigrid & Miniworld:
Modular & Customizable Reinforcement Learning Environments for
Goal-Oriented Tasks." NeurIPS, 2023.

[Gerstner14] W. Gerstner, W. M. Kistler, R. Naud, L. Paninski.
"Neuronal Dynamics: From Single Neurons to Networks and Models of
Cognition." Cambridge University Press, 2014.

[Schaar15] J. van Schaarsbergen, et al. "ndarray: an N-dimensional array
with numpy-like API in Rust." crates.io, 2015–2024.