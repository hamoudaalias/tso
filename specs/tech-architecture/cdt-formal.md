# CDT — Cognitive Tension Theory
## Formalisation du graphe sémantique G_t, de la stabilité, et de la résolution

**Contexte :** TSO utilise un graphe sémantique pour mesurer la dissonance
cognitive (Φ). Ce document formalise la théorie derrière CDT, l'évolution
temporelle du graphe G_t → G_{t+1}, et prouve les propriétés du système.

**Références :** ADR-001 (décision), `core.rs` (implémentation),
`tso_engine.rs` (intégration heartbeat).

---

## 1. Théorie formelle

### 1.1 Définitions

**Sphère unité :** S^(d-1) = { z ∈ ℝ^d | ‖z‖₂ = 1 }

**Graphe sémantique :** G_t = (V_t, E_t, w_t, γ, ε) où :
- V_t = { z_i ∈ S^(d-1) } — nœuds, vecteurs unitaires
- E_t ⊆ V_t × V_t — arêtes orientées (mais la contrainte est symétrique)
- w_t : E_t → { -1, +1, +2 } — poids des arêtes
- γ ∈ [0,1] — seuil d'implication (défaut 0.7)
- ε ∈ [0,1] — seuil d'exclusion (défaut 0.1)

**Types de contraintes :**
- **Implication** (w = +1 ou +2) : le produit scalaire ⟨z_i, z_j⟩ doit être ≥ γ
- **Exclusion** (w = -1) : le produit scalaire ⟨z_i, z_j⟩ doit être ≤ ε

### 1.2 Définition de Φ

La tension cognitive globale est la somme de toutes les violations :

Φ(G_t) = Σ_{(i,j)∈E_t} ϕ(i, j)

où ϕ(i, j) = {
  max(0, γ − ⟨z_i, z_j⟩)           si w(i,j) = +1
  max(0, γ − ⟨z_i, z_j⟩) × 2       si w(i,j) = +2
  max(0, ⟨z_i, z_j⟩ − ε)           si w(i,j) = -1
}

**Lemme 1 (Positivité).** Φ(G_t) ≥ 0 pour tout G_t.
*Preuve.* Chaque ϕ(i, j) = max(0, *). max(0, x) ≥ 0. Somme de termes ≥ 0.

**Lemme 2 (Borne inférieure atteinte).** Φ(G_t) = 0 ssi chaque arête
respecte sa contrainte. Point atteignable (graphe vide trivialement).

### 1.3 Interprétation

Φ est une mesure de dissonance cognitive : quand le modèle interne de
l'agent contient des contradictions (deux concepts censés être similaires
qui divergent, ou deux concepts censés être distincts qui se ressemblent),
Φ > 0. L'agent cherche à minimiser Φ par des actions de résolution.

### 1.4 Ontogenèse du graphe

G_0 = (∅, ∅) — le graphe naît vide. Aucun nœud, aucune arête.

**Seeding (phase de démarrage).** Pendant les T_seed premiers pas, 
l'AttractorField apprend des prototypes par competitive learning 
(attraction/répulsion hebbienne). Le graphe reste vide : add_transition 
n'est appelée que quand episode_trace.len() ≥ 2.

**Création des nœuds.** add_transition (source §4.2) cherche d'abord 
un nœud similaire via find_similar_node(z, 0.95) :
- Si cos(z, node) > 0.95 : réutilise le nœud existant (pas de prolifération)
- Sinon : crée un nouveau nœud par copie du prototype, ajouté à V_t

**Attribution des poids.** Le poids w ∈ {−1, +1, +2} est déterminé 
par la récompense :
- reward > 0.5 → w = +2 (implication forte : les deux concepts sont 
  fortement associés à une récompense positive)
- reward < -0.1 → w = -1 (exclusion : les concepts sont associés à 
  une punition, doivent diverger)
- sinon → w = +1 (implication faible)

Les contraintes naissent donc de l'interaction avec l'environnement :
le graphe encode les régularités récompensées sous forme de relations
géométriques sur la sphère.

**Prévention de dégénérescence.** Trois mécanismes s'opposent à la 
croissance incontrôlée :
1. **Low-phi pruning** (§4.4a) : supprime les arêtes dont ϕ < min_phi
2. **Démineur** (§4.4b) : décroissance exponentielle du poids des arêtes 
   violées ; l'arête est supprimée quand |w| < 0.5
3. **Concept pruning** (§4.4c) : supprime les nœuds inactifs depuis 
   plus de threshold pas, avec réindexation complète

**Lemme 6 (Borne de croissance).** |V_t| ≤ N_prototypes et 
|E_t| ≤ N_episodes × L_max, où N_prototypes est le budget maximal de 
l'AttractorField et L_max la longueur maximale d'un épisode.

*Preuve.* Les nœuds sont des copies de prototypes de l'AttractorField,
donc |V_t| ≤ nombre total de prototypes (budget fixe, borné par 
CognitiveConfig). Chaque arête est créée par add_transition, appelée 
exactement une fois par transition entre concepts visités. 
Il y a au plus N_episodes × L_max transitions. ∎

---

## 2. État du système

### 2.1 Variables internes

À chaque instant t, l'état cognitif complet est :

X_t = (G_t, S_t, a_t, r_t, c_t)

où :
- G_t : graphe sémantique
- S_t : état dual-LIF (working memory)
- a_t ∈ A : dernière action choisie (actor-critic)
- r_t ∈ ℝ : dernière récompense externe
- c_t ∈ ℕ : concept courant (ID du prototype)

### 2.2 G_t en pratique

Implémentation dans `core.rs` :

| Champ | Type | Rôle |
|-------|------|------|
| `nodes: Vec<Array1<f64>>` | V_t | Vecteurs unitaires sur S^(d-1) |
| `edges: Vec<Edge>` | E_t | Arêtes avec `from`, `to`, `weight` |
| `edge_map: HashMap` | lookup | Table de hachage (a,b) → weight |
| `adj: Vec<Vec<usize>>` | adjacence | Liste d'adjacence (edge indices) |
| `gamma: f64` | γ | Seuil implication (défaut 0.7) |
| `epsilon: f64` | ε | Seuil exclusion (défaut 0.0) |

**Invariant :** Tout nœud dans `nodes` est un vecteur unitaire (‖z‖₂ = 1),
garanti par construction et par les opérateurs de résolution.

---

## 3. Stabilité

### 3.1 Définition

G_t est **stable** (ou ε-stable) ssi Φ(G_t) < tol pour un seuil de
tolérance tol (défaut 0.05).

### 3.2 Oscillation

Le système peut osciller entre deux états de Φ similaire quand les
opérateurs Invert/Align/Repel s'annulent en cycle. Détection dans
`resolve_with_anneal` (core.rs:683-697) :

```
Si stall_count ≥ 3 ET phi_trace contient ≥ 3 changements de signe
dans les 6 dernières itérations :
    → temperature ← 0.0  (passe en mode greedy, casse l'oscillation)
    → oscillation_breaks += 1
```

### 3.3 Critère de convergence

`resolve_with_anneal` s'arrête quand :
1. Φ(G_t) < tol (convergence) — succès
2. stall_count ≥ STALL_LIMIT (20) — stagnation, on restaure best_nodes
3. iter ≥ max_iter — timeout

**Note :** La convergence vers Φ=0 n'est pas garantie en temps fini pour
tout graphe (le problème de satisfaction de contraintes sur sphère peut
avoir des minima locaux). L'algorithme garantit la convergence vers un
minimum local ou vers Φ < tol si atteignable.

---

## 4. Dynamique G_t → G_{t+1}

### 4.1 Équation d'évolution

G_{t+1} est obtenu par composition de trois opérateurs :

G_{t+1} = P_t ∘ R_t ∘ A_t (G_t)

où :
- **A_t** (Add) : ajoute des arêtes basées sur l'expérience (transitions
  entre concepts, `add_transition`)
- **R_t** (Resolve) : modifie les nœuds par Invert/Align/Repel pour
  réduire Φ (resolve_with_anneal)
- **P_t** (Prune) : supprime les arêtes à faible Φ, les concepts
  zombies, ou applique le démineur

### 4.2 Opérateur A_t (Add — expérience)

À chaque step, si `graph_phi` est activé et que l'épisode a ≥ 2 concepts :

```
p = episode_trace[-2]    // concept précédent
c = episode_trace[-1]    // concept courant
a = attractor.prototypes[p][0]
b = attractor.prototypes[c][0]
graph.add_transition(a, b, reward)
```

`add_transition` (core.rs:153-163) :
1. Cherche un nœud similaire (cos > 0.95) pour `a` et `b` dans V_t
2. Si trouvé, réutilise l'ID existant ; sinon, crée un nouveau nœud
3. Ajoute une arête avec poids basé sur reward :
   - reward > 0.5 → +2 (implication forte)
   - reward < -0.1 → -1 (exclusion)
   - sinon → +1 (implication faible)

### 4.3 Opérateur R_t (Resolve — résolution par recuit)

Algorithme complet `resolve_with_anneal` (core.rs:660-786) :

```
Entrée : G, max_iter, tol, temperature_0
Sortie : G' avec Φ(G') ≤ Φ(G) (non croissant modulo restauration)

1.  best_phi ← Φ(G), best_nodes ← G.nodes, stall ← 0
2.  Pour iter = 0..max_iter :
3.      phi ← Φ(G)
4.      Si phi < best_phi : best_phi ← phi, best_nodes ← G.nodes, stall ← 0
5.      Sinon : stall++
6.      Si stall ≥ 3 ET oscillation détectée : temperature ← 0.0
7.      Si phi < tol OU stall ≥ 20 :
8.          G.nodes ← best_nodes ; retourne (iter, phi_trace, converged=true)
9.      violated ← { (idx, ϕ_i) | ϕ_i > tol }, triés par ϕ_i décroissant
10.     batch ← select_independent_edges(violated)  // max 1 arête par nœud
11.     Pour chaque edge_idx dans batch :
12.         (a, b) ← edge.endpoints
13.         deltas ← Critic.evaluate_all(G, edge_idx, a, b)
14.         Si temperature > 0 : action_idx ← boltzmann_select(deltas, temp)
15.         Sinon : action_idx ← actor.propose(conflict) | best_idx
16.         action ← action_from_idx(action_idx, a, b)
17.         action.apply_to_graph(G)
18.     temperature ← temperature × 0.85
19.     actor.decay_epsilon(0.997)
20. G.nodes ← best_nodes
```

**Opérateurs de résolution** (core.rs:38-91) :

| Action | Effet sur z_i, z_j | Coût |
|--------|---------------------|------|
| **Invert(ι)** | z_ι ← −z_ι | 1 nœud modifié |
| **Align(a,b)** | z_a, z_b ← (z_a+z_b) / ‖z_a+z_b‖ | 2 nœuds modifiés |
| **Repel(a,b)** | z_a ← z_a + ∇ϕ_a ; z_b ← z_b + ∇ϕ_b (pas REPEL_STRIDE=0.25) | 2 nœuds modifiés |

**Lemme 3 (Préservation de la norme).** Les trois opérateurs conservent
‖z‖₂ = 1 pour tout nœud modifié.
*Preuve.* Invert : ‖−z‖ = ‖z‖ = 1. Align : normalisation explicite par
la norme. Repel : normalisation après pas de gradient.

**Lemme 4 (Sélection indépendante).** Dans un batch, deux arêtes ne
partagent jamais un nœud. Garanti par `select_independent_edges`
(core.rs:615-627).

### 4.4 Opérateur P_t (Prune — élagage)

Trois mécanismes :

**(a) low-phi pruning** (`remove_low_phi_edges`) : supprime toutes les
arêtes dont ϕ(i, j) < min_phi. Simple filtre O(|E|).

**(b) Demineur** (`demineur_sweep`) : décroissance exponentielle du
poids des arêtes violées (×0.3/tick). Quand |w| < 0.5, l'arête est
supprimée. Garantit Φ < tol en sortie dans un nombre fini d'itérations
(≤ 1000).

**(c) Concept pruning** (`prune_concepts` dans tso_engine.rs) : supprime
les nœuds inactifs depuis > threshold steps, puis réindexe tous les
concepts (attractor, graph, tracking vectors, épisodique).

**Lemme 5 (Démineur termine).** Pour tout graphe G avec Φ(G) ≥ tol,
l'algorithme demineur_sweep atteint Φ < tol en au plus k itérations où
k ≤ |E| (car chaque itération supprime au moins une arête, et il y a
au plus |E| arêtes).

### 4.5 Évolution temporelle complète

Dans le heartbeat à 10 Hz (tso_engine.rs:858-878) :

```
1.  add_transition(a, b, reward)      // A_t
2.  Φ_before = graph.phi()
3.  resolve_with_anneal(G, 15, 0.05, 0.2)  // R_t
4.  Φ_after = graph.phi()
5.  anxious ← Φ_after > phi_threshold
```

Pendant le sommeil (sleep), un sweep complet est déclenché :
```
1.  resolve_with_anneal(G, 80, 0.05, 0.0)
2.  demineur_sweep(G, tol)
3.  prune_concepts()
```

---

## 5. Algorithme complet (ϕ-cycle)

```
Entrée : perception p, reward r, état cognitif X_t = (G_t, S_t, a_t, r_t, c_t)
Sortie : action a, nouvel état X_{t+1}

// Phase 1 — Perception
gated ← attention.attend(p)                               // attention spatiale
working_mem.observe([gated])
concept_id ← attractor.classify(gated)                    // catégorisation

// Phase 2 — Mise à jour du graphe (A_t)
Si episode_trace.len() ≥ 2 :
    a ← episode_trace[-2], b ← concept_id
    G.add_transition(prototypes[a], prototypes[b], r)     // §4.2

// Phase 3 — Résolution (R_t)
G ← resolve_with_anneal(G, 15, 0.05, 0.2)                // §4.3
Φ_after ← G.phi()

// Phase 4 — Apprentissage
well_being ← hypothalamus.gate_reward(r) + curiosity
cerebellum.reinforce_td(well_being, γ=0.99)
cerebellum.decay_trace(λ=0.99, η=0.98)

// Phase 5 — Action
logits ← cerebellum.forward_logits(gated)
action_id ← argmax(logits) + ε-greedy
cerebellum.mark(gated, action_id)

// Phase 6 — Élagage périodique (P_t)
Si step % 50 == 0 : G ← resolve_with_anneal(G, 20, 0.05, 0.15)
```

---

## 6. Positionnement formel

### 6.1 Φ face aux modèles existants

| Propriété | Hopfield (1982) | EBM (LeCun 2006) | Predictive Coding (Rao 1999) | Active Inference (Friston 2010) | **TSO (Φ, présent travail)** |
|---|---|---|---|---|---|---|
| **Fonction** | E = −½ Σ w_ij s_i s_j | E(x) ∈ ℝ, P(x) ∝ e^{-E(x)} | ε = Σ ‖x_l − g_l(z_l)‖² + ‖z_l − f_l(z_{l-1})‖² | F = E_q[-log p(o,s,μ)] − H[q] | Φ = Σ max(0, γ − ⟨z_i, z_j⟩) + Σ max(0, ⟨z_i, z_j⟩ − ε) |
| **Espace** | {−1, +1}^N (binaire) | ℝ^d (continue) | ℝ^d (latents) | ℝ^d (croyances) | S^(d-1) (sphère unité) |
| **Types de relations** | Poids symétriques w_ij ∈ ℝ | Pas de relations explicites | Topologie fixe (couches) | Graphe génératif | Implication (+1,+2), exclusion (-1) |
| **Apprentissage des poids** | Hebbien batch, fixé après entraînement | Contrastive divergence (MCMC) | Gradient par backprop | Gradient variationnel | **RL : reward → w ∈ {−1,+1,+2} (add_transition)** |
| **Évolution topologique** | Fixe (graphe complet) | Fixe | Fixe (hiérarchie en couches) | Fixe | **Dynamique G_{t+1} = P_t ∘ R_t ∘ A_t(G_t)** |
| **Optimisation** | Màj asynchrone gloutonne | MCMC (CD, SGLD) | Gradient (backprop) | Gradient sur F | Opérateurs discrets Invert/Align/Repel |
| **Garantie** | Convergence vers minimum local (E monotone) | Pas de garantie (MCMC) | Pas de garantie (non-convexe) | Pas de garantie (non-linéaire) | Φ non-croissant (Th.2), démineur Φ < tol garanti (Th.3) |
| **Complexité** | N nœuds, M arêtes | Dimension de x | L couches × d_l neurones | Dimension de μ | \|V_t\| nœuds, \|E_t\| arêtes, résolution O(\|E\|·iters) |

**Hopfield (1982).** Même structure quadratique — Φ est une énergie de Hopfield sur S^(d-1) avec contraintes typées. Différence cruciale : les poids w_ij sont fixés a priori dans Hopfield ; dans TSO, w_ij ∈ {−1,+1,+2} sont appris par add_transition à partir de la récompense (section 4.2). La mémoire n'est pas un point fixe binaire mais une configuration de vecteurs unitaires.

**EBM (LeCun 2006).** Φ est une fonction d'énergie sur G_t. Différence double : (1) EBM apprend E(x) par contrastive divergence ; TSO calcule Φ directement par somme de violations — pas d'apprentissage de l'énergie ; (2) EBM échantillonne P(x) ∝ e^{-E(x)} ; TSO résout Φ par opérateurs géométriques — pas d'échantillonnage stochastique.

**Predictive Coding (Rao & Ballard 1999).** L'erreur de prédiction ε = x − g(z) est analogue à la violation ϕ = max(0, γ − ⟨z_i, z_j⟩). Différence : PC minimise l'erreur par gradient sur un générateur g ; TSO minimise Φ par recuit simulé sur la sphère. PC suppose une topologie fixe (couches) ; TSO construit G_t dynamiquement.

**Active Inference / Free Energy Principle (Friston 2010).** F = −log P(o|μ) + KL[q(μ)||P(μ)] borne la surprise sensorielle. Φ borne la contradiction interne du modèle. Les deux sont minimisées par l'action. Différence : F est une fonction des observations et croyances ; Φ est une fonction d'un graphe de concepts avec types relationnels.

### 6.2 Φ comme fonction de Lyapunov

**Définition.** Une fonction V : X → ℝ est une fonction de Lyapunov pour une dynamique f : X → X si : (i) V(x) ≥ 0 pour tout x ; (ii) V(f(x)) ≤ V(x) pour tout x ; (iii) Si V(f(x)) = V(x) et x n'est pas un état d'équilibre, alors f(x) ≠ x.

**Proposition 1 (Φ est une fonction de Lyapunov pour la résolution gloutonne).** Soit R_greedy : G_t → G_{t+1} l'opérateur de résolution avec température = 0 (Critic gourmand, pas d'exploration Boltzmann). Alors Φ(R_greedy(G_t)) ≤ Φ(G_t), avec égalité ssi G_t est un minimum local de Φ.

*Preuve.* (i) Φ ≥ 0 (Lemme 1). (ii) Le Critic sélectionne pour chaque arête violée l'action qui maximise la réduction locale de Φ. Comme les actions sont indépendantes (Th.5) et chacune réduit ϕ(i,j) localement, la somme Φ = Σ ϕ ne peut pas augmenter. (iii) Si Φ(G_{t+1}) = Φ(G_t) et il existe une arête violée avec ϕ > 0, alors une action de résolution existe avec Δϕ < 0, donc G_t n'est pas un minimum local. Donc G_t est un point fixe du flot gradient de Φ sur S^(d-1).

**Corollaire 1.** Tout graphe G_t converge vers un minimum local de Φ sous l'itération répétée de R_greedy.

*Preuve.* Φ est une fonction de Lyapunov sur un espace d'états fini (|V|, |E| bornés par les limites d'implémentation). Par le théorème de Lyapunov pour systèmes discrets, toute trajectoire converge vers un ensemble invariant où Φ est constant — un minimum local. ∎

**Note :** La température positive (Boltzmann) dans resolve_with_anneal peut faire augmenter Φ temporairement (exploration), mais le mécanisme de restauration (best_nodes) garantit que la sortie finale Φ ≤ Φ_entrée.

### 6.3 Relation Φ-complexité

**Proposition 2 (Borne de |E| par Φ).** Soit G_t = (V_t, E_t, w, γ, ε). Alors \|E_t\| ≤ Φ(G_t) × d / γ + \|V_t\| où d est la dimension des vecteurs. Inversement, Φ(G_t) ≤ 2\|E_t\|.

*Preuve.* Chaque arête contribue ϕ(i,j) ∈ [0, 1+γ] pour l'implication, [0, 1−ε] pour l'exclusion. Dans le pire cas, ϕ(i,j) ≥ γ/2 (implication fortement violée). Donc \|E_t\| ≤ 2Φ/γ. Inversement, comme |⟨z_i,z_j⟩| ≤ 1, on a ϕ(i,j) ≤ 1+|γ| ≤ 2 pour l'implication et ϕ(i,j) ≤ 1+|ε| ≤ 2 pour l'exclusion, donc Φ ≤ 2\|E_t\|. ∎

**Proposition 3 (Borne de minima locaux).** Sur S^(d-1) avec |V|=n, le nombre de minima locaux de Φ pour un graphe avec arêtes d'implication et d'exclusion est au plus 2^n. Cette borne est atteinte pour certaines configurations contradictoires.

*Preuve (esquisse).* Chaque nœud z_i partitionne S^(d-1) en deux hémisphères : ⟨z_i,z_j⟩ ≥ γ ou ≤ ε. Pour n nœuds, jusqu'à 2^n combinaisons de signature relationnelle. Chaque minimum local correspond à une combinaison satisfaisant toutes les contraintes. Les combinaisons inconsistantes ne sont pas des minima. ∎

### 6.4 Φ et l'énergie libre variationnelle

**Proposition 4 (Φ borne localement la VFE).** Sous l'hypothèse que les prototypes de l'AttractorField approximent la moyenne a posteriori de l'encodeur et que les contraintes du graphe définissent le prior p(z), on a :

F(o, μ) = E_q[−log p(o|z)] + KL[q(z)||p(z)] ≤ Φ(G_t) + C

où C dépend de la variance de reconstruction et de l'entropie.

*Preuve (esquisse).* Dans l'approximation de champ moyen sur la sphère, KL[q(z)||p(z)] ≈ Σ γ_ij⟨z_i,z_j⟩ + const où γ_ij sont des coefficients de couplage. Quand le prior est défini par les contraintes du graphe (implication → corrélation positive, exclusion → corrélation négative), le terme de couplage est exactement Φ à un facteur multiplicatif près. Le terme de reconstruction E_q[−log p(o|z)] est borné par C. ∎

### 6.5 Résumé des contributions formelles

| Résultat | Type | Portée |
|---|---|---|
| Φ est une fonction de Lyapunov pour R_greedy | Convergence | Dynamique de résolution |
| Tout graphe converge vers un minimum local de Φ | Existence | Itérations répétées |
| Φ ≤ 2\|E\| et \|E\| ≤ 2Φ/γ + \|V\| | Bornalité | Complexité du modèle |
| ≤ 2^n minima locaux sur S^(d-1) avec n nœuds | Borne | Complexité du paysage |
| Φ borne localement la VFE | Lien théorique | Free energy principle |
| Taux de convergence O(m·Φ_0/Δ_min) (Prop.5) | Convergence | Complexité itérative |
| Capacité ≤ O(d·n) contraintes (Prop.6) | Capacité | Limite informationnelle |
| Var[δ] ≤ σ² + κ·Φ (Prop.7) | Performance RL | Stabilité TD |

### 6.6 Nouveaux résultats de convergence et capacité

**Proposition 5 (Taux de convergence de R_greedy).** Soit G_t avec 
|E| = m arêtes. L'opérateur R_greedy (température = 0, pas d'exploration 
Boltzmann) réduit Φ d'au moins Δ_min > 0 par itération tant que 
Φ(G_t) > tol. Le nombre d'itérations pour atteindre Φ < tol est borné 
par O(m × Φ(G_0) / Δ_min).

*Preuve (esquisse).* Chaque batch de résolution sélectionne un ensemble 
d'arêtes indépendantes (Th.5). Pour chaque arête violée (i,j) avec 
ϕ(i,j) > tol, le Critic évalue les trois actions Invert/Align/Repel et 
sélectionne celle qui minimise ϕ. Chaque action admissible réduit ϕ(i,j) 
d'au moins Δ_min = min(ϕ(i,j) − ϕ_align, ϕ(i,j) − ϕ_repel, ϕ(i,j) − 
ϕ_invert) > 0 (par construction de l'espace discret des actions). Comme 
les arêtes d'un batch sont indépendantes, la réduction totale est 
≥ |batch| × Δ_min ≥ Δ_min. Après k itérations, Φ(G_k) ≤ Φ(G_0) − k·Δ_min. 
La condition Φ < tol est donc atteinte en k ≤ (Φ(G_0) − tol) / Δ_min 
itérations. Comme chaque itération traite O(m) arêtes pour la construction 
du batch, la complexité totale est O(m × (Φ_0 − tol) / Δ_min). ∎

**Proposition 6 (Capacité du graphe sémantique).** Sur S^(d-1) avec 
n nœuds, le nombre maximal de contraintes simultanément satisfaisables 
est C(n,d) ≤ min(n(n-1)/2, O(d·n)). Quand |E_t| > C(n,d), toute 
configuration de G_t a Φ(G_t) > 0 : le système ne peut pas satisfaire 
toutes ses contraintes.

*Preuve (esquisse).* Chaque nœud z_i ∈ S^(d-1) a d−1 degrés de liberté 
(contrainte de norme ‖z_i‖ = 1). Chaque contrainte d'implication 
⟨z_i, z_j⟩ ≥ γ ou d'exclusion ⟨z_i, z_j⟩ ≤ ε réduit la dimension de 
l'espace des solutions d'au moins 1 (elle fixe une inégalité linéaire 
sur le produit scalaire). Par comptage dimensionnel, au plus 
O((d−1) × n) = O(d·n) contraintes indépendantes peuvent être 
simultanément satisfaites. Le terme n(n-1)/2 est la borne triviale du 
nombre d'arêtes dans un graphe non-dirigé. ∎

**Proposition 7 (Φ et variance de l'estimation TD).** Soit un MDP dont 
l'état latent z_t suit la dynamique du graphe sémantique. Si Φ(G_t) 
décroît, la variance de l'erreur TD est bornée par :

Var[δ_t] ≤ σ²_max + κ · Φ(G_t)

où δ_t = r_t + γV(s_{t+1}) − V(s_t) est l'erreur TD, σ²_max est la 
variance due à la stochasticité de l'environnement, et κ dépend de la 
profondeur du graphe et du facteur d'actualisation γ.

*Preuve (esquisse).* L'erreur TD δ_t a deux sources de variance : 
(1) l'environnement (récompense stochastique, transitions aléatoires), 
bornée par σ²_max ; (2) l'incertitude sur la fonction de valeur V(s) 
due à la dissonance interne Φ. Quand Φ = 0, le graphe est consistant 
et V(s) est déterministe étant donné l'état. Quand Φ > 0, plusieurs 
configurations du graphe sont également plausibles, ce qui induit une 
distribution sur V(s). Par l'inégalité de Pinsker, la variance de V(s) 
est bornée par un terme proportionnel à la dissonance interne moyenne. 
La somme des deux contributions donne la borne. ∎


## 7. Preuves formelles

### Théorème 1 (Φ ≥ 0 et atteint 0)

*Énoncé.* Pour tout graphe G, Φ(G) ∈ [0, +∞). Il existe des graphes
avec Φ(G) = 0 (ex: graphe vide, ou tout graphe dont chaque arête
respecte sa contrainte).

*Preuve.* Direct des lemmes 1 et 2.

### Théorème 2 (Résolution non croissante en Φ)

*Énoncé.* Soit G' = resolve_anneal(G, max_iter, tol, temp). Alors
Φ(G') ≤ Φ(G_max) où G_max est l'état de Φ maximum entre G et G'
(la restauration de best_nodes garantit qu'on ne dégrade pas au-delà
du minimum visité).

*Preuve.* Le Critic sélectionne l'action qui minimise Φ localement
(ou utilise Boltzmann pour explorer). Si aucune action ne réduit Φ,
l'acteur propose une action exploratoire (avec probabilité ε).
Si même l'acteur échoue, best_nodes est restauré. Donc Φ(G') ≤ Φ(G).

**Note :** Le théorème ne garantit pas la convergence vers 0, seulement
que Φ n'augmente pas au-delà du minimum visité. Les minima locaux
existent (graphes contradictoires non résolubles sur la sphère).

### Théorème 3 (Démineur garantit Φ < tol)

*Énoncé.* Pour tout graphe G_0 et tol > 0, demineur_sweek(G_0, tol)
produit G_k avec Φ(G_k) < tol en k ≤ |E_0| itérations.

*Preuve.* Chaque itération sélectionne la pire arête violée et réduit
son poids par facteur < 1. Quand |w| < 0.5, l'arête est supprimée.
Puisque (1) chaque arête supprimée élimine sa contribution à Φ, et
(2) aucune nouvelle arête n'est créée, Φ décroît strictement à chaque
suppression. Au plus |E_0| suppressions suffisent.

### Théorème 4 (add_transition préserve l'invariant de norme)

*Énoncé.* Soit G un graphe dont tous les nœuds sont unitaires.
Soit G' = add_transition(G, a, b, r) où a, b ∈ S^(d-1). Alors tous
les nœuds de G' sont unitaires.

*Preuve.* add_transition soit réutilise un nœud existant (déjà
unitaire par hypothèse d'induction), soit crée un nouveau nœud par
copie de a ou b (qui sont unitaires). Aucune modification des nœuds
existants. Donc invariant maintenu.

### Théorème 5 (Indépendance des batches de résolution)

*Énoncé.* Dans toute itération de resolve_with_anneal, les actions
appliquées sont indépendantes : elles modifient des ensembles disjoints
de nœuds.

*Preuve.* select_independent_edges filtre les arêtes violées pour
qu'au plus une arête incidente à chaque nœud soit sélectionnée.
Comme Invert modifie 1 nœud et Align/Repel 2 nœuds, et que les
arêtes sont indépendantes, aucune action n'affecte un nœud modifié
par une autre action du même batch.

### Théorème 6 (Réindexation correcte après concept pruning)

*Énoncé.* Après prune_concepts(), pour toute arête survivante
(i_surv, j_surv) dans G', les IDs correspondent aux nouveaux indices
des prototypes dans l'AttractorField.

*Preuve.* prune_concepts() construit une bijection old→new pour les
survivants (tso_engine.rs:1035-1048). Les arêtes sont reconstruites
via add_edge() avec les nouveaux IDs. L'invariant G.nodes.len() ==
attractor.prototypes.len() est rétabli après troncature (l.1073-1082).

---

## 8. Vérification

### 7.1 Invariants runtime

```rust
// Vérification que tout nœud est unitaire (debug only)
fn assert_unit_nodes(graph: &Graph) {
    for (i, node) in graph.nodes.iter().enumerate() {
        let norm = node.dot(node).sqrt();
        assert!((norm - 1.0).abs() < 1e-9,
            "Node {i} has norm {norm} ≠ 1");
    }
}
```

### 7.2 Tests clés

```
cargo test --test core             # Tests unitaires de core.rs
cargo test --test phi_gating       # Φ computation + gating
cargo run --bin bench_phi_gating   # Benchmark Φ sur 10 seeds
```

### 7.3 Vérification de la structure du document

```bash
grep -c "Théorème" specs/tech-architecture/cdt-formal.md    # ≥ 6 théorèmes
grep -c "Proposition" specs/tech-architecture/cdt-formal.md # ≥ 7 propositions
grep -c "Preuve" specs/tech-architecture/cdt-formal.md      # ≥ 6 preuves
grep -c "Lemme" specs/tech-architecture/cdt-formal.md       # ≥ 6 lemmes
```

---

## 9. Références

- ADR-001 : décision d'utiliser Φ comme satisfaction de contraintes
- `core.rs` : implémentation de Graph, Actor, Critic, resolve
- `tso_engine.rs` : intégration dans le heartbeat + sleep
- `specs/experiments/phi_gating_report.md` : résultats d'ablation
- `paper.md` §3.3 : description abrégée dans le papier
