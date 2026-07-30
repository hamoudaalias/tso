# Prior Art — Positionnement de TSO
## TSO face à ACT, PonderNet, EBM, SNN et Active Inference

---

## Tableau comparatif

| Dimension | ACT (Graves 2016) | PonderNet (Banino 2021) | EBM | SNN (Dual-LIF) | Active Inference (Friston) | **TSO (Φ gating)** |
|-----------|-------------------|------------------------|-----|-----------------|---------------------------|---------------------|
| **Déclencheur d'arrêt** | Neurone de halting (sigmoid, appris) | Probabiliste (λ_t, Bernoulli) | N/A (pas d'arrêt) | Seuil de potentiel de membrane | Minimum de free energy | **Seuil géométrique Φ > γ** |
| **Signal de tension** | Coût de ponder N+R | KL divergence vs prior géométrique | E(x) énergie scalaire | Potentiel de membrane | Free energy F | **Φ = Σ violations de contraintes** |
| **Structure du signal** | Scalaire (somme pas) | Distribution (géométrique) | Scalaire (Boltzmann) | Vectoriel (membrane) | Scalaire (borne sup de surprise) | **Graphe G_t, arêtes typées (impl/excl)** |
| **Apprentissage** | Backprop BPTT + ponder cost | Backprop + KL régularisation | CD / SGLD (MCMC) | STDP / surrogate grad | Gradient sur F(μ,a;s) | **TD(λ) sur récompense + recuit simulé** |
| **Espace** | RNN hidden state | RNN hidden state | X → ℝ | Spike trains V_m | μ (croyances) | **Vecteurs unitaires S^(d-1)** |
| **Optimisation** | Différentiable (soft halting) | Différentiable (expectation) | MCMC | Non-différentiable (spike) | Gradient | **Opérateurs discrets (Invert/Align/Repel)** |
| **Extrapolation** | Limitée (max steps = training) | Oui (distribution géométrique) | Plus de steps MCMC = meilleur | N/A | N/A | **Démineur garantit Φ < tol (Th.3)** |

---

## 1. ACT — Adaptive Computation Time (Graves 2016)

**Mécanisme.** Un neurone de halting émet un poids h_t à chaque pas de
calcul. Quand le budget B = Σ h_t ≥ 1 − ε, la boucle s'arrête. La sortie
est une combinaison pondérée de tous les états intermédiaires.

**Différence avec TSO.**
- ACT apprend à pondérer le calcul via un signal differentiable ; TSO
  utilise un seuil dur sur Φ (skip/no-skip).
- ACT ne mesure pas la cohérence interne d'un modèle du monde ; il
  optimise une métrique de coût de calcul.
- TSO n'apprend pas Φ — Φ est une mesure géométrique directe, pas un
  sous-produit du gradient.

**Similarité.** Les deux systèmes ont un coût proportionnel au nombre
d'étapes de calcul (ponder cost vs résolution par recuit).

---

## 2. PonderNet (Banino et al. 2021)

**Mécanisme.** Distribution de halting probabiliste : à chaque pas t,
λ_t = σ(W_h h_t + b_h). La probabilité de s'arrêter exactement à t est
p_t = λ_t Π_{j=1}^{t-1} (1 − λ_j). Une KL divergence régularise la
distribution vers un prior géométrique.

**Différence avec TSO.**
- PonderNet est probabiliste ; TSO est déterministe (Φ > threshold →
  skip).
- PonderNet peut extrapoler à plus de pas qu'à l'entraînement ; TSO
  n'a pas ce problème — le nombre de pas de résolution est fixe.
- PonderNet optimise une borne (ELBO-like) ; TSO optimise directement
  Φ par recuit simulé.

**Similarité.** Les deux approches ont un terme de régularisation qui
pénalise la complexité de calcul (KL vs chronic_tension).

---

## 3. EBM — Energy-Based Models

**Mécanisme.** Une fonction d'énergie E(x) assigne un scalaire à chaque
configuration. La densité suit Boltzmann : P(x) ∝ e^{-E(x)}.
L'apprentissage minimise E sur les données et la maximise sur des
échantillons MCMC.

**Différence avec TSO.**
- EBM apprend l'énergie par CD/SGLD ; TSO calcule Φ directement par
  somme de violations sur un graphe construit à partir de l'expérience.
- EBM opère sur X ∈ ℝ^d ; TSO opère sur des vecteurs unitaires
  S^(d-1) avec des contraintes typées (implication, exclusion).
- EBM utilise MCMC pour l'inférence ; TSO utilise des opérateurs
  géométriques spécialisés (Invert, Align, Repel) pour la résolution.

**Similarité.** Φ est formellement une fonction d'énergie sur G_t :
Φ(G) ≥ 0, minimisée par la résolution. Les deux cadres mesurent la
dissonance d'un modèle interne.

---

## 4. SNN — Spiking Neural Networks (Dual-LIF)

**Mécanisme.** Chaque neurone est un accumulateur à fuite (LIF). Le
potentiel de membrane V_m(t) suit dV/dt = −αV + I(t). Quand V > seuil,
le neurone émet un spike et se reset. Dual-LIF a deux compartiments
avec des constantes de temps différentes (slow/fast).

**Différence avec TSO.**
- L'état du TSO n'est pas spike-based (pas de trains de spikes, pas
  de STDP). DualLIF est utilisé comme mémoire de travail — intégration
  temporelle continue, pas de communication par spikes.
- Le SNN n'a pas de graphe conceptuel ni de Φ. La "tension" dans un
  SNN est le potentiel de membrane ; dans TSO c'est la violation de
  contraintes sémantiques.

**Similarité.** Dual-LIF (working_memory.rs) est un vrai SNN —
deux LIF avec constantes α_slow (défaut 0.95) et α_fast (défaut 0.5).
TSO réutilise le formalisme LIF pour l'intégration temporelle.

---

## 5. Active Inference / Free Energy Principle (Friston)

**Mécanisme.** Free energy F = E_q[-log p(ψ̇,s,a,μ|ψ)] − H[q] est une
borne sup de la surprise sensorielle. La perception minimise F par
mise à jour des croyances μ ; l'action minimise F en changeant
l'environnement : boucle perception-action complète.

**Différence avec TSO.**
- Active inference formalise la perception ET l'action sous un seul
  principe. TSO sépare la catégorisation (AttractorField) du RL
  (Cerebellum) et de la résolution (Core).
- Free energy est apprise (modèle génératif). Φ est construit
  directement à partir des transitions récompensées.
- Active inference utilise EFE (expected free energy) pour la
  sélection d'action. TSO a un terme EFE optionnel (efe_weight) mais
  la sélection principale est TD(λ).

**Similarité.** Φ et F sont tous deux des mesures de désalignement entre
le modèle interne et l'expérience. Tous deux sont minimisés par
l'action et la mise à jour du modèle.

---

## 6. Position de TSO

**Ce que TSO apporte de nouveau :**

1. **Graphe typé de contraintes.** Les arêtes d'implication (+1,+2)
   et d'exclusion (−1) permettent de représenter des relations logiques
   entre concepts — pas seulement une similarité scalaire.

2. **Opérateurs géométriques de résolution.** Invert, Align, Repel
   sont des actions discrètes sur S^(d-1) avec garantie de
   préservation de la norme. Aucun gradient n'est nécessaire.

3. **Φ comme signal intrinsèque.** La tension cognitive Φ est utilisée
   comme signal d'anxiété pour le gating comportemental — pas seulement
   comme fonction de perte.

4. **Démineur.** Algorithme de sweep garanti (Théorème 3) qui
   force Φ < tol en nombre fini d'itérations, sans gradient.

**Ce que TSO ne fait pas :**
- Pas d'apprentissage de la fonction d'énergie (contrairement aux EBM).
- Pas de halting probabiliste (contrairement à PonderNet).
- Pas de gradient à travers la décision de gating (skip/no-skip dur).
- Pas de STDP ou de communications spike-based (malgré DualLIF).

---

## 7. Références

| Méthode | Papier clé |
|---------|-----------|
| ACT | Graves. "Adaptive Computation Time for Recurrent Neural Networks." 2016. |
| PonderNet | Banino et al. "PonderNet: Learning to Ponder." 2021. |
| EBM | LeCun et al. "A Tutorial on Energy-Based Learning." 2006. |
| EBM moderne | Grathwohl et al. "Your Classifier is Secretly an Energy Based Model." 2020. |
| Dual-LIF | Zhang et al. "DA-LIF: Dual Adaptive Leaky Integrate-and-Fire." ICASSP 2025. |
| Active Inference | Friston et al. "The free-energy principle: a unified brain theory?" Nat Rev Neuro 2010. |
| Φ gating | ADR-001, specs/tech-architecture/cdt-formal.md |
| TSO ablation | paper.md |
