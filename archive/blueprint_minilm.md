# Blueprint Révisé : Auto-encodeur Topologique TSO-MiniLM

**Objectif** : Produire un embedding sémantique de dimension 384 purement à partir de la dynamique interne de TSO, sans **aucune** supervision externe, même à l'entraînement. L'architecture apprend à compresser l'état topologique (issu des trois niveaux α, β, γ) en un vecteur compact, par un mécanisme d'auto-encodage linéaire entraîné localement (règle Delta). Le résultat est un modèle de plongement autonome, frugal et événementiel, comparable à MiniLM mais dénué de toute dépendance pré-entraînée.

---

## 1. Les Trois Niveaux Hiérarchiques (rappel)

- **Niveau α (Lexical)** : Graphe de co-occurrences temporelles des mots bruts, construit par R-STDP locale avec fenêtre glissante.
- **Niveau β (Compositionnel)** : Nœuds représentant des syntagmes fréquents, recrutés automatiquement lorsque l'éligibilité d'une paire de nœuds α dépasse un seuil. Chaque nœud β encode une liaison stable.
- **Niveau γ (Conceptuel)** : Calcul du vecteur de **tri-friction** `[support, conflict, novelty]` entre deux ensembles de tokens (prémisse/hypothèse ou fenêtres temporelles), comme dans les phases 20–25.

---

## 2. L'État Topologique Interne (la clé de l'auto-encodeur)

Pour chaque phrase, après traversée complète des niveaux α → β → γ, on forme un **vecteur d'état topologique** `s`, concaténation de :

- **Vecteur β** : Un vecteur binaire de taille fixe `MAX_BETA` (par exemple 512) indiquant quels nœuds compositionnels sont actifs (1) ou inactifs (0).
- **Vecteur γ** : Les trois valeurs continues de tri-friction entre la phrase et elle-même (ou entre deux fenêtres, selon le contexte). Ces valeurs sont normalisées entre 0 et 1.

La dimension totale de `s` est `M = MAX_BETA + 3`. Ce vecteur est une signature purement topologique de la phrase, sans aucun embedding externe.

---

## 3. L'Auto-Encodeur Linéaire à Apprentissage Local

L'architecture est composée de deux matrices :

- **Encodeur (Inverse Motor)** : `W_enc` de taille `(M, 384)`.  
  L'embedding 384-d se calcule par :
  $$h = W_{enc}^T \, s$$

- **Décodeur (Forward Motor)** : `W_dec` de taille `(384, M)`.  
  Reconstruction de l'état :
  $$\hat{s} = W_{dec} \, h$$

L'apprentissage est **auto-supervisé** : il minimise l'erreur de reconstruction $\|s - \hat{s}\|^2$ en utilisant uniquement des **règles Delta locales**, sans rétropropagation.

### Règles de mise à jour (phrase par phrase)

1. **Phase forward** :
   - Calculer `s` (état topologique de la phrase).
   - `h = W_enc^T @ s`
   - `s_hat = W_dec @ h`

2. **Mise à jour du décodeur (W_dec)** :
   Pour chaque neurone de sortie $i$, l'erreur $e_i = s_i - \hat{s}_i$ est disponible localement.
   $$\Delta W_{dec}[i, j] = \eta \cdot h_j \cdot e_i \quad \text{(outer product)}$$

3. **Mise à jour de l'encodeur (W_enc)** :
   On propage l'erreur de reconstruction vers `h` via la transposée du décodeur :
   $$\delta_h = W_{dec}^T \, e$$
   Puis mise à jour :
   $$\Delta W_{enc}[i, j] = \eta \cdot s_i \cdot \delta_h[j]$$

**Propriété cruciale** : Aucune rétropropagation globale, pas de chaînage de gradients. L'apprentissage est entièrement local dans le temps et l'espace, assimilable à une forme de *Predictive Coding* à deux couches.

---

## 4. Routage Événementiel par Friction (Skip-Attention)

Lors de la traversée d'une phrase, on mesure en continu une **friction locale** entre α et β (dissimilarité cosinus).

- Si la friction dépasse un seuil → **alignement géométrique** (Double Mapping local ou recrutement de nœud β supplémentaire). Ce calcul coûteux n'est effectué que sur les ruptures sémantiques.
- Sinon (friction basse) → l'information circule linéairement.

Ce mécanisme est intégré dans la construction de `s` : le vecteur β est d'autant plus riche que des fusions ont été déclenchées.

---

## 5. Apprentissage en Ligne (Single-Pass) sur Corpus Non Annoté

L'entraînement de l'auto-encodeur se fait **sans aucune cible externe**, en parcourant un flux de texte brut (Wikipedia, livres, etc.). Pour chaque phrase :

1. Remettre à zéro la fenêtre α et les activations β.
2. Défiler les tokens → mise à jour des niveaux α, β, γ → obtention de `s`.
3. Forward auto-encodeur → `h`, `s_hat`.
4. Appliquer les règles Delta sur `W_dec` et `W_enc`.
5. (Optionnel) Moduler le taux d'apprentissage par la friction maximale rencontrée dans la phrase, pour favoriser l'apprentissage sur les structures surprenantes.

Cette boucle est purement locale, événementielle (seuls les nœuds actifs sont mis à jour) et ne nécessite qu'une seule passe sur les données.

---

## 6. Génération de l'Embedding Final

Après apprentissage, l'encodeur `W_enc` est figé. Pour une phrase quelconque :

- On calcule son état topologique `s`.
- L'embedding TSO-MiniLM est simplement `h = W_enc^T s`, un vecteur de 384 dimensions.

Utilisable directement en similarité sémantique, classification, recherche, etc.

---

## 7. Conformité Totale aux Principes TSO

| Principe | Respect |
|----------|---------|
| **Zéro rétropropagation globale** | Règles Delta locales uniquement, pas de gradient global. La dynamique α/β/γ est non supervisée (R-STDP, fusion par seuil). |
| **Plasticité locale** | Poids de α (R-STDP) et β (fusion) ajustés localement. W_enc/W_dec mis à jour avec des signaux d'erreur locaux. |
| **Calcul événementiel (friction)** | Routage déclenchant les opérations lourdes uniquement en cas de friction élevée. |
| **Autonomie sémantique** | Aucun embedding pré-entraîné, ni pour l'état, ni pour la supervision. Tout émerge de la topologie et de l'auto-encodage. |

---

## 8. Avantages Énergétiques

- **Complexité** : Construire `s` pour une phrase de longueur L coûte $O(L \cdot (K + \text{fusions}))$ (K = taille de fenêtre). L'auto-encodeur coûte $O(M \cdot 384)$ en forward et update.
- **Comparaison MiniLM** : Transformer pré-entraîné avec attention quadratique vs. quelques milliers de nœuds topologiques + deux matrices (~200k paramètres pour 512×384). Bien plus rapide et frugal, surtout sur séquences longues.
- **Qualité attendue** : L'état topologique capture déjà des relations sémantiques (phases 20–25). L'auto-encodeur apprend une compression qui les préserve.

---

## 9. Prochaines Étapes (Phase 26 révisée)

1. **Implémenter `hierarchy.py`** avec la dynamique α, β, γ et la génération de `s`.
2. **Implémenter `autoencoder.py`** avec `W_enc`, `W_dec` et les règles Delta.
3. **Entraîner en single-pass** sur un corpus de texte brut (ex. 100K phrases Wikipedia).
4. **Évaluer les embeddings 384-d** sur STS-Benchmark et comparer à MiniLM, à un embedding aléatoire et à un PCA de l'état topologique.
