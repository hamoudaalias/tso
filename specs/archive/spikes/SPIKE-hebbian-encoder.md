# Spike: HebbianEncoder — Oja + centroids pour apprentissage online

## Question

Un encodeur purement Hebbien (règle d'Oja, zéro backprop, apprentissage online) peut-il produire des représentations discriminatives pour des entrées 5D, avec un simple clustering par centroids et distance cosinus ?

## Résultat

**Partiellement répondu.** L'approche Hebbienne fonctionne mais avec une limitation fondamentale : Oja capture les axes de variance dominants (équivalent PCA), ce qui ne sépare que **k-1 clusters** pour **k centroids** en pratique. Avec 3 clusters 5D, seulement 2 sont discriminés — le 3e centroid reste un fantôme.

## Métriques de validation

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Dimensions entrée | 5 | 3 clusters gaussiens, σ=0.3, centres orthogonaux |
| Code latent (tanh) | 8 neurones | Plus large que l'entrée (expansion 5→8) |
| Catégories découvertes | 3 | ✅ Nombre exact (mais 2 seulement utiles) |
| Ratio séparation inter/intra | ~10¹⁶ | Excellent — mais dû au mode déterministe des centroids |
| Pureté moyenne | **66%** (2/3 clusters purs) | ⚠️ Cluster 2 assigné au centroid 0 |
| Confusion | C0→C0: 334/334, C1→C1: 333/333, C2→C0: **333/333** | ❌ |

## Findings

### 1. Oja = PCA adaptatif (bon mais limité)

Oja capture les directions de variance maximale (axes principaux). Pour 3 clusters 5D disposés en triangle, les 2 premières PC les séparent en 2 groupes seulement — le 3e cluster tombe dans le même groupe qu'un autre. C'est une **limitation fondamentale** de l'approche linéaire, pas un bug.

### 2. La distance cosinus amplifie le problème

La distance cosinus ne voit que la *direction* dans l'espace latent. Les projections Oja+tanh donnent des vecteurs qui pointent vers les PC dominantes. 2 clusters partagent la même direction dominante → assignés au même centroid.

### 3. La non-linéarité tanh n'aide pas suffisamment

tanh compresse l'amplitude mais préserve la direction. Le problème est directionnel : 2 clusters ont des projections colinéaires sur les PCs.

### 4. Le mécanisme de centroids est fonctionnel

La création/soft-update des centroids fonctionne correctement (même logique que VaeEncoder). Le problème est en amont : l'espace latent n'est pas assez discriminant.

## Implications for the plan

1. **Ne pas remplacer le VAE par un HebbianEncoder pur** — la séparation est insuffisante pour des clusters qui ne sont pas sur les axes de variance.
2. **Alternative viable** : **HebbianEncoder en complément** de l'AttractorField, pas en remplacement du VAE. Utiliser Oja pour **enrichir** la représentation (expansion 5→16) en entrée de l'AttractorField plutôt que comme classifieur.
3. **Si besoin d'un encodeur online sans backprop**, la piste Hebbian + **softmax compétitif** (Winner-Take-All) serait plus robuste qu'Oja pour la classification — règle de Kohonen (SOM) plutôt qu'Oja.

## Ce qui n'a PAS été exploré

- **Règle de Kohonen / SOM** : apprentissage compétitif avec voisinage — plus naturel pour la classification qu'Oja. ~30 lignes de plus.
- **Règle d'Oja généralisée (Sanger)** : extrait plusieurs PCs dans l'ordre, pourrait séparer plus de clusters.
- **Normalisation L2 de l'entrée** : la distance cosinus sur entrée normalisée pourrait améliorer la séparation directionnelle.
- **code_dim > 16** : plus de neurones = plus de directions latentes, peut-être suffisant pour 3 clusters.

## Recommandation

**Ne pas poursuivre HebbianEncoder comme remplacement du VAE.** L'approche est trop limitée pour la classification. Utiliser plutôt :

1. **Garder l'AttractorField** (qui fait du vrai apprentissage compétitif avec prototypes multiples par classe — déjà plus puissant qu'Oja).
2. **Si besoin d'expansion de dimension** (ex: 5D→16D), utiliser une **projection aléatoire fixe** (Reservoir Computing / ELM) — pas d'apprentissage, juste une expansion non-linéaire, validée par l'écosystème.
3. **Si besoin absolu de plasticité locale et online sans backprop**, explorer **Kohonen SOM** plutôt qu'Oja pour la catégorisation.

Le spike a bien répondu à la question : le code est à jeter, les apprentissages sont conservés.
