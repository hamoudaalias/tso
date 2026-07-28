# ADR 001 : Séparation Encodeur/Classifieur et rejet d'Oja pour la catégorisation

**Statut :** Accepté
**Date :** 2026-10-10
**Source :** Spike `SPIKE-hebbian-encoder.md`

## Contexte

Le VAE existant nécessite un pré-entraînement hors-ligne (batch, backprop). Nous cherchions un encodeur purement Hebbien (règle d'Oja) pour un apprentissage online, zéro backprop, compatible ndarray, qui remplacerait le VAE dans la boucle cognitive de TSO.

## Décision

1. **Conserver l'`AttractorField` comme seul classifieur de TSO.** Il utilise un apprentissage compétitif (Winner-Take-All avec attraction/répulsion) qui crée une tessellation de Voronoï — frontières de décision non-linéaires, biologiquement plausible, local et online.
2. **Abandonner la règle d'Oja pour toute tâche de catégorisation ou classification.**
3. **Maintenir la séparation des rôles** dans le trait `Encoder` : l'encodeur **projette** (réduction/changement de dimension), le classifieur **catégorise**.
4. **Noter comme piste future** la projection aléatoire fixe (Reservoir / Random Projection) pour le scaling dimensionnel sans backprop — à activer le jour où les entrées dépassent la 5D (ex: vision 4096D).

## Justification

La règle d'Oja est un algorithme de PCA adaptatif : elle capture les axes de variance maximale. Pour des distributions non-linéaires (ex: 3 clusters en triangle dans un espace 5D), Oja ne sépare que *k-1* clusters car les projections sont colinéaires sur les PCs dominants. L'`AttractorField`, avec ses prototypes multiples par classe (k voisins), crée des frontières localement linéaires qui couvrent des distributions arbitrairement complexes.

Un spike sur données synthétiques 5D a confirmé : pureté de classification 66% (2/3 clusters), échec systématique sur le 3e cluster.

## Conséquences

- **Positives** : L'`AttractorField` reste le classifieur unique, sans modification. Pas de nouveau code à maintenir.
- **Négatives** : Le VAE hors-ligne reste nécessaire pour la réduction dimensionnelle non-linéaire (vision). La règle d'Oja n'apporte rien de plus que l'`AttractorField` existant.
- **Neutres** : La piste `RandomProjection` est documentée pour le futur mais n'est pas prioritaire. Le trait `Encoder` peut accueillir une projection aléatoire comme nouveau variant sans changer le classifieur.
