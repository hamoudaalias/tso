# ADR-001 : Tension cognitive Φ comme satisfaction de contraintes

## statut
Accepté

## contexte
L'agent a besoin d'un signal interne d'« anxiété » ou de « dissonance cognitive » pour détecter les contradictions dans son modèle du monde et agir pour les résoudre. Ce signal doit être :
- Mesurable (une valeur numérique)
- Minimisable (l'agent peut apprendre à le réduire)
- Directement lié à la cohérence du modèle interne

## décision
Utiliser un graphe sémantique dont les nœuds sont des vecteurs unitaires (sur la sphère ℝⁿ) et les arêtes sont des contraintes :
- Implication (+1, +2) : le produit scalaire doit être ≥ γ=0.7
- Exclusion (-1) : le produit scalaire doit être ≤ ε=0.1

Φ = somme des violations = ∑ max(0, γ − dot(u,v)) + ∑ max(0, dot(u,v) − ε)

## conséquences
- + Signal interprétable géométriquement
- + Résolution possible par recuit simulé (Invert, Align, Repel)
- + Passe à l'échelle avec resolve_parallel
- + Démineur permet de forcer Φ→0 en supprimant les arêtes violées
- − Complexité O(|E|) pour le calcul de Φ
- − Les vecteurs sur la sphère limitent l'espace des représentations possibles
