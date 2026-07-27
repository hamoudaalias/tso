# ADR-003 : Mémoire de travail DualLIF + Associative

## statut
Accepté

## contexte
L'agent opère dans un environnement partiellement observable (POMDP). Il a besoin :
1. D'intégrer le contexte sur plusieurs pas de temps (mémoire de travail)
2. De stocker et rappeler des patterns discrets (mémoire associative)
3. De deux échelles temporelles : rapide pour les variations immédiates, lente pour le contexte

## décision
Combiner deux systèmes de mémoire :
- **DualLIF** : deux intégrateurs à fuite (LIF) en parallèle — lent (α=0.95, ~20 pas) et rapide (α=0.5, ~2 pas) — pour l'intégration contextuelle
- **AssociativeMemory** : stockage de vecteurs avec rappel par similarité cosinus (code sparse, analogue à l'hippocampe)

## conséquences
- + Double échelle temporelle fidèle à la hiérarchie des constantes de temps corticales
- + DualLIF encode le contexte, AssociativeMemory stocke les patterns — deux rôles distincts
- + Le rappel cosinus est O(n) en temps linéaire
- − AssociativeMemory ne scale pas bien avec des millions d'entrées
- − Les deux systèmes ne partagent pas de représentation latente commune
