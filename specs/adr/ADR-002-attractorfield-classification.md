# ADR-002 : Classification par prototypes (AttractorField)

## statut
Accepté

## contexte
L'agent reçoit des perceptions sensorielles continues (4 distances de moustaches). Il doit les catégoriser en concepts discrets pour pouvoir raisonner dessus (mémoire épisodique, graphe sémantique). Les catégories ne sont pas connues à l'avance — elles doivent être découvertes dynamiquement.

## alternatives envisagées
- Réseau de neurones supervisé : besoin d'étiquettes
- k-means : pas d'apprentissage incrémental
- SOM (Kohonen) : plus complexe, pas de seuil de nouveauté explicite
- VQ (Vector Quantization) simple : pas de prototypes multiples par classe

## décision
Utiliser un AttractorField à prototypes multiples par classe avec :
- Apprentissage hebbien compétitif (attraction/répulsion)
- Seuils adaptatifs par concept (gain 0.05, ratio cible 0.6)
- Création dynamique de concepts quand la distance > novelty_threshold
- Élagage des concepts inactifs (>500 pas)

## conséquences
- + Découverte non supervisée de concepts
- + Seuils adaptatifs évitent le sur-ajustement
- + Plusieurs prototypes par classe gèrent les catégories multimodales
- − Sensible à l'ordre de présentation des données
- − Élagage agressif (500 pas) peut supprimer des concepts utiles rares
- − Réindexation complète après élagage (coûteuse)
