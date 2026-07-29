# SIMULATION — Mock Reviewer sur papier TSO

## Reviewer : Senior Researcher en systèmes neuro-inspirés

### Évaluation générale

Ce papier décrit une architecture intéressante mais **sur-vend ses résultats**
de manière problématique pour une soumission académique.

### Problèmes majeurs

**1. Baseline inéquitable**
Le baseline n'est pas un actor-critic, c'est une régression logistique
multiclasse (couche linéaire 147D→4 actions). Aucun RL moderne n'est
comparé. Un DQN à 2 couches (ou même un MLP à 1 couche cachée avec
Target Q) battrait ce baseline. L'affirmation "+105%" est vraie mais
trompeuse — c'est +105% du pire baseline possible.

**2. Absence de comparaison avec la catégorisation standard**
L'article affirme que la "catégorisation par prototypes apporte un
avantage" mais ne la compare pas à K-means, GMM, ou un classifieur
naïf. L'avantage mesuré est celui de *n'importe quelle abstraction*
vs *aucune abstraction*, pas de l'approche spécifique.

**3. Φ non isolé**
Le mécanisme signature du papier — friction topographique comme
déclencheur de calcul — n'est pas testé en isolation. On ne sait pas
si c'est Φ qui fait la différence ou simplement l'architecture en
deux étages.

**4. VAE n'est pas "online"**
La littérature VAE online (Kingma 2013, Rezende 2014) implique
un entraînement en continue. L'ADR du projet confirme que le VAE
est pré-entraîné hors-ligne. Le terme "VAE online" est incorrect.

**5. Pas de validation statistique**
10 seeds ne suffisent pas pour affirmer une supériorité avec σ=0.34.
Un intervalle de confiance ou un test de Welch est nécessaire.

### Problèmes mineurs

- "Zéro dépendance Python" : vrai mais non pertinent — l'article compare
  avec des travaux en Python qui utilisent PyTorch. Le choix technique
  n'est pas une contribution scientifique.
- "45 tests unitaires valident" : un test unitaire ne valide pas un
  comportement émergent. Reformulation nécessaire.
- Travaux futurs trop vagues ("en cours").

### Verdict
**Ne pas soumettre en l'état.** Trois révisions majeures nécessaires :
1. Baselines plus fortes (MLP, DQN simple)
2. Isolation de Φ (ablation)
3. Validation statistique (intervalle de confiance, test de Welch)
