# Spec: Protocole "but tournant" — analyse d'adaptation

## Problème
Le RotatingT change de but tous les N épisodes mais le benchmark ne mesure
que le reward moyen global. On ne sait pas :
- Combien d'épisodes TSO met à s'adapter après un changement
- S'il oublie l'ancien but ou s'il garde les stratégies
- Si les sous-systèmes (épisodique, attracteur) aident l'adaptation

## Protocole

### Environnement
- RotatingT 5×5, 4 buts (2 observations distinctes)
- Switch tous les 50 épisodes
- 200 épisodes total = 4 phases

### Métriques par phase
- Avant : reward moyen épisodes 40-49 (dernier 10 avant switch)
- Après : reward moyen épisodes 50-59 (premier 10 après switch)
- Courbe : reward par épisode, lissée sur 3 épisodes
- Convergence : épisode où le reward dépasse 80% du plateau final

### Agents testés
1. Linear AC (référence — pas de mémoire, doit réapprendre)
2. TSO attracteur (catégorisation seule)
3. TSO full (tous sous-systèmes)
4. TSO + épisodique (mémoire des phases précédentes)

### Analyse
- Si TSO full converge plus vite que linear après switch → validation de Φ
- Si TSO + épisodique converge encore plus vite → la mémoire épisodique aide
- Si attracteur seul réapprend aussi vite que full → les autres sont inutiles

### Visualisation
`bench_adapt.rs` produit :
- Table : "épisodes pour converger après switch" par config
- Table : "reward avant/après switch" par config
- Détection : est-ce que l'agent utilise l'alias pour transférer ?
