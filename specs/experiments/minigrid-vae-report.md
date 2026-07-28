# Rapport : TSO × Minigrid — Pipeline VAE pré-entraîné

## Résumé
TSO tourne sur Minigrid (EmptyEnv-8×8) via PyO3 avec un VAE pré-entraîné
comme encodeur. Succès : 12% sur 50 épisodes (ε=0.2). La cause principale
est le dataset synthétique du VAE, pas l'architecture.

## Pipeline

```
Dataset (200 images 147D synthétiques)
    ↓
VAE trainer (147→32→8→32→147, 100 epochs, MSE=0.0026)
    ↓
vae_weights.bin (85 KB, serde)
    ↓
VaeEncoder (déterministe, freeze) dans TsoEngine
    ↓
TSO sur Minigrid EmptyEnv-8×8 (PyO3)
```

## Résultats

| Métrique | Valeur |
|----------|--------|
| Succès 50 épisodes | **12%** (6/50) |
| Temps total | 41.5s |
| Steps/épisode | 90–3159 (médiane ~600) |
| VAE MSE | 0.0026 |
| VAE stabilité | 100% |
| Observation dim | 147D → 8D latent |
| Actions | 7 |

## Analyse des 12%

Le dataset synthétique (cercle + bruit sinusoïdal) ne ressemble pas aux
frames réelles de Minigrid (couleurs vives, murs, agent, but). Le VAE
apprend à reconstruire des cercles, pas des observations Minigrid.
Les centroids 8D ne correspondent à rien de structuré → l'AttractorField
ne peut pas discerner les états.

## Recommandation

1. **Collecter un vrai dataset Minigrid** via PyO3 (200 images suffisent,
   mais extraction 3D → Vec<f64> lente). Alternative : sauvegarder les
   frames au format .npy via Python, les charger en Rust.

2. **Réduire la dimension d'observation** : au lieu de 147D brut (7×7×3),
   utiliser un embedding plus simple (somme des canaux RGB → 49D, ou
   détection de contours → 49D binaire).

3. **Augmenter le nombre d'épisodes** : 50 épisodes à ε=0.2 ne suffisent
   pas pour un espace 8D inconnu. Passer à 500 épisodes avec annealing
   (ε 0.8→0).

## Prochaines étapes

| Priorité | Action | Effort estimé |
|----------|--------|--------------|
| P0 | Collecter 200 vrais frames Minigrid (script Python séparé) | 30 min |
| P1 | Ré-entraîner VAE sur vrai dataset | 5 min |
| P2 | Relancer eval_minigrid avec vrai VAE | 5 min |
| P3 | 500 épisodes avec annealing ε | 10 min |
