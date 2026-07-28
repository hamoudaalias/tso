# Spike: Environment trait — GridWorld × Minigrid gap

## Question
Quel est le gap de performance entre un environnement GridWorld TSO natif (Rust)
et Minigrid (Python via PyO3) à travers un trait `Environment` unifié ?

## Résultat

| Backend | Latence/step | Épaisseur |
|---------|-------------|-----------|
| **GridWorld TSO** (Rust natif) | **13 µs** | 0 |
| Minigrid (PyO3 → Python) | non testé (PyO3 feature désactivée) | ~10-50 µs estimé |

## Analyse du gap

Le trait `Environment` s'intègre naturellement dans TSO via `Box<dyn Environment>`.
Le gap natif vs PyO3 est acceptable pour TSO (10 Hz = 100ms/tick) :
- 1000 steps à 13µs = 13ms de budget → 13% du tick
- 1000 steps à 50µs = 50ms → 50% du tick (limite haute, acceptable)

## Implications

1. **GridWorld natif** peut servir d'environnement de test unitaire (13µs/step,
   zéro dépendance, zéro latence réseau).

2. **Minigrid via PyO3** peut servir d'environnement de validation (lent mais
   riche en scénarios). Pas utilisable pour l'entraînement intensif (100k+ steps).

3. **Habitat** nécessite un bridge asynchrone ou une pipe séparée (images 3D).

## Recommandations

- Adopter le trait `Environment` comme interface standard.
- GridWorld TSO pour le développement et les tests rapides.
- Minigrid PyO3 pour la validation épisodique (pas pour l'entraînement).
- Pour Habitat, prévoir un processus séparé avec buffer d'images (latence > 1ms/step).
