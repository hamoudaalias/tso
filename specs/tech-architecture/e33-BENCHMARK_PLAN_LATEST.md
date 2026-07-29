# Benchmark Plan: Validation expérimentale MiniGrid

## 1. Problèmes identifiés (review)

| # | Problème | Impact |
|---|----------|--------|
| 1 | Baseline unique (actor-critic linéaire) | Strawman — on ne sait pas si TSO bat un vrai baseline |
| 2 | 3 environnements × dimension confondus | On compare des tasks différentes, pas juste la dimension |
| 3 | Pas de test de significativité | 10 seeds donnent des σ, mais pas de p-valeur |
| 4 | Unités du tableau (1.03 / 2.13) | Reward moyen par épisode, mais >1 sur DoorKey mérite explication |
| 5 | Pas d'ablation structurée | On ne sait pas quel composant apporte quoi |

## 2. Scénarios

### 2.1 Phase 1 — Baselines (30 runs, ~15 min)

| ID | Description | Risque | Niveau | Seeds | Métrique |
|----|-------------|--------|--------|-------|----------|
| SC-P0-01 | Linear actor-critic (147D) | P0 | E2E | 10 | reward/épisode |
| SC-P0-02 | CNN + PPO (MiniGrid default) | P0 | E2E | 10 | reward/épisode |
| SC-P0-03 | MLP (256) + PPO | P0 | E2E | 10 | reward/épisode |

### 2.2 Phase 2 — TSO ablations (50 runs, ~25 min)

| ID | Description | Risque | Niveau | Seeds |
|----|-------------|--------|--------|-------|
| SC-P0-04 | TSO + attracteur seul | P0 | E2E | 10 |
| SC-P0-05 | TSO + VAE (16D) seul | P0 | E2E | 10 |
| SC-P0-06 | TSO + VAE + attracteur | P0 | E2E | 10 |
| SC-P0-07 | TSO + VAE + FPI | P0 | E2E | 10 |
| SC-P0-08 | TSO + VAE + FPI + attracteur | P0 | E2E | 10 |

### 2.3 Phase 3 — Dimension contrôlée (40 runs, ~20 min)

Même environnement (MiniGrid DoorKey 7×7) avec observation réduite :

| ID | Description | Risque | Niveau | Seeds |
|----|-------------|--------|--------|-------|
| SC-P1-09 | TSO sur obs 5D (whiskers seulement) | P1 | E2E | 10 |
| SC-P1-10 | TSO sur obs 25D (grid encoding) | P1 | E2E | 10 |
| SC-P1-11 | TSO sur obs 147D (RGB) | P1 | E2E | 10 |
| SC-P1-12 | Linéaire sur obs 147D | P1 | E2E | 10 |

→ Ici la seule variable est la dimension d'entrée, pas l'environnement.

### 2.4 Phase 4 — Significativité (2 runs comparatives)

| ID | Description | Risque | Niveau |
|----|-------------|--------|--------|
| SC-P1-13 | Welch t-test TSO(max) vs CNN+PPO sur 10 seeds | P1 | Analyse |
| SC-P1-14 | Cohen's d pour taille d'effet | P1 | Analyse |

## 3. Métriques et analyse

### Métriques primaires
- **Reward cumulé par épisode** (moyenne ± σ sur 10 seeds)
- **Taux de succès** par épisode (porte ouverte)
- **Pas de convergence** : épisodes pour atteindre 90% du reward max

### Analyse statistique
- Welch t-test (two-sided, α=0.05) entre TSO(max) et chaque baseline
- Cohen's d pour taille d'effet
- Intervalle de confiance bootstrap (95%) sur la médiane

### Unités et bornes
- DoorKey reward max/théorique : ~0.9–1.0 par épisode (ouvrir porte + atteindre but)
- Reward >1.0 possible si épisode long (reward négatif cumulé? ou steps > max?) → **à documenter**
- Valeurs actuelles (1.03–2.13) suggèrent soit reward non borné, soit scaling différent

## 4. Implémentation

### Nouvelles dépendances
- minigrid_env.rs existant — ajouter mode obs tronquée (5D, 25D)
- Script Python optionnel pour CNN+PPO (hors kernel Rust)
- Script R ou Python pour analyse statistique

### Fixture
- Seed fixe 0..9 avec rand::SeedableRng::seed_from_u64
- Même grille DoorKey 7×7 pour tous les runs
- 200 épisodes entraînement / 50 épisodes test (ε=0)

## 5. Hors scope
- Benchmark Procgen / Habitat (travaux futurs)
- Analyse de latence temps réel
- Comparaison avec SNN purs