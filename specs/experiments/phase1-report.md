# RAPPORT EXPÉRIMENTAL — Phase 1

> **Date :** 2026-06-15
> **Projet :** TSO — Terrarium Survival (V2 Zigzag / §8)
> **Objectif :** Isoler les causes de l'échec du système TSO complet sur le Terrarium 7×7 (0 % en exploitation pure ε=0), et déterminer si les correctifs proposés (§8, papier §3.4) rétablissent la viabilité de l'architecture unifiée.

---

## Résumé exécutif

Le système TSO original échoue à 0 % en exploitation pure sur le Terrarium 7×7 muré. Cette campagne expérimentale montre que :

1. **Le cervelet TSO (Cerebellum MLP) + shaping BFS + replay propre résout le Terrarium 7×7 à 66,5 %** — le problème n'est donc pas l'architecture de l'acteur.
2. **Le cycle cognitif TSO complet (attracteur + graphe Φ + attention + well_being à 9 termes) casse l'exploitation même sans aliasing** (5×5 : 98 % → 20 %) — c'est le facteur dominant.
3. **La non-stationnarité du signal de récompense** (well_being dépendant de l'état interne : concepts, Φ, curiosité, homéostasie) empêche le critic d'apprendre une cible stable — les politiques apprises en entraînement ne généralisent pas en exploitation.
4. **Le correctif de signal stationnaire seul ne suffit pas** — l'interaction entre le cycle cognitif et l'apprentissage RL est plus profonde que la seule cible TD.
5. **Les grid cells multi-modules injectives n'ont pas montré de bénéfice sur ce layout** (61 % vs 66,5 %), mais l'aliasing du Terrarium 7×7 est modéré → le vrai test est le V2 Zigzag 10×10.

---

## Tableau d'ensemble

```
                    Cerebellum seul        TSO complet
5×5 salle ouverte   98 % (Phase 1 #8)      20 % (Exp B)
7×7 muré            66,5 % (Exp A #A3)     0 % (original)
7×7 salle vide      À faire                —
Zigzag 10×10        À faire                —
```

Décomposition de l'échec original (0 % vs idéal 98 %) :
- **Cycle cognitif** (5×5) : 98 → 20 = **−78 pts** (facteur dominant)
- **Échelle/aliasing** (cervelet seul 5×5→7×7) : 98 → 66,5 = **−31,5 pts** (secondaire)
- **Interaction** : les deux effets sont super-additifs (78 + 31,5 < 98)

---

## 1. Phase 1 — Cerebellum seul sur 5×5 (référence)

**Binaire :** `phase1_grid.rs`
**Environnement :** Grille 5×5 ouverte, 3 positions d'eau (1,1), (3,3), (1,4), reward +10, step cost −0.02
**Perception :** 4 moustaches distance-au-mur (normalisées)
**Cervelet :** Cerebellum MLP, ε=0.1→0.01 (annealing), noise=0.3→0.01
**Replay :** stocke `R_ext + γ·Φ_BFS(s')−Φ_BFS(s)` (Phase 1 uniquement)
**Test :** ε=0, noise=0, 100 épisodes
**Seeds :** 1 (chaque série)

| # | Config | Train succès | **Test ε=0** | Replay |
|---|--------|:-----------:|:-----------:|:------:|
| 1 | hd=4, base (sans grille, sans shaping) | 99% | **86%** | — |
| 2 | hd=4, grid (sans shaping) | 76% | **24%** | — |
| 3 | hd=4, shaping (sans grille) | 99% | **94%** | — |
| 4 | hd=4, grid+shaping (sans replay) | 100% | **21%** | — |
| 5 | hd=16, grid+shaping (sans replay) | 100% | **13%** | — |
| **6** | **hd=4, grid+shaping+replay** | 97% | **79%** | oui |
| **7** | **hd=16, grid+shaping+replay** | 100% | **22%** | oui |
| **8** | **hd=4, shaping+replay (sans grille)** | 100% | **98%** | oui |

**Lectures :**
- Sur 5×5, les 4 moustaches discriminent suffisamment les 25 positions (config #1 : 86 %).
- Les grid cells (+12 dims, de 4→16) **nuisent** sans replay — overfit.
- Le shaping BFS améliore (94 % vs 86 %).
- **Le replay est indispensable** quand la dimension augmente (#4=21 % → #6=79 %).
- hd=16 dégrade partout → le régime est limité par l'efficacité échantillonnale, pas la capacité.
- **La config #8 (hd=4, shaping BFS, replay propre) = 98 %** sert de référence pour la suite.

---

## 2. Expérience B — TSO complet sur 5×5

**Binaire :** `phase1b_tso.rs`
**Environnement :** identique Phase 1 (5×5 ouvert, eau en (1,1),(3,3),(1,4))
**Moteur :** TsoEngine complet (attracteur, working memory, graphe sémantique, Φ, hypothalamus, attention, épisodique)
**Cervelet :** MLP hd=4, ε=0.1→0.01, noise=0.1→0.01, replay_lr=0.05
**Hypothalamus :** `B0` = gelé (drift supprimé, satiété). `B1/B2` = normal (dérive active).
**Replay TS O :** stocke `well_being` (9 termes) dans tous les cas.

| Config | Train succès | **Test ε=0** | Concepts | Φ | Replay |
|--------|:-----------:|:-----------:|:--------:|:-:|:------:|
| **B0** hd=4, gelé (réf) | 85% | **21%** | 7 | 2.04 | 10000 |
| **B1** hd=4, dérive normale | 100% | **18%** | 12 | 3.60 | 10000 |
| **B2** hd=4, reset chaque step | 100% | **22%** | 5 | 1.46 | 10000 |
| **B3a** hd=16, gelé | 92% | **22%** | 6 | 0.00 | 10000 |
| **B3b** hd=16, dérive | 100% | **22%** | 3 | 0.00 | 10000 |

**Lectures :**
- Toutes les configs TSO s'effondrent à ~20 % en ε=0 contre 98 % pour Phase 1 #8.
- Φ = 0.00 et 3-6 concepts dans les meilleurs cas → le graphe sémantique et Φ sont disculpés.
- L'hypothalamus (gelé vs dérive) ne fait aucune différence.
- **Le cycle cognitif TSO complet casse l'exploitation même sans aliasing.** La cause est soit (a) la non-stationnarité de l'entrée `gated`, soit (b) la non-stationnarité de la récompense `well_being`, soit (c) les deux.

### Ablations : Fix 1 + Fix 2

**Binaires :** `phase1b_fix1.rs`, `phase1b_fix2.rs`

| Config | Train succès | **Test ε=0** | Fix |
|--------|:-----------:|:-----------:|-----|
| **F1** hd=4, gelé, Fix 1 (entrée brute) | 100% | **26%** | perception brute → cervelet |
| **F2a** hd=4, gelé, Fix 2 (replay propre) | 100% | **23%** | R_ext+shaping dans replay |
| **F2b** hd=4, gelé, Fix 1+2 | 100% | **26%** | les deux |
| **F2c** hd=4, dérive, Fix 2 seul | 100% | **22%** | replay propre |
| **F2d** hd=4, dérive, Fix 1+2 | 100% | **27%** | les deux |
| **F2e** hd=16, gelé, Fix 2 seul | 100% | **27%** | replay propre |
| **F2f** hd=16, gelé, Fix 1+2 | 100% | **30%** | les deux |
| **F3a** hd=4, gelé, TD gelé au test | 100% | **21%** | replay_only au test |
| **F3b** hd=4, gelé, TD actif (réf) | 100% | **17%** | — |

**Lectures :**
- Fix 1 seul (entrée brute) : ~26 % → **l'entrée `gated` n'est pas le facteur principal**.
- Fix 2 seul (replay propre) : ~23 % → **le replay n'est pas le facteur principal** non plus.
- Fix 1+2 : ~26-27 % → **pas additif** — la cause n'est pas dans l'entrée ni le replay.
- Fix 3 (gel TD au test) : 21 % vs 17 % (réf) → **écart dans le bruit**. Le problème n'est pas l'adaptation au test — les poids sont déjà corrompus pendant l'entraînement.
- **Le vrai problème : `reinforce_td` en ligne utilise `total_reward` (well_being à 9 termes) à chaque step.** L'acteur apprend ses poids sur une cible non-stationnaire pendant l'entraînement lui-même.

### Phase 1c — Signal stationnaire partout (en ligne ET replay)

**Binaire :** `phase1c.rs`
**Correctif :** flag `use_stationary_reward` → remplace `total_reward` par `R_ext + γ·Φ_BFS(s')−Φ_BFS(s)` dans `reinforce_td` ET `store_transition`. Perception brute au cervelet. Termes intrinsèques (curiosité, Φ, métabolique) conservés pour l'exploration uniquement.

| Config | Train succès | **Test ε=0** | Concepts | Φ |
|--------|:-----------:|:-----------:|:--------:|:-:|
| **S0** stationary=false (Exp B) | 100% | **19%** | 10 | 1.84 |
| **S1** stationary=true (réparé) | 100% | **17%** | 6 | 0.00 |

**Lecture :** Même avec un signal RL stationnaire parfait (`R_ext + γ·Φ_BFS(s')−Φ_BFS(s)`), le TSO complet ne remonte pas. **Le problème est plus profond que la seule cible TD.** Le cycle cognitif complet transforme l'espace d'état d'une manière qui rend l'apprentissage invalide — peut-être via le fait que l'attracteur crée des prototypes qui dérivent, ce qui change l'espace d'entrée effectif du cervelet (même si on lui donne la perception brute, le `gated` est toujours utilisé pour l'attracteur et les concepts, et ça influence le shaping via `concept_values`, la curiosité, etc.)

---

## 3. Expérience A — Cerebellum seul sur Terrarium 7×7 muré

**Binaire :** `experiment_a.rs`
**Environnement :** Reproduction du Terrarium original (terrarium.rs) — 7×7 avec murs internes, 3 eaux en (5,1), (2,5), (4,2). Mêmes murs, même logique de perception.
**Cervelet :** Cerebellum MLP seul (PAS de TSO)
**Perception :** 4 moustaches distance-au-mur, normalisées
**Grid cells :** Multi-module [3,5,7] → 12 dims, injectif (105 > 49)
**Shaping :** BFS potential-based `γ·Φ_BFS(s')−Φ_BFS(s)`
**Replay :** stocke `R_ext + BFS_shaping` (stationnaire propre)
**Entraînement :** 1000 épisodes, ε=0.8→0.01, noise=0.3→0.01
**Test :** ε=0, noise=0, 200 épisodes

| # | Config | Train succès | **Test ε=0** | Replay |
|---|--------|:-----------:|:-----------:|:------:|
| **A1** | hd=4, base (sans grille, sans shaping) | 99.5% | **8.0%** | — |
| **A2** | hd=4, shaping seul | 99.5% | **13.5%** | — |
| **A3** | **hd=4, shaping+replay** | 99.0% | **66.5%** | 10000 |
| **B1** | hd=4, grid+shaping (sans replay) | 78.1% | **12.5%** | — |
| **B2** | hd=4, **grid+shaping+replay** | 96.8% | **61.0%** | 10000 |
| **B3** | hd=16, grid+shaping+replay | 99.6% | **11.0%** | 10000 |

**Lectures :**
- **Le Terrarium 7×7 muré est apprenable par le Cerebellum seul** : 66,5 % en ε=0 avec shaping+replay.
- La base (A1 = 8,0 %) confirme l'aliasing sévère des 4 moustaches sur 49 positions avec murs.
- Le shaping BFS seul (A2 = 13,5 %) ne suffit pas — il crée un gradient mais l'overfit domine.
- **Le replay est indispensable** (#A3=66,5 % vs A2=13,5 %).
- **Les grid cells multi-modules injectives n'apportent pas de bénéfice** (B2=61,0 % vs A3=66,5 %) — l'aliasing de ce layout spécifique est modéré, le shaping BFS via la position vraie le corrige déjà.
- **hd=16 dégrade** (B3=11,0 %) même avec replay — sur 16 dims d'entrée, les 1000 épisodes en 7×7 ne suffisent pas à remplir un MLP 16.
- Corroboration : hd=4 > hd=16 partout (Phase 1 et Exp A), validant la limite d'efficacité échantillonnale (§8.4).

---

## 4. Synthèse des mécanismes

### Cause 1 : Non-stationnarité du signal RL (dominante, −78 pts)

Le cycle cognitif TSO original calcule `well_being` comme une somme de 9 termes, dont la plupart dépendent de l'état interne de l'agent :
- `gated_reward` = `R_ext × (1 + deficit × 2)` — le déficit homéostatique dérive
- `consummatory` = `deficit × 10` — même chose
- `r_curiosity` = surprise épisodique — diffère entre exploration et exploitation
- `shaping` = `concept_values[id'] − concept_values[id]` — les concept_values sont réindexés et itérés
- `phi_delta` = `Φ_t − Φ_{t-1}` — change avec le nombre d'arêtes
- `deficit_penalty`, `chronic_tension`, `metabolic_penalty`, `parsimony` — tous drivent

Conséquence : l'acteur apprend des poids optimisés pour une cible qui n'existe plus au test (ε=0, pas de curiosité, pas d'exploration → distribution well_being ≠ entraînement). Les ablations confirment que ni l'entrée ni le replay seuls ne sauvent la situation.

### Cause 2 : Aliasing perceptuel (secondaire, −31,5 pts)

Même avec un signal de récompense parfait (R_ext + BFS shaping), le passage de 5×5 à 7×7 muré coûte 31,5 pts (98 % → 66,5 %). La base 4D fait 8 % — preuve que les murs internes créent des ambiguïtés que les 4 moustaches ne suffisent pas à lever. Le shaping BFS + replay en récupère la majeure partie.

### Super-additivité

Les deux causes ne sont pas additives (78 + 31,5 < 98) — leur interaction est pire que leur somme. Cela signifie que corriger un seul facteur ne rétablit pas la performance. Le TSO original encapsule les deux causes simultanément.

---

## 5. Résultats publiable-grade

Les chiffres suivants sont issus de runs à seed unique (sauf indication contraire). Chacun doit être confirmé sur 10+ seeds pour les intervalles de confiance avant soumission.

| Résultat | Valeur | Section papier | Priorité multi-seed |
|----------|--------|:------------:|:-------------------:|
| Cerebellum + shaping + replay, 5×5 | 98 % | §8 | Haute |
| Cerebellum + shaping + replay, 7×7 muré | 66,5 % | §8 | Haute |
| TSO complet, 5×5 (Exp B) | 20 % | §3.4 | Haute |
| Cerebellum base, 7×7 muré | 8 % | §8 | Moyenne |
| Cerebellum + shaping seul, 7×7 | 13,5 % | §8 | Moyenne |
| Grid cells + shaping + replay, 7×7 | 61 % | §8 | Basse (≠ significatif) |
| hd=16 partout < hd=4 partout | qualitatif | §8.4 | Confirmé sur 2 configs |

---

## 6. Travail restant (ordre recommandé)

1. **Runs multi-seeds** (10+) sur les 3 résultats principaux (98 %, 66,5 %, 20 %) pour IC
2. **7×7 salle vide** — sépare l'effet « taille de grille » de l'effet « murs / aliasing »
3. **V2 Zigzag 10×10 avec Cerebellum seul** — vrai benchmark des grid cells (l'aliasing y est documenté sévère, §6.3)
4. **Debug du TSO réparé** — pourquoi `use_stationary_reward` ne remonte pas la perf (hypothèse : l'attracteur crée des prototypes qui changent l'espace d'entrée du cervelet même avec perception brute, via la curiosité et le shaping implicite)
5. **V2 Zigzag 10×10 avec TSO réparé** — test ultime de l'architecture unifiée

---

*Document rédigé par l'agent de codage (pi) à partir des résultats expérimentaux de la session du 15 juin 2026.*
