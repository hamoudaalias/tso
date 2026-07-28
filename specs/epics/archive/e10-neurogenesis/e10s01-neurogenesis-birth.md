# e10s01 — Phase 1.5 : Neurogenèse — naissance de nouveaux concepts

## 1. Business narrative

En tant que moteur cognitif TSO, je veux créer de nouvelles classes de
concepts pendant mon cycle sommeil, afin d'explorer de nouvelles régions
de l'espace perceptif et d'augmenter la diversité de mes représentations.

## 2. Prérequis

- `sleep_cycle()` dans `tso_engine.rs` avec ses 5 phases actuelles
- `AttractorField` avec `add_class(proto)` / `n_classes()` / `get_prototype(i)`
- `Graph` avec `add_node(id)` / `add_edge(from, to, weight)`
- `CognitiveConfig` dans `tso_engine.rs`

## 3. Acteurs

- **TSO Engine** : exécute la Phase 1.5 pendant `sleep_cycle()`
- **AttractorField** : reçoit les nouvelles classes
- **Graph** : reçoit les nouveaux nœuds et arêtes

## 4. Flux principal

1. `sleep_cycle()` commence, Phase 1 (replay) s'exécute
2. Phase 1.5 itère sur les concepts existants (0..n_classes)
3. Pour chaque concept, avec probabilité `sleep_neurogenesis_rate`, muter
   le prototype avec du bruit gaussien (σ = sleep_noise_std)
4. `add_class(prototype_muté)` → nouveau concept id
5. `graph.add_node(new_id)` → nœud dans le graphe
6. Connecter à 2-3 voisins aléatoires : `add_edge(new_id, neighbor, 0.1)`
7. Initialiser `concept_maturation[new_id] = sleep_maturation_cycles`
8. Ajouter les entrées dans `concept_novelty_thresholds`, `concept_values`,
   `last_active_step` (valeur 0)
9. Continuer vers Phase 2 (résolution de graphe)

## 5. Flux alternatifs

- **Budget max atteint** : avant la naissance, vérifier
  `n_classes() < sleep_max_concepts`. Sinon, déclencher e10s03 (remplacement).
- **sleep_neurogenesis_rate = 0** : Phase 1.5 est un no-op.
- **Pas de prototypes existants** : Phase 1.5 skip.

## 6. Contraintes et règles métier

- Ne pas créer plus de concepts que `sleep_max_concepts` (défaut 50)
- `sleep_neurogenesis_rate` défaut 0.2 (20% de chance par concept par cycle)
- `sleep_maturation_cycles` défaut 3 (cycles sommeil de protection)
- Le bruit de mutation doit partager `sleep_noise_std` avec le replay existant
- Les IDs des nouveaux concepts sont contigus (add_class retourne n_classes précédent)

## 7. Données

### Ajout à `CognitiveConfig`

```rust
pub sleep_neurogenesis_rate: f64,   // défaut 0.2
pub sleep_max_concepts: usize,      // défaut 50
pub sleep_maturation_cycles: usize, // défaut 3
```

### Ajout à `TsoEngine`

```rust
/// Compteur de maturation pour chaque concept (0 = mature, >0 = période critique).
/// Indexé par concept_id.
concept_maturation: Vec<usize>,
```

## Implementation Steps

**type:** feat  
**risk:** P0  
**context:** domain  
**Context:** Ajout des champs de configuration (CognitiveConfig) et du vecteur concept_maturation (TsoEngine), puis implémentation de la Phase 1.5 dans sleep_cycle() qui crée de nouvelles classes de concepts par mutation bruitée.

## Steps

1. Ajouter `sleep_neurogenesis_rate: f64` (défaut 0.2), `sleep_max_concepts: usize` (50), `sleep_maturation_cycles: usize` (3) à CognitiveConfig avec derive par défaut → verify: `cd tso-engine && cargo check --lib`

2. Ajouter `concept_maturation: Vec<usize>` à TsoEngine, initialisé à `vec![0; n]` dans `new()` et `with_hidden()` → verify: `cd tso-engine && cargo check --lib`

3. Synchroniser `concept_maturation` dans `prune_concepts()` : filtrer les entrées des survivants, idem `last_active_step` et autres tracking vectors → verify: `cd tso-engine && cargo test --lib 2>&1 | grep 'ok\|FAILED' | head -5`

4. Synchroniser `concept_maturation` dans le remap de fin de `prune_concepts()` : old_to_new mapping comme pour les autres vecteurs → verify: `cd tso-engine && cargo test --lib 2>&1 | grep 'ok\|FAILED' | head -5`

5. Ajouter `sleep_neurogenesis_rate` et `sleep_max_concepts` aux tests existants (fill CognitiveConfig correctement) → verify: `cd tso-engine && cargo test --lib 2>&1 | grep 'ok\|FAILED'`

6. Implémenter Phase 1.5 dans sleep_cycle() : itérer sur 0..n_classes, avec probabilité sleep_neurogenesis_rate, muter le prototype (bruit σ = sleep_noise_std), add_class, add_node, connecter à 2-3 voisins aléatoires (poids 0.1), initialiser concept_maturation[new_id], étendre tracking vectors → verify: `cd tso-engine && cargo test -- test_neurogenesis_birth 2>&1 | grep ok`

## Verification Script

```bash
cargo test -- test_neurogenesis_birth 2>&1
# Vérifier : n_classes augmente, maturation = sleep_maturation_cycles, arêtes dans le graphe
cargo test -- test_neurogenesis_budget_max 2>&1
# Vérifier : budget max respecté
cargo test -- test_neurogenesis_rate_zero 2>&1
# Vérifier : rate=0 → pas de naissance
```

## Out of scope

- Modification du cervelet (Cerebellum) ou de l'encodeur (VaeEncoder)
- Changement de l'API publique d'AttractorField

## Risks

- **Désynchronisation tracking vectors** : si concept_maturation n'est pas redimensionné dans tous les paths (new, with_hidden, prune_concepts), panique out-of-bounds. Mitigation : chaque path redimensionne.
- **Explosion du nombre de concepts** : sleep_max_concepts borne. Si rate trop haut, on atteint le max en 1-2 cycles → normal (remplacement prend le relais via e10s03).

## 8. Prior Art

| Candidate | Source | Verdict | Notes |
|-----------|--------|---------|-------|
| **NEAT** (NeuroEvolution of Augmenting Topologies) | [NEAT Paper (UT Austin)](https://www.cs.utexas.edu/ftp/AI-Lab/tech-reports/UT-AI-TR-01-290.pdf) | **compose** | Ajout de nœuds par mutation de connexion (split) + spéciation pour protéger l'innovation. Notre Phase 1.5 est une forme simplifiée : bruit → nouveau concept, connexion aléatoire faible au lieu de split. Pas de crossover ni de spéciation dans TSO.
| **Cascade Correlation** | [rfann crate](https://docs.rs/rfann/latest/rfann/cascade/index.html) | **compose** | Ajoute des neurones cachés un par un, maximisant la corrélation avec l'erreur résiduelle. Notre approche est plus bio-inspirée (bruit + période critique), moins optimisée mathématiquement. L'idée de **geler les poids entrants** est similaire à notre protection anti-pruning.
| **FEAGI Brain Development** | [crates.io](https://crates.io/crates/feagi-brain-development) | **extend** | Synaptogenèse et règles de connectivité bio-inspirées. Architecture très différente (FEAGI est un framework AGI complet), mais valide l'approche Rust pour la plasticité structurelle.
| **neuropool** | [docs.rs](https://docs.rs/neuropool/latest/neuropool/) | **extend** | Neurones LIF avec STDP et cycle de vie thermique pour les synapses. Intéressant mais trop SNN — TSO travaille sur des prototypes distribués, pas des spikes.
| **neuromod** | [GitHub](https://github.com/limen-neural/neuromod) | **extend** | SNN avec redimensionnement dynamique. Même remarque : monde spike, pas notre espace latent.
| **Sleep-like replay (biologie computationnelle)** | [Nature Comms 2022](https://www.nature.com/articles/s41467-022-34938-7) | **compose** | Confirme que le rejeu non-supervisé pendant le sommeil protège les anciens mémoires. Valide notre **Phase 1** (replay) déjà existante, et justifie l'ajout de neurogenèse (Phase 1.5).
| **Neurogenesis in incremental learning** | [arXiv 2018](https://ar5iv.labs.arxiv.org/html/1811.02113) | **compose** | Les réseaux dynamiques surpassent les réseaux statiques à budget mémoire égal en apprentissage incrémental. Valide le **remplacement homéostatique** (e10s03) : un nouveau concept vaut mieux qu'un ancien inactif.

## 9. Tests (Gherkin)

```gherkin
Scenario: Naissance d'un nouveau concept pendant le sommeil
  Given un TsoEngine avec 3 concepts existants (id=0,1,2)
    And sleep_neurogenesis_rate = 1.0 (forcage)
    And sleep_max_concepts = 50
  When sleep_cycle() est appelé
  Then n_classes() augmente d'au moins 1
    And le nouveau concept a maturation = sleep_maturation_cycles
    And le nouveau concept a une arête dans le graphe

Scenario: Budget max empêche la naissance
  Given un TsoEngine avec sleep_max_concepts = 3
    And n_classes() = 3
    And sleep_neurogenesis_rate = 1.0
  When sleep_cycle() est appelé
  Then n_classes() ≤ 3

Scenario: sleep_neurogenesis_rate = 0 désactive la neurogenèse
  Given un TsoEngine avec sleep_neurogenesis_rate = 0.0
  When sleep_cycle() est appelé
  Then Phase 1.5 n'ajoute aucun nouveau concept
```
