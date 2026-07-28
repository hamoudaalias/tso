# e10s03 — Homéostasie : remplacement du concept le moins actif

## 1. Business narrative

En tant que système cognitif à budget limité, je veux remplacer le
concept le moins utile quand le nombre maximum de concepts est atteint,
afin de maintenir un pool de représentations fraîches sans exploser
en mémoire.

## 2. Prérequis

- e10s01 (Phase 1.5, `sleep_max_concepts`)
- `prune_concepts()` existant avec le mécanisme de remap

## 3. Flux

### 3.1 Avant toute naissance (Phase 1.5)

Quand `n_classes() >= sleep_max_concepts` :
1. Trouver le concept avec le plus petit `last_active_step`
   (le moins activé récemment)
2. Exclure les concepts en période critique (maturation > 0)
3. Si tous les concepts sont en période critique et le budget est plein,
   skip la neurogenèse pour ce cycle
4. Supprimer le concept choisi :
   - Retirer son prototype de `attractor.prototypes`
   - Retirer son nœud du graphe
   - Retirer ses arêtes
   - Retirer ses entrées dans tous les vecteurs de tracking
   - Remapper les IDs (via le mécanisme existant dans `prune_concepts`)
5. Créer le nouveau concept à la place

### 3.2 Mécanisme de remplacement

Utiliser la même logique que `prune_concepts()` mais pour UN concept
à la fois :

```rust
fn replace_least_active_concept(&mut self) -> Option<usize> {
    let n = self.attractor.n_classes();
    if n == 0 || n < self.sleep_max_concepts { return None; }

    // Trouver le moins actif (hors période critique)
    let target = (0..n)
        .filter(|&i| self.concept_maturation.get(i).copied().unwrap_or(0) == 0)
        .min_by_key(|&i| self.last_active_step.get(i).copied().unwrap_or(0));

    let target = target?;

    // Marquer comme à remplacer : mettre last_active_step = 0
    // pour que prune_concepts() le nettoie puis créer le nouveau
    // ... ou implémenter le retrait direct ici.

    Some(target)
}
```

## Implementation Steps

**type:** feat  
**risk:** P1  
**context:** domain  
**Context:** Quand le nombre de concepts atteint `sleep_max_concepts`, Phase 1.5 déclenche le remplacement du concept le moins actif (hors période critique) avant la création d'un nouveau.

## Steps

1. Implémenter la fonction `find_least_active_concept(&self) -> Option<usize>` : parcourir 0..n_classes, filtrer ceux avec `concept_maturation[i] == 0`, trouver le `min_by_key` sur `last_active_step[i]` → verify: `cd tso-engine && cargo check --lib`

2. Intégrer l'appel à `find_least_active_concept` dans Phase 1.5 : quand `n_classes() >= sleep_max_concepts`, trouver le moins actif, le retirer manuellement (prototype du attractor, nœud du graphe, arêtes, tracking vectors), puis créer le nouveau concept à sa place → verify: `cd tso-engine && cargo test -- test_neurogenesis_replacement 2>&1 | grep ok`

3. Assurer que le remap d'IDs après retrait est correct : les IDs des concepts suivants doivent être décalés (utiliser le même mécanisme old_to_new que prune_concepts) → verify: `cd tso-engine && cargo test -- test_neurogenesis_replacement 2>&1 | grep ok`

4. Tester le cas limite : tous les concepts en période critique → skip la naissance et le remplacement pour ce cycle → verify: `cd tso-engine && cargo test -- test_neurogenesis_all_in_critical 2>&1 | grep ok`

## Verification Script

```bash
cargo test -- test_neurogenesis_replacement 2>&1
cargo test -- test_neurogenesis_all_in_critical 2>&1
```

## Out of scope

- Mort neuronale programmée (apoptose multi-cycle) — juste un remplacement immédiat
- Réordonnancement des épisodes stockés (le remap dans prune_concepts gère déjà)

## Risks

- **Remap incorrect** : si old_to_new ne couvre pas tous les IDs (attractor, graph, tracking vectors, episodics), des références pendantes causent des paniques. Mitigation : réutiliser le mécanisme existant de prune_concepts ligne à ligne.
- **Perte de concept utile** : le concept le moins actif pourrait être crucial mais rare. Accepté : l'inactivité est un bon proxy d'inutilité, et la période critique protège les nouveau-nés.

## 4. Tests (Gherkin)

```gherkin
Scenario: Remplacement quand budget atteint
  Given un TsoEngine avec sleep_max_concepts = 5
    And n_classes() = 5
    And sleep_neurogenesis_rate = 1.0
    And un concept est inactif depuis > 100 steps
  When sleep_cycle() est appelé
  Then n_classes() reste ≤ 5
    And le concept inactif a été remplacé par un nouveau

Scenario: Pas de remplacement si tous les concepts sont en période critique
  Given un TsoEngine avec sleep_max_concepts = 3
    And n_classes() = 3
    And les 3 concepts sont en période critique
  When sleep_cycle() est appelé
  Then n_classes() = 3 (aucun remplacement, aucun ajout)

Scenario: Le concept en période critique n'est jamais le remplacé
  Given un TsoEngine avec 2 concepts matures et 1 en période critique
  When replace_least_active_concept() est appelé
  Then le concept remplacé est l'un des 2 matures
```
