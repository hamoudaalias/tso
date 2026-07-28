# e10s02 — Période critique : protection et hyperplasticité des nouveau-nés

## 1. Business narrative

En tant que concept nouvellement créé, je veux être protégé du pruning
et bénéficier d'une plasticité accrue pendant mes premiers cycles sommeil,
afin d'avoir le temps de m'intégrer dans le réseau conceptuel avant de
pouvoir être élagué.

## 2. Prérequis

- e10s01 (concept_maturation[] présent)
- `prune_concepts()` dans tso_engine.rs
- `sleep_cycle()` avec sa Phase 5 (pruning)

## 3. Comportement de la période critique

### 3.1 Protection anti-pruning

Dans `prune_concepts()`, un concept avec `maturation_counter > 0` est
toujours considéré comme "actif" quel que soit son `last_active_step`.

```rust
// Dans prune_concepts() :
let survivors: Vec<bool> = (0..n)
    .map(|i| {
        self.step_count - self.last_active_step[i] <= threshold
            || self.concept_maturation[i] > 0  // ← nouveau
    })
    .collect();
```

### 3.2 Hyperplasticité

Pendant la période critique, le concept a :
- **Learning rate 3×** dans l'AttractorField (appliqué dans
  `train_step()` quand le concept prédit est en maturation)
- **Seuil de nouveauté 0.5×** (le concept s'adapte plus vite aux
  nouvelles perceptions)

Application :
```rust
// Dans step(), après predict_with_distance() ou train_step() :
if let Some(cid) = self.current_concept_id {
    if cid < self.concept_maturation.len() && self.concept_maturation[cid] > 0 {
        // Appliquer lr boost et seuil réduit
    }
}
```

### 3.3 Décrémentation

À la fin de `sleep_cycle()`, décrémenter tous les compteurs non-nuls :
```rust
for m in &mut self.concept_maturation {
    if *m > 0 { *m -= 1; }
}
```

## Implementation Steps

**type:** feat  
**risk:** P1  
**context:** domain  
**Context:** Protection anti-pruning des nouveau-nés dans prune_concepts(), puis boost du learning rate et réduction du seuil de nouveauté pendant la période critique dans step().

## Steps

1. Modifier `prune_concepts()` : dans la construction de `survivors`, ajouter `|| self.concept_maturation[i] > 0` pour les concepts en période critique → verify: `cd tso-engine && cargo test -- test_neurogenesis_critical_period 2>&1 | grep ok`

2. Ajouter la décrémentation des compteurs en fin de `sleep_cycle()` : boucle sur `&mut concept_maturation`, `*m > 0 { *m -= 1 }` → verify: `cd tso-engine && cargo test -- test_neurogenesis_maturation_decrements 2>&1 | grep ok`

3. Dans `step()`, après catégorisation : si `current_concept_id` est en maturation (`concept_maturation[cid] > 0`), appliquer `attractor.lr *= 3.0` et `novelty_threshold *= 0.5` pour ce step uniquement → verify: `cd tso-engine && cargo test -- test_neurogenesis_lr_boost 2>&1 | grep ok`

4. Restaurer attractor.lr et novelty_threshold après le step (sauvegarde/restauration autour du traitement) → verify: `cd tso-engine && cargo test -- test_neurogenesis_lr_boost 2>&1 | grep ok`

## Verification Script

```bash
cargo test -- test_neurogenesis_critical_period 2>&1
cargo test -- test_neurogenesis_maturation_decrements 2>&1
cargo test -- test_neurogenesis_lr_boost 2>&1
```

## Out of scope

- Modification de l'AttractorField API (lr est déjà un champ public)
- Changement du cycle online (step reste O(dim × n_classes) inchangé)

## Risks

- **Oubli de restauration** : si le boost de lr n'est pas restauré, le lr global reste à 3× → perturbation de tous les concepts. Mitigation : sauvegarde/restauration systématique.
- **Pruning trop agressif après maturation** : un concept peut être pruné immédiatement après sa période critique s'il n'a pas été activé. C'est le comportement désiré (remplacement).

## 4. Tests (Gherkin)

```gherkin
Scenario: Nouveau concept protégé du pruning
  Given un TsoEngine avec un concept en période critique (maturation > 0)
    And tous les autres concepts sont inactifs depuis > threshold steps
  When prune_concepts() est appelé
  Then le concept en période critique survit

Scenario: Période critique dure N cycles sommeil
  Given un TsoEngine avec sleep_maturation_cycles = 3
    And un nouveau concept créé (maturation = 3)
  When 3 cycles sommeil sont exécutés
  Then maturation du concept = 0

Scenario: Après maturation, le concept peut être pruné normalement
  Given un TsoEngine avec un concept mature (maturation = 0)
    And last_active_step > threshold
  When prune_concepts() est appelé
  Then le concept peut être élagué

Scenario: lr 3× pendant la période critique
  Given un TsoEngine avec un concept en maturation
  When step() traite une perception qui active ce concept
  Then le lr effectif = lr_normal × 3
```
