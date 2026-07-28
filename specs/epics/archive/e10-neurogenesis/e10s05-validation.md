# e10s05 — Validation : diversité, Φ borné, non-régression benchmarks

## 1. Business narrative

En tant qu'ingénieur TSO, je veux valider que la neurogenèse augmente la
diversité conceptuelle sans dégrader la stabilité cognitive (Φ), afin de
confirmer que le nouveau mécanisme ne casse rien.

## Implementation Steps

**type:** feat  
**risk:** P2  
**context:** domain  
**Context:** Après l'implémentation des stories e10s01-04, écrire les tests de validation qui prouvent que la neurogenèse augmente la diversité, ne fait pas exploser Φ, ne casse pas les benchmarks existants.

## Steps

1. Créer `tso-engine/tests/neurogenesis.rs` avec les 4 tests de validation → verify: `test -f tso-engine/tests/neurogenesis.rs`

2. Écrire `test_neurogenesis_diversity_increases` : 10 steps + sleep_cycle, vérifier `n_after > n_before` → verify: `cd tso-engine && cargo test -- test_neurogenesis_diversity 2>&1 | grep ok`

3. Écrire `test_neurogenesis_phi_bounded` : 20 cycles sommeil avec 50 steps chacun, vérifier `phi_after ≤ 0.5` → verify: `cd tso-engine && cargo test -- test_neurogenesis_phi_bounded 2>&1 | grep ok`

4. Écrire `test_neurogenesis_does_not_break_phi_convergence` : clone du test e05s01 avec `sleep_neurogenesis_rate = 0.1`, vérifier `phi_after ≤ phi_before + 0.05` → verify: `cd tso-engine && cargo test -- test_neurogenesis_phi_convergence 2>&1 | grep ok`

5. Écrire `test_neurogenesis_gridworld_stable` : benchmark GridWorld 5×5 avec neurogenèse, vérifier success_rate ≥ 0.9 → verify: `cd tso-engine && cargo test -- test_neurogenesis_gridworld 2>&1 | grep ok`

## Verification Script

```bash
cargo test -- test_neurogenesis_diversity 2>&1
cargo test -- test_neurogenesis_phi_bounded 2>&1
cargo test -- test_neurogenesis_phi_convergence 2>&1
cargo test -- test_neurogenesis_gridworld 2>&1
```

## Out of scope

- Benchmark de performance (temps d'exécution) — simple test fonctionnel
- Validation sur Minigrid ou Sokoban (hors scope TSO v2, ADR-002)

## Risks

- **Test flaky** : la neurogenèse est probabiliste (rate). Si le RNG ne coopère pas, `n_after > n_before` peut échouer. Mitigation : forcer `sleep_neurogenesis_rate = 1.0` dans les tests unitaires, rate modérée (0.1) seulement dans le test de non-régression.
- **Phi peut transitoirement augmenter** : de nouveaux concepts = nouvelles arêtes = nouvelles tensions potentielles. La tolérance +0.05 absorbe ce bruit.

## 2. Tests de validation

### 2.1 Test unitaire : diversité des classes

```rust
#[test]
fn test_neurogenesis_diversity_increases() {
    let mut engine = TsoEngine::new(6, 4);
    engine.sleep_neurogenesis_rate = 1.0;
    engine.sleep_max_concepts = 50;

    let n_before = engine.attractor.n_classes();
    for _ in 0..10 {
        let obs = engine.random_perception();
        engine.step(&obs, 0.0, None, &[]);
    }
    engine.sleep_cycle();
    let n_after = engine.attractor.n_classes();

    assert!(n_after > n_before,
        "La neurogenèse devrait augmenter le nombre de classes");
}
```

### 2.2 Test unitaire : Φ reste borné

```rust
#[test]
fn test_neurogenesis_phi_bounded() {
    let mut engine = TsoEngine::new(6, 4);
    engine.sleep_neurogenesis_rate = 0.3;
    engine.sleep_max_concepts = 30;

    let max_phi = 0.5;
    for cycle in 0..20 {
        for _ in 0..50 {
            let obs = engine.random_perception();
            engine.step(&obs, 0.0, None, &[]);
        }
        let report = engine.sleep_cycle();
        assert!(report.phi_after <= max_phi,
            "Cycle {}: Φ = {:.3} > seuil {}", cycle, report.phi_after, max_phi);
    }
}
```

### 2.3 Test : non-régression des sleep_phi_convergence

```rust
#[test]
fn test_neurogenesis_does_not_break_phi_convergence() {
    // Même test que test_sleep_phi_convergence mais avec neurogenèse activée
    // (déjà présent dans e05s01)
    let mut engine = /* setup standard */;
    engine.sleep_neurogenesis_rate = 0.1; // faible, réaliste

    let phi_before = engine.graph.phi();
    engine.sleep_cycle();
    let phi_after = engine.graph.phi();

    // La neurogenèse n'empêche pas la convergence : Φ doit baisser ou stagner
    assert!(phi_after <= phi_before + 0.05,
        "Φ ne devrait pas augmenter de plus de 0.05 après neurogenèse");
}
```

### 2.4 Benchmark : GridWorld 5×5

Vérifier que le taux de succès sur GridWorld 5×5 reste stable (100%
avec shaping BFS). La neurogenèse ne doit pas dégrader la navigation.

```rust
#[test]
fn test_neurogenesis_gridworld_stable() {
    // Même setup que bench_gridworld mais avec neurogenèse activée
    // Le taux de succès doit rester ≥ 90% (tolére une légère perturbation)
    let success_rate = /* run benchmark */;
    assert!(success_rate >= 0.9, "La neurogenèse ne doit pas casser GridWorld");
}
```

## 3. Métriques de succès

| Métrique | Cible | To-lérance |
|----------|-------|------------|
| Nombre de classes après 20 cycles sommeil | ≥ n_initial × 1.5 | Peut saturer à max_concepts |
| Φ après sommeil | ≤ 0.1 | +0.05 en transitoire |
| Succès GridWorld 5×5 | ≥ 90% | -5% toléré |
| Temps d'exécution sleep_cycle | ≤ 2× baseline | La neurogenèse est O(n×k) |
