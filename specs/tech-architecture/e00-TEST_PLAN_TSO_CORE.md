# Test Design: TSO-CORE — Architecture de test à risque
## 17 scénarios structurés en 4 phases de risque

Basé sur les résultats d'ablation (paper.md §6) et la configuration par défaut
(CognitiveConfig::default() = AttractorField + Cerebellum + graphe Φ, tous les
autres sous-systèmes désactivés — cf. C13).

## 1. Risk Matrix & Scenarios

### Risk assessment
| Risque | Niveau | Justification |
|--------|--------|--------------|
| Régression cœur | P0 | AttractorField + Cerebellum = seul sous-ensemble validé (d=2.59). Toute régression casse le résultat principal |
| Φ stabilité | P1 | Φ peut diverger ou osciller (resolve_with_anneal) — pas de garantie de convergence hors des benchmarks |
| Variance seed RL | P1 | σ=0.20–0.29 sur MiniGrid : un seed peut masquer une régression |
| Homéostasie | P2 | hypothalamus désactivé par défaut, ne casse pas le cœur |
| Feature gates | P2 | attention/episodic/hypothalamus peuvent casser sans impacter le défaut |
| Scaling | P3 | Pas de benchmark >19×19 ; comportement à 10³ concepts inconnu |

### Phase 1 — Cœur validé (P0, hard gate de release)

| ID | Description | Niveau | Cible |
|----|-------------|--------|-------|
| SC-P0-01 | AttractorField classe 5 prototypes sur 147D aléatoire | Unit | attractor.rs — predict() retourne un concept_id valide, prototype_count ≥ 5 |
| SC-P0-02 | Cerebellum forward_logits + reinforce_td stabilise V(s) sur MDP à 1 état | Unit | cerebellum.rs — mock perception, reward cyclique, V(s) converge |
| SC-P0-03 | Cycle complet step() sans panic sur 100 pas, MiniGrid 7×7 | E2E | tso_engine.rs — step() avec config par défaut, pas de panic |
| SC-P0-04 | Φ gating ne dégrade pas le reward : gating ON ≥ gating OFF - 0.1σ | E2E | bench_phi_gating.rs — 5 seeds, threshold=0.5 |
| SC-P0-05 | AttractorField seul > Linear AC (d de Cohen > 1.0) sur Rotating-T | E2E | bench_ablation.rs — A1 vs A0, 10 seeds |
| SC-P0-06 | Tous les tests librairie passent | Integration | `cargo test --lib 2>&1 | grep "test result: ok"` |

### Phase 2 — Stabilité Φ (P1)

| ID | Description | Niveau | Cible |
|----|-------------|--------|-------|
| SC-P1-07 | Φ ne diverge pas sur 1000 steps avec graph_phi activé | E2E | core.rs — résolution longue, Φ(t) < borne (Φ < 10.0) |
| SC-P1-08 | Φ gating économie wall-clock : temps gating ≤ 0.5 × passif | E2E | bench_phi_gating_v2.rs — 3 seeds, threshold=0.5 |
| SC-P1-09 | add_transition crée ≤ 1 arête par step (pas de prolifération) | Unit | core.rs — add_transition size invariant |
| SC-P1-10 | resolve_with_anneal termine en O(max_iter × batch_count) | Unit | core.rs — mock graphe 10 nœuds, convergence en < max_iter |

### Phase 3 — Feature gates et non-cœur (P2)

| ID | Description | Niveau | Cible |
|----|-------------|--------|-------|
| SC-P2-11 | hypothalamus.step() ne panique pas sur 500 ticks | Unit | hypothalamus.rs — tick avec énergie/besoin par défaut |
| SC-P2-12 | Hypothalamus désactivé : gate_reward() = reward (pass-through) | Unit | hypothalamus.rs — gate_reward identité |
| SC-P2-13 | step() avec config par défaut = step() avec épisodique OFF | Integration | tso_engine.rs — toggle episodic_curiosity, reward invariant |
| SC-P2-14 | Feature gates compilent sans erreur | E2E | `cargo check --features hypothalamus,attention,episodic 2>&1 | grep -q Finished` |

### Phase 4 — Passage à l'échelle (P3, exploratoire)

| ID | Description | Niveau | Cible |
|----|-------------|--------|-------|
| SC-P3-15 | AttractorField avec 100 prototypes : classification < 2 ms | Unit | attractor.rs — bench temps de classif O(n) |
| SC-P3-16 | Graphe 10³ arêtes : resolve_with_anneal < 1s par appel | Integration | core.rs — bench résolution O(|E|) |
| SC-P3-17 | MiniGrid 19×19 : TSO ≥ Linear AC (non-régression) | E2E | bench_scale.rs |

### Niveaux de test
| Niveau | Scénarios | % |
|--------|-----------|---|
| Unit | SC-P0-01, P0-02, P1-09, P1-10, P2-11, P2-12, P3-15 | 7 | 41% |
| Integration | SC-P0-06, P2-13, P3-16 | 3 | 18% |
| E2E | SC-P0-03, P0-04, P0-05, P1-07, P1-08, P3-17, P2-14 | 7 | 41% |

## 2. Fixture Architecture

### Données de test
- **Perception 147D aléatoire** : `Array1::zeros(147)` + bruit gaussien ±0.1
- **MiniGrid DoorKey 7×7** : benchmark standard, 30 warm-up + 70 eval
- **MDP 1 état** : reward = +1 tous les 5 steps, sinon 0
- **Graphe mock** : 10 nœuds, 20 arêtes, Φ pré-calculé
- **Rotating-T** : benchmark non-stationnaire, 50 épisodes par phase

### Configuration standard
```rust
// CognitiveConfig::default() — attractor + cerebellum + graphe Φ, tout off
CognitiveConfig {
    attractor: true,
    graph_phi: true,
    phi_gating: false,
    attention: false,
    episodic_curiosity: false,
    metabolic_cost: false,
    hypothalamus: false,
    use_fpi: false,
    efe_weight: 0.0,
    delta_clip_max: 5.0,
    sleep_neurogenesis_rate: 0.0,
    sleep_maturation_cycles: 0,
    sleep_synaptic_scaling: false,
}
```

### Isolation
- Tests unitaires : pas de mock réseau, pas de E/S sauf stdio bench
- Benchmarks : `--release`, seeds fixées par argument CLI
- E2E : seeds [0..10], warm-up 30 épisodes avant mesure
- Φ gating : comparer gating ON vs OFF sur mêmes seeds

## 3. NFR Verification

| NFR Type | Requirement | Verification Command |
|----------|-------------|----------------------|
| Perf | step() < 1ms (config défaut) | `cargo bench -- step 2>&1` |
| Perf | Φ gating ≥ 50% wall-clock saving | `cargo run --release --bin bench_phi_gating_v2 -- 5 0.5` |
| Stability | Φ < 10.0 sur 1000 steps (graph_phi activé) | `cargo run --release --bin bench_phi_gating` |
| Reliability | Tous les tests passent | `cargo test 2>&1 | grep "test result: ok"` |
| Regression | A1 > A0 (d > 1.0) sur 10 seeds | `cargo run --release --bin bench_ablation -- 10` |

## 4. Out of Scope
- GPU / calcul parallèle (CPU-only)
- Réseau / MPI / distribué
- NLP / Dual-LIF
- Inférence active (FPI/EFE)
- VAE (retiré v0.2 — d=0.02 vs attracteur seul)
- Environnements inutilisés (Sokoban, Terrarium, ZigzagGrid)
