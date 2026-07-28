# Design d'Interface — Module Neurogenesis

**Source :** `design-interface` post-e10 (3 designs parallèles)
**Date :** 2025-10-10
**Statut :** Recommandation acceptée

## Problème

La neurogenèse est actuellement inline dans `sleep_cycle()` (~120 lignes) avec
4 flags dans `CognitiveConfig`. Pas de module dédié, pas de frontière claire.
Interface ad-hoc : la phase 1.5 connaît les tracking vectors de TsoEngine,
le graphe, l'attractor — tout est couplé.

## Callers

- `sleep_cycle()` — l'appelant principal, une fois par cycle sommeil
- Tests unitaires (13 existants)
- Benchmarks (optionnel)
- Inspection/debug (optionnel)

## 3 Designs explorés

### Design 1 : Minimal (recommandé interface publique)

```rust
pub struct NeurogenesisConfig {
    pub rate: f64,
    pub max_concepts: usize,
    pub maturation_cycles: usize,
    pub synaptic_scaling: bool,
}

pub struct NeurogenesisOutcome {
    pub births: usize,
    pub deaths: usize,
    pub phi_after: f64,
}

pub struct Neurogenesis {
    config: NeurogenesisConfig,
    maturation: Vec<usize>,
}

impl Neurogenesis {
    pub fn new(config: NeurogenesisConfig) -> Self;
    pub fn cycle(
        &mut self,
        attractor: &mut AttractorField,
        graph: &mut Graph,
        last_active_step: &[usize],
        noise_std: f64,
    ) -> NeurogenesisOutcome;
}
```

**Depth :** 120 lignes de logique derrière 1 appel → deep module ✅
**Borrow checker :** ⚠️ 5 références mutuelles → risque si le module grossit

### Design 2 : Phases explicites (recommandé tests + debug)

```rust
impl Neurogenesis {
    pub fn birth_phase(...) -> usize;
    pub fn protect_newborns(&self, survivors: &mut [bool]) -> Vec<usize>;
    pub fn make_room(...) -> Option<usize>;
    pub fn scale_synapses(&self, graph: &mut Graph);
    pub fn end_cycle(&mut self);
    pub fn maturation_snapshot(&self) -> &[usize];
}
```

**Depth :** 6 méthodes pour ~130 lignes → shallow module ❌
**Testabilité :** chaque phase testable isolément ✅

### Design 3 : Règles déclaratives (rejeté — prématuré)

```rust
pub enum NeuroRule {
    Birth { priority: u8, rate: f64 },
    CriticalProtection { priority: u8, maturation_cycles: usize },
    Homeostasis { priority: u8, strategy: ReplacementStrategy },
    SynapticScaling { priority: u8 },
}
```

**Extensibilité :** ✅ mais YAGNI — une seule politique utile aujourd'hui
**Complexité :** moteur de règles pour 4 cas ❌

## Recommandation

**Design 1 pour l'interface publique**, **Design 2 en privé** :

```rust
impl Neurogenesis {
    // Public — 1 méthode, tout caché
    pub fn cycle(&mut self, ...) -> Outcome;

    // Test only — phases explicites accessibles
    #[cfg(test)]
    pub fn birth_phase(&mut self, ...) -> usize;
    #[cfg(test)]
    pub fn scale_synapses(&self, ...);
}
```

Justification :
1. Le caller principal (`sleep_cycle()`) ne veut qu'une ligne — pas 6
2. Les tests ont besoin de phases isolées — pas besoin que ce soit public
3. Si un jour on a 3+ politiques, on passera au Design 3 — pas avant
4. Le borrow checker est gérable tant qu'on passe `&mut TsoEngine` plutôt que 5 refs séparées
