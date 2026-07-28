# Deepen Architecture — Pipeline TSO

**Candidats identifiés** par exploration de code + churn heuristique.

## 1. `tso_engine.rs::step()` — 300+ lignes mono-méthode

**Fichier :** `tso_engine.rs` (1553 lignes, 43 fonctions)
**Score :** 2 (shallow — interface complexe, pas de sous-modules)

**Problème :** `step()` gère attractor, encoder, episodic, hypothalamus, attention,
cerebellum, reward shaping, Φ, curiosity — tout dans une seule fonction.
Comprendre un bug nécessite de lire les 300 lignes. Les 10+ flags de
`CognitiveConfig` sont testés en séquence dans la même fonction.

**Solution :** extraire chaque sous-système en phase nommée :
```rust
// step() → 5 phases :
let gated = self.phase_attention(perception);
let concept = self.phase_categorization(&gated);
let reward = self.phase_reward(concept, external_reward);
let action = self.phase_decision(concept, reward, bfs_value);
self.phase_episodic(concept);
return action;
```

**Bénéfice :** chaque phase testable isolément, step() devient un orchestreur
de 5 appels. **Locality** : bug dans le reward shaping → `phase_reward()`.

## 2. `CognitiveConfig` — struct plate qui voyage dans 8 bins

**Fichier :** `tso_engine.rs` (14 champs, 4 flags)
**Score :** 2

**Problème :** 6 binaires reconstruisent `CognitiveConfig { ..default() }`.
14 champs plats, pas de sous-groupes. Changement d'un champ → 6 fichiers modifiés.

**Solution :** sous-groupes :
```rust
pub struct CognitiveConfig {
    pub subsystems: SubsystemFlags, // attractor, graph_phi, attention, episodic, metabolic, hypothalamus
    pub neurogenesis: NeurogenesisConfig,
    pub delta_clip_max: f64,
}
```

**Bénéfice :** `..default()` marche sans mise à jour. Sous-groupes sérialisables.
**Impact :** rupture de compatibilité binaire → ADR.

## 3. `SleepReport` / `MetricsSnapshot` / `NeurogenesisOutcome` — 3 structs qui se chevauchent

**Fichier :** `tso_engine.rs`, `neurogenesis.rs`
**Score :** 1 (très shallow — duplication)

**Problème :** `SleepReport` a `replay_count`, `prototypes_pruned`, etc.
`MetricsSnapshot` a `phi`, `well_being`, `energy`, etc.
`NeurogenesisOutcome` a `births`, `deaths`.
Rien n'est partagé. Ajouter une métrique → 3 structs à modifier + 3 sérialisations.

**Solution :** `struct Metrics { phi, births, deaths, pruned, ... }` (tout).
`SleepReport` et `NeurogenesisOutcome` deviennent des vues ou alias.

**Bénéfice :** une source de vérité pour toutes les métriques. Ajout d'un champ = 1 modification.

## 4. `prune_concepts()` — réindexation manuelle de 10 vecteurs

**Fichier :** `tso_engine.rs:966-1091`
**Score :** 2

**Problème :** 125 lignes de réindexation manuelle avec `old_to_new` pour
10 vecteurs parallèles (prototypes, graph nodes, edges, tracking vectors,
episodic memory, transition log). Chaque nouveau vecteur (`concept_maturation`)
nécessite 4 modifications : (1) while resize, (2) survivor filter, (3) old_to_new remap.

**Solution :** encapsuler les vecteurs dans un `ConceptTable` qui gère
l'ajout et le retrait :
```rust
pub struct ConceptTable<T> {
    data: Vec<T>,
    active: Vec<bool>,
}
impl ConceptTable<T> {
    fn add(&mut self, val: T);
    fn remove_inactive(&mut self, threshold: usize) -> Vec<(usize, T)>; // removed items + new id mapping
}
```

**Bénéfice :** l'ajout d'un nouveau vecteur ne nécessite plus de toucher
`prune_concepts()`. **Locality** : la logique de réindexation est en un seul endroit.

## 5. `constraint_redirection.rs` — module à un caller

**Fichier :** `constraint_redirection.rs` (467 lignes)
**Score :** 3

**Problème :** 2 fonctions publiques, appelées uniquement depuis `tso_engine.rs`.
Le module est profond mais sous-utilisé : `sync_prototypes_from_graph` n'est
jamais appelé dans le code actuel.

**Solution :** fusionner dans `core.rs` ou garder avec un warning de code mort.

**Bénéfice :** 1 fichier de moins, 0 perte de fonctionnalité.

---

## Priorisation

| # | Candidat | Score | Effort | Impact | Conseil |
|---|----------|-------|--------|--------|---------|
| 1 | `step()` phases | 2 | 2h | Fort | Faire |
| 2 | CognitiveConfig sous-groupes | 2 | 30min | Moyen | Faire |
| 3 | Structs métriques dupliquées | 1 | 15min | Faible | Faire |
| 4 | ConceptTable | 2 | 1h | Fort | Réfléchir |
| 5 | constraint_redirection fusion | 3 | 10min | Faible | Faire |

**Lequel veux-tu explorer ?** (ou tous si on lance `dispatch-agents`)
