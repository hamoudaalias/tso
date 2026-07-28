# ADR 007 : Neurogenèse structurelle dans le cycle sommeil

**Statut :** Accepté
**Date :** 2025-10-10
**Source :** elaborate-spec suite à BUG-001 + ADR-002

## Contexte

Le cycle sommeil de TSO (`sleep_cycle()`) comporte 5 phases :
1. Replay bruité (attractor) avec neurogenèse minimale (prototypes dans classe existante)
2. Résolution profonde du graphe (recuit simulé)
3. Pruning des prototypes redondants (AttractorField.prune_redundant)
4. Suppression des arêtes à faible Φ (Graph.remove_low_phi_edges)
5. Pruning des concepts inactifs (prune_concepts)

BUG-001 et ADR-002 ont montré que le cervelet TD(λ) ne peut pas apprendre
de tâches complexes (>4 actions, reward sparse). La recommandation est de
pivoter vers les forces de TSO, dont la **neurogenèse** fait partie.

Or la neurogenèse actuelle est symbolique : `add_prototype(&noisy, cid)`
ajoute un prototype à une classe **existante**. Aucune nouvelle classe
n'est créée pendant le sommeil. Le système ne peut pas émerger de
nouvelles catégories conceptuelles.

## Décision

1. **Nouvelle Phase 1.5 dans `sleep_cycle()` — Neurogenèse** : création
   de nouvelles classes dans l'AttractorField à partir de prototypes
   existants mutés (bruit), avec connexion initiale au graphe sémantique.
2. **Période critique** : compteur de maturation (paramétrable) pendant
   lequel le nouveau concept a un `lr` 3× plus élevé et un seuil de
   nouveauté réduit de moitié.
3. **Protection anti-pruning** : les concepts en période critique ne sont
   jamais élagués par `prune_concepts()`.
4. **Homéostasie structurelle** : budget de concepts maximum ; au-delà,
   le concept le moins actif est remplacé (pas de mort neuronale, juste
   un remplacement).
5. **Scaling synaptique** : après chaque cycle de neurogenèse, normaliser
   les poids d'arêtes incidents pour éviter l'emballement des connexions.

## Justification

- **Bio-inspiration** : la neurogenèse adulte du gyrus denté (DG)
  hippocampal est le modèle canonique. Les nouveaux neurones granulaires
  ont une période critique de ~4 semaines (haute excitabilité,
  plasticité LTP renforcée). Ils reçoivent des inputs de l'EC et envoient
  des outputs vers CA3.
- **Simplicité technique** : l'AttractorField supporte déjà
  `add_class(prototype)`. Le graphe supporte `add_node()` et `add_edge()`.
  La période critique est un compteur sur le concept.
- **Pas de perturbation RL** : la neurogenèse agit sur la catégorisation
  et le graphe, pas sur le cervelet. Elle ne peut pas dégrader ce qui ne
  marche pas déjà (ADR-002).

## Conséquences

- **Positives** :
  - Le système peut émerger de nouvelles catégories conceptuelles
  - La diversité des représentations augmente
  - Les concepts obsolètes sont remplacés naturellement
  - Paramétrage simple (un taux, un budget max)
- **Négatives** :
  - Coût O(nouveaux_concepts × connexions) pendant le sommeil
  - Risque d'explosion du nombre de concepts si le budget max est trop haut
  - Les IDs de concepts changent → les séquences épisodiques avec des IDs
    de concepts remplacés sont orphelines (traitées par remap dans
    `prune_concepts()` existant)
- **Risques** :
  - Si `sleep_neurogenesis_rate` est trop haut, Φ peut augmenter
    (nouveaux concepts = nouvelles arêtes = nouvelles tensions potentielles)

## Prior Art

| Approche | Similitudes | Différences |
|----------|-------------|-------------|
| **NEAT** (Stanley & Miikkulainen, 2002) | Ajout de nœuds par mutation, protection de l'innovation | Speciation + crossover ; pas de période critique, pas de remplacement homéostatique |
| **Cascade Correlation** (Fahlman & Lebiere, 1990) | Ajout séquentiel de neurones cachés, gel des anciens poids | Optimisation mathématique (corrélation avec l'erreur) vs bio-inspiration (bruit + maturation) |
| **FEAGI / neuropool / neuromod** | Rust, bio-inspiré, plasticité structurelle | Architecture SNN, pas de graphe conceptuel, pas de Φ |
| **Sleep-like replay** (Nature Comms 2022) | Rejeu non-supervisé protège les mémoires | Ne fait pas de neurogenèse, juste consolidation |
| **Neurogenesis incrémentale** (arXiv 2018) | Dynamique > statique à budget égal | Supervisé, pas de période critique |

Aucune solution prête à l'emploi ne combine **attractor field + graphe conceptuel + neurogenèse avec période critique + remplacement homéostatique** dans un moteur bio-inspiré comme TSO. Verdict : **build**.

## Implémentation

### Modifications dans `tso_engine.rs`

```rust
/// Configuration additionnelle dans CognitiveConfig
pub struct CognitiveConfig {
    // ... champs existants ...
    pub sleep_neurogenesis_rate: f64,  // défaut 0.2
    pub sleep_max_concepts: usize,     // défaut 50
    pub sleep_maturation_cycles: usize, // défaut 3
}
```

### Nouvelle structure `ConceptMetadata`

```rust
pub struct ConceptMetadata {
    pub maturation_counter: usize,
    pub birth_cycle: usize,
}
```

### Nouvelle Phase 1.5 dans `sleep_cycle()`

```rust
// ── Phase 1.5: Neurogenèse ──
// Parcourt les prototypes existants et crée de nouvelles classes
// à partir de versions bruitées, puis les connecte au graphe.
for i in 0..self.attractor.n_classes() {
    if !rng.gen_bool(self.sleep_neurogenesis_rate) { continue; }
    let proto = self.attractor.get_prototype(i);
    // Muter le prototype pour créer un nouveau concept
    let noise: Array1<f64> = ...;
    let new_proto = proto + &noise;
    let new_id = self.attractor.add_class(&new_proto);
    // Ajouter un nœud au graphe sémantique
    self.graph.add_node(new_id);
    // Connecter aléatoirement à 2-3 voisins existants
    for _ in 0..rng.gen_range(2..=3) {
        let neighbor = rng.gen_range(0..self.attractor.n_classes());
        self.graph.add_edge(new_id, neighbor, 0.1);
    }
    // Initialiser les métadonnées de période critique
    self.concept_maturation[new_id] = self.sleep_maturation_cycles;
}
```

### Modification de `prune_concepts()`

Les concepts avec `maturation_counter > 0` sont protégés du pruning.

### Scaling synaptique

Dans `remove_low_phi_edges()` existant, ajouter une normalisation :
si le poids total incident d'un nœud dépasse 2× la moyenne, scaling linéaire.

## Design d'interface retenu

Après exploration de 3 designs (voir `specs/tech-architecture/neurogenesis-interface-design.md`) :

**Design 1 (Minimal) pour l'interface publique** :
- `NeurogenesisConfig` : 4 champs (rate, max_concepts, maturation_cycles, synaptic_scaling)
- `Neurogenesis::new(config)` + `cycle(&mut self, attractor, graph, last_active, noise_std) -> Outcome`
- 1 méthode publique, deep module (~120 lignes cachées)

**Design 2 (Phases) pour les tests** (`#[cfg(test)]`) :
- `birth_phase()`, `homeostasis()`, `scale_synapses()`, `end_cycle()`
- Testable isolément sans exposer l'ordre des phases

**Design 3 (Règles déclaratives)** — rejeté : prématuré, YAGNI, moteur de règles pour 4 cas.

Le module scaffold est dans `specs/neurogenesis/mod.rs`. La migration du code inline
de `sleep_cycle()` vers ce module est une étape ultérieure.
