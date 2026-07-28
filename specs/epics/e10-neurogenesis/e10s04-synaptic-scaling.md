# e10s04 — Scaling synaptique : normalisation des arêtes après neurogenèse

## 1. Business narrative

En tant que graphe sémantique, je veux normaliser les poids des arêtes
après chaque cycle de neurogenèse, afin d'éviter que les connexions des
nouveaux concepts ne créent des tensions artificielles ou ne déséquilibrent
la structure cognitive.

## 2. Prérequis

- e10s01 (Phase 1.5 avec connexion aléatoire à poids 0.1)
- Graph avec `edges: Vec<Edge>` et `remove_low_phi_edges()`

## 3. Scaling synaptique

### 3.1 Bio-inspiration

Dans le cerveau biologique, le **scaling synaptique homéostatique**
normalise les forces synaptiques globalement pendant le sommeil :
si un neurone est trop excité (poids totaux élevés), tous ses poids
sont réduits proportionnellement. L'effet net préserve le contraste
relatif entre les entrées.

### 3.2 Implémentation

Nouvelle Phase 3.5 (après le pruning des prototypes, avant le pruning
des arêtes à faible Φ) :

```rust
// ── Phase 3.5: Scaling synaptique homéostatique ──
// Normalise les poids incidents de chaque nœud pour qu'aucun
// noeud n'ait un poids total > 2× la moyenne.
let n_nodes = self.graph.nodes.len();
if n_nodes > 0 && self.sleep_synaptic_scaling {
    // Calculer le poids total par nœud
    let mut incident_weight = vec![0.0f64; n_nodes];
    for e in &self.graph.edges {
        incident_weight[e.from] += e.weight;
        incident_weight[e.to] += e.weight;
    }
    let mean_weight = incident_weight.iter().sum::<f64>() / n_nodes as f64;
    let threshold = mean_weight * 2.0;

    for (i, &total) in incident_weight.iter().enumerate() {
        if total > threshold && total > 0.0 {
            let scale = threshold / total;
            for e in &mut self.graph.edges {
                if e.from == i || e.to == i {
                    e.weight *= scale;
                }
            }
        }
    }
}
```

### 3.3 Activation conditionnelle

Le scaling est contrôlé par `sleep_synaptic_scaling: bool`
(défaut `true`) dans `CognitiveConfig`.

## Implementation Steps

**type:** feat  
**risk:** P2  
**context:** domain  
**Context:** Après la neurogenèse, normaliser les poids des arêtes pour éviter qu'un nœud ne concentre trop de connexions. Phase 3.5 dans sleep_cycle(), contrôlée par sleep_synaptic_scaling: bool.

## Steps

1. Ajouter `sleep_synaptic_scaling: bool` (défaut true) à CognitiveConfig → verify: `cd tso-engine && cargo check --lib`

2. Implémenter la Phase 3.5 dans sleep_cycle() : calculer le poids incident total par nœud, si > 2× la moyenne, multiplier tous les poids incidents de ce nœud par `threshold / total` → verify: `cd tso-engine && cargo test -- test_neurogenesis_synaptic_scaling 2>&1 | grep ok`

3. Tester la préservation du contraste relatif : si un nœud a deux arêtes de poids 0.8 et 0.2, après scaling le ratio doit rester 4:1 → verify: `cd tso-engine && cargo test -- test_neurogenesis_synaptic_contrast 2>&1 | grep ok`

4. Tester que sleep_synaptic_scaling = false skip la phase (pas de modification des poids) → verify: `cd tso-engine && cargo test -- test_neurogenesis_scaling_disabled 2>&1 | grep ok`

## Verification Script

```bash
cargo test -- test_neurogenesis_synaptic_scaling 2>&1
cargo test -- test_neurogenesis_synaptic_contrast 2>&1
cargo test -- test_neurogenesis_scaling_disabled 2>&1
```

## Out of scope

- Normalisation globale (tous les poids, pas seulement les excessifs)
- Scaling dépendant de l'activité (bio) — juste un scaling statique post-neurogenèse

## Risks

- **Aucun** : le scaling est conditionnel (bool), ne touche que les nœuds au-dessus du seuil, et l'effet est réversible (multiplication scalaire).

## 4. Tests (Gherkin)

```gherkin
Scenario: Scaling réduit les poids excessifs
  Given un graphe avec un nœud ayant poids incident = 5.0
    And la moyenne des poids incidents = 1.0
    And scaling_synaptic_enabled = true
  When la Phase 3.5 s'exécute
  Then le nœud a un poids incident ≤ 2.0

Scenario: Scaling préserve le contraste relatif
  Given un graphe avec arêtes A(0→1, 0.8) et B(0→2, 0.2)
    And le nœud 0 dépasse le seuil
  When le scaling s'exécute
  Then A.weight / B.weight = 4.0 (contraste préservé)

Scenario: Pas de scaling si désactivé
  Given sleep_synaptic_scaling = false
  When la Phase 3.5 est conditionnellement sautée
  Then aucun poids d'arête n'est modifié
```
