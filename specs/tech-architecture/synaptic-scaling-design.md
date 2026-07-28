# Scaling Synaptique Homéostatique — Phase 3.5

**Date :** 2025-10-10
**Source :** e10s04, implémenté dans `tso_engine.rs::sleep_cycle()` Phase 3.5

## Algorithme

```
Phase 3.5 (après prune_redundant, avant remove_low_phi_edges)

Pour chaque nœud du graphe :
  1. Calculer le poids total incident (sum of all edge weights touching this node)
  2. Si total > 2 × moyenne des totaux de tous les nœuds :
     - scale = (2 × moyenne) / total
     - Multiplier tous les poids incidents par scale
     - Clamper le résultat dans [0, 127]

Préserve le contraste relatif entre les arêtes d'un même nœud.
```

## Bio-inspiration

Dans le cerveau, le **scaling synaptique homéostatique** normalise la force
totale des synapses d'un neurone pendant le sommeil (Turrigiano, 2008).
Les neurones trop actifs voient leurs synapses réduites proportionnellement,
préservant le contraste relatif entre entrées fortes et faibles.

## Implémentation

```rust
let n_nodes = self.graph.nodes.len();
if n_nodes > 0 {
    let mut incident_weight = vec![0i64; n_nodes];
    for e in &self.graph.edges {
        incident_weight[e.from] += e.weight as i64;
        incident_weight[e.to] += e.weight as i64;
    }
    let mean_weight = incident_weight.iter().sum::<i64>() as f64 / n_nodes as f64;
    let threshold = mean_weight * 2.0;
    if threshold > 0.0 {
        for (i, &total) in incident_weight.iter().enumerate() {
            let total_f = total as f64;
            if total_f > threshold {
                let scale = threshold / total_f;
                for e in &mut self.graph.edges {
                    if e.from == i || e.to == i {
                        let scaled = (e.weight as f64 * scale).round() as i8;
                        e.weight = scaled.clamp(0, 127);
                    }
                }
            }
        }
    }
}
```

## Paramètres

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `sleep_synaptic_scaling` | `true` | Activer/désactiver la phase |
| Seuil de déclenchement | 2× moyenne | Au-dessus, le nœud est "trop connecté" |
| Clamp | [0, 127] | i8 signé mais les poids sont positifs |

## Tests

- `test_neurogenesis_synaptic_scaling` — scaling réduit les poids excessifs
- `test_neurogenesis_synaptic_contrast` — le ratio 4:1 est préservé
- `test_neurogenesis_scaling_disabled` — `false` = pas de modification
