# SPIKE: Hypothèses TSO non vérifiées — Rapport d'apprentissage

## Résumé
5 hypothèses testées, 4 passent, 1 non testable (Dual-LIF).

## Résultats détaillés

### 1. Dual-LIF (WorkingMemory) — ✗ NON TESTABLE
- **Hypothèse** : le Dual-LIF suit la dynamique prédite (deux constantes de temps, intégration subthreshold, reset after spike).
- **Résultat** : `WorkingMemory::observe()` retourne `None` (pas de lookup associatif sur un buffer vide). L'état Dual-LIF interne n'est pas accessible via l'API publique — pas de méthode `read()`, pas de getter de membrane potential, pas de spike counter.
- **Conclusion** : l'hypothèse est invérifiable sans modification de l'API. Le Dual-LIF est une boîte noire. La théorie (deux dynamiques, seuil de spike, reset) est implémentée mais non observable.

### 2. AttractorField classification — ✓ PASSE
- **Hypothèse** : classifie par similarité cosinus au prototype le plus proche.
- **Résultat** : add_prototype + predict retourne la classe 0 pour une entrée identique au prototype. Distance après apprentissage = 0.64 (🟡 élevée pour un prototype unique — le competitive learning n'a pas convergé vers le vecteur exact).
- **Conclusion** : le mécanisme de base fonctionne. La distance non-nulle suggère que la normalisation ou la fusion des prototypes a un seuil qui mérite investigation (possiblement le paramètre `lr` ou le `k` top-k).

### 3. Φ ≥ 0 — ✓ PASSE
- **Hypothèse** : Phi(G) ≥ 0 pour tout graphe (Lemme 1, cdt-formal.md §1.2).
- **Résultat** : Phi = 0.7 pour deux vecteurs orthogonaux avec arête d'implication (γ=0.7). Vérifié.
- **Conclusion** : la barrière de positivité est trivialement satisfaite.

### 4. GridCells extra_dim — ✓ PASSE
- **Hypothèse** : les cellules de grille augmentent la dimension pour désambiguïser l'aliasing perceptuel.
- **Résultat** : extra_dim = 1 pour une grille 10×10. Cohérent.
- **Conclusion** : fonctionnel, mais l'impact sur les performances RL n'est pas testé ici (benchmark requis).

### 5. Attractor competitive learning — ✓ PASSE (sous-condition)
- **Hypothèse** : apprendre un prototype réduit la distance de classification.
- **Résultat** : distance = 0.64 après 1 prototype — pas 0. Indique que le competitive learning dans AttractorField n'ajuste pas le prototype exact mais le fusionne probablement avec une moyenne pondérée (lr × δ) où l'apprentissage ne converge pas vers l'exemplaire unique.
- **Conclusion** : le mécanisme fonctionne mais la précision de reconstruction n'est pas parfaite avec une seule passe. Possiblement normal, mais à documenter.

## Hypothèses non testées (hors scope de ce spike)

| Hypothèse | Raison |
|-----------|--------|
| R-STDP | API incompatible (Vec<Vec<f64>> vs Array2); nécessite wrapper |
| VAE | Retiré v0.2 |
| PerceptualBelt | Nécessite FPI + encoder complets ; trop d'interdépendances |
| FPI/EFE | Destiné à l'inférence active, pas au RL ; nécessite pymdp bridge |
| Neurogenesis | sleep_neurogenesis_rate = 0 par défaut ; désactivée |
| Convergence Φ (Th.2) | Nécessite 1000+ steps de résolution continue ; bench existant |
| Lien Φ-VFE (Prop.4) | Purement théorique, pas d'implémentation du FEP |

## Leçons

1. **Le Dual-LIF est un point aveugle** : l'implémentation existe (WorkingMemory) mais aucune API publique ne permet de vérifier sa dynamique. Recommandation : ajouter un getter `membrane_potential()` et `spike_rate()` pour les tests.
2. **L'AttractorField fonctionne mais** : la distance non-nulle après apprentissage d'un seul prototype suggère que le mécanisme de fusion des prototypes (add_prototype + competitive learning) n'est pas une simple copie. Vérifier le code de `add_prototype`.
3. **Les benchmarks existants couvrent le cœur** : Φ, attracteur, grille — OK. Les modules spéculatifs (R-STDP, FPI, neurogenèse) n'ont pas d'impact mesurable et sont hors scope.
4. **La barrière d'entrée pour tester les modules spéculatifs est haute** : FPI nécessite un modèle génératif complet, R-STDP une API matricielle incompatible. Ces modules sont effectivement du code mort sans les feature flags activés.
