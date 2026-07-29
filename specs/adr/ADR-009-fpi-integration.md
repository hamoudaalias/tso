# ADR-009 : Intégration FPI dans le cycle cognitif

## Statut
Accepté — e11/efe-integration, merge sur master.

## Contexte
Le cycle cognitif TSO utilise resolve_with_anneal (recuit simulé sur
Invert/Align/Repel) pour minimiser la tension cognitive Φ. FPI (Fixed
Point Iteration) est une approche alternative : itération de point fixe
sur le simplexe qui minimise la VFE (Variational Free Energy).

## Décision
- FPI est disponible comme alternative à resolve_with_anneal
- Activé via CognitiveConfig.use_fpi = true (default=false)
- Pas de remplacement : les deux mécanismes coexistent
- FPI utilise des softmax sur log-vraisemblance au lieu du recuit géométrique

## Architecture
1. step() avec use_fpi=true ignore l'attracteur et le graphe
2. run_factorized_fpi calcule la croyance postérieure q(s)
3. Le concept_id est l'argmax de q(s) (winner-take-all compatible TSO)
4. EFE scoring optionnel via efe_weight dans les logits du cerebellum

## Conséquences
- Positif : FPI donne une distribution q(s) complète (vs concept discret)
- Positif : EFE prospectif peut remplacer le TD rétrospectif
- Négatif : FPI ne produit pas de Φ — la tension cognitive disparaît
- Risque : le passage use_fpi=true désactive l'attracteur + graphe
  (les deux sont incompatibles par construction)
