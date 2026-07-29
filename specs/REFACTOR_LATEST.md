# REFACTOR — Papier TSO

## Stratégie : "Fixer d'abord le périmètre, puis réduire les claims aux preuves"

### Commit 1 — Retirer le cadre CDT comme "théorie"
- Supprimer §3 titré "Théorie de la Dissipation Cognitive"
- Renommer en "§3. Friction topographique (Φ)"
- Garder les définitions mathématiques (minimales), supprimer le narratif "fondement théorique"
- Déplacer la dynamique Dual-LIF sous Architecture (§4)

### Commit 2 — Supprimer les comparaisons non-chiffrées
- §2 "Travaux connexes" : remplacer le texte positionnel par max 1 phrase. "TSO emprunte à ACT, SNN, Active Inference mais s'en distingue par la friction comme déclencheur."
- Supprimer toute phrase de type "TSO bat/surpasse/se distingue"

### Commit 3 — Réduire les contributions listées
- Dans le Résumé, supprimer la liste à 4 contributions
- Remplacer par 2 max : "(i) Architecture à friction topographique, (ii) Benchmark MiniGrid"
- VAE online + Gumbel-STE passe en sous-section technique dans Architecture

### Commit 4 — Nettoyer les claims non-supportés
- Partout remplacer "montre que", "démontre", "prouve" par "suggère", "indique", "permet"
- Quantifier précisément chaque claim avec le nombre de benchmarks qui le soutiennent

### Commit 5 — Redimensionner les travaux futurs
- 2 lignes max : "Extension à Procgen/Habitat en cours. Code sur GitHub."
- Supprimer les 6 axes détaillés

### Commit 6 — Nettoyer les références
- Chaque référence citée dans le corps doit être utilisée
- Ne garder que celles qui sont citées au moins une fois

### Commit 7 — Renforcer le benchmark
- Section Expériences : ajouter explicitement les limites (10 seeds seulement, un seul scénario, pas d'ensemble de validation)
- Ne pas cacher que c'est le seul benchmark
