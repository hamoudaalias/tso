# Automatisation — Matrice d'ablations du bien-être

## Binaire
`tso-engine/src/bin/ablation_matrix.rs`

## Sortie
CSV sur stdout : `terme,Neutre,Faim,Anxiété,Surprise,Métabolique`

Chaque cellule = taux de succès ε=0 (moyenne sur 5 seeds, terme ablaté à 0).

## Commande
```bash
cd tso-engine && cargo run --bin ablation_matrix > specs/experiments/ablation_matrix_$(date +%Y%m%d).csv
```

## Pipeline d'automatisation

### Phase 1 — Génération (11 min)
```bash
cargo run --release --bin ablation_matrix > ablation.csv
```

### Phase 2 — Visualisation (gnuplot)
```bash
gnuplot -e "
set term png size 1200,800; set output 'ablation_heatmap.png';
set title 'Ablation matrix: 9 terms × 5 regimes';
set xlabel 'Regime'; set ylabel 'Term (ablated)';
plot 'ablation.csv' using 2:xtic(1) matrix with image;
"
```

### Phase 3 — Rapport automatique
```bash
# Lire le CSV et identifier le terme dominant par régime
python3 -c "
import csv, sys
r = csv.DictReader(open('ablation.csv'))
for row in r:
    regime_cols = [k for k in row if k != 'terme']
    best = max(regime_cols, key=lambda c: float(row[c]))
    print(f'{row[\"terme\"]} → {best}: {row[best]}%')
"
```

## CI
Ajouter une workflow `cron` hebdomadaire dans `.github/workflows/ablation.yml` :
```yaml
on:
  schedule:
    - cron: '0 6 * * 1'  # chaque lundi 6h UTC
jobs:
  ablation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd tso-engine && cargo run --release --bin ablation_matrix
      - uses: actions/upload-artifact@v4
        with:
          name: ablation-matrix
          path: ablation.csv
```

## Résultats récents (2026-07-31)

| Terme ablaté | Neutre | Faim | Anxiété | Surprise | Métabolique |
|-------------|--------|------|---------|----------|-------------|
| gated_reward | 38% | 33% | **70%** | 63% | 49% |
| consummatory | 58% | 61% | **75%** | 57% | 73% |
| curiosity | 49% | **78%** | 53% | 39% | 48% |
| shaping | **69%** | 63% | 31% | **71%** | 53% |
| phi_delta | 50% | 33% | 57% | 56% | 63% |
| chronic_tension | 41% | 51% | **72%** | 32% | 57% |
| deficit_penalty | **67%** | 50% | 61% | 43% | 44% |
| metabolic_penalty | **71%** | 55% | 53% | 46% | 39% |
| parsimony | 51% | 56% | 51% | 31% | **71%** |

### Lecture
- **Anxiété (Φ élevé)** : `gated_reward`, `consummatory`, `chronic_tension` dominent.
  Quand le graphe est tendu, la modulation de récompense externe est critique.
- **Faim** : `curiosity` (78%) domine largement. Un agent affamé explore plus.
- **Métabolique** : `consummatory` (73%), `parsimony` (71%). Beaucoup de concepts → pression ontologique.
- **Neutre** : `metabolic_penalty` (71%), `shaping` (69%). Le coût cognitif et le BFS shaping sont les guides principaux.
- **Surprise** : `shaping` (71%), `gated_reward` (63%). Environnement nouveau → le shaping BFS et la récompense externe reprennent le dessus.
