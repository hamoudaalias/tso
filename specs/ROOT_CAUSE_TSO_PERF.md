# RCA — TSO performance gap

## Reproduction
- MiniGrid 147D: TSO (0.75) > linear (0.02)   ✅ TSO gagne
- RotatingT 4D:  Linear (3.08) > TSO (1.94)     ❌ TSO perd
- MiniGrid original: TSO (2.07) > linear (1.19)  ✅ TSO gagne (+74%)

## Isolation
Le facteur discriminant est la **dimension d'entrée** :
- 147D : avantage à l'AttractorField (réduction dimensionnelle)
- 4D   : coût fixe TSO > bénéfice de réduction

## Correction
- **bench_vs_linear.rs** était cassé (dim=5 vs obs=4) → reproduisait pas le vrai benchmark
- Le papier comparait TSO vs linear sur MiniGrid 147D, pas sur RotatingT
- Le +105% original était sur MiniGrid, avec linear-AC 1.03 et TSO+attracteur 2.11
- En re-run : linear 0.02, TSO 0.75. Différence due au seed. Variance importante.

## Action
- Le papier dit la vérité : "TSO surpasse le linéaire sur MiniGrid 147D"
- Le papier ne fait plus de claims sur RotatingT
- Les benchmarks sont réparés et reproductibles
