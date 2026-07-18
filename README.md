# TSO : Topographic Stabilization Operator (V3.1 — Kernel)

Une architecture neuromorphique événementielle basée sur la dissipation de friction cognitive.

TSO propose un changement de paradigme : le calcul n'est pas déclenché par le flux de données, mais par la nécessité de maintenir l'homéostasie interne. Le langage y est modélisé comme une action motrice visant à réduire une tension interne ($\Phi$) générée par des contradictions logiques. L'apprentissage séquentiel y est obtenu sans rétropropagation dans le temps (BPTT).

## Structure du Projet

```
tso/
├── tso_kernel/         # Noyau mathématique pur (NumPy uniquement)
│   ├── neurons.py      # LIF clusters, dynamics
│   ├── plasticity.py   # R-STDP, eligibility traces
│   ├── friction.py     # Phi computation
│   ├── operators.py    # Double Mapping, Inverse Motor
│   └── core.py         # TSOCore orchestrator
├── tso_nlp/            # Interface langage (PyTorch, HF)
│   ├── embedder.py     # MiniLM wrapper
│   ├── som.py          # Self-Organizing Map
│   └── decoder.py      # Transition graph, Inverse Motor
├── experiments/        # Scripts de validation
│   ├── phase0_geometry.py
│   └── phase13_shakespeare.py
├── tests/              # Tests unitaires
│   └── test_friction.py
├── src/                # Scripts legacy des phases 0-14
├── paper.md
└── README.md
```

## Résultats Principaux

- **Sevrage du NLI :** Détection endogène des relations entre concepts (implication/contradiction) par dynamique électrique, sans DeBERTa externe.
- **Moteur Inverse Sémantique :** Sélection de mots parmi 1000+ par projection (50×384), sans matrice dense.
- **Génération Conceptuelle :** Prédiction du cluster conceptuel attendu (Phase 13), pas du mot exact — les alternatives syntaxiques ne s'annulent plus.
- **Corpus Shakespeare :** Φ chute à **−1.78** (280% mieux que hasard, 62.5% d'amélioration), apprentissage Hebbien local.
- **Le Kernel est certifié :** Lemme 1 vérifié (3/3 tests), Φ computation testée (7 tests unitaires).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Tests unitaires
python tests/test_friction.py

# Kernel seulement (NumPy, sans GPU)
python -c "from tso_kernel.core import TSOCore; print('Kernel OK')"

# Validation géométrique
python experiments/phase0_geometry.py

# Shakespeare conceptuel (GPU recommandé)
python experiments/phase13_shakespeare.py
```

## Auteur

**Hamouda ALIAS** — Institut de Neuro-Cybernétique
