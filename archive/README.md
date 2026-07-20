# Archive — Références Python (TSO Phase Recherche)

Ces fichiers sont conservés pour référence historique et architecturale.
Le kernel Rust (`tso-kernel/`) remplace tout le code Python de production.

## Contenu

| Chemin | Description |
|--------|-------------|
| `tso_nlp/` | Interface NLP Python (embedder MiniLM, SOM, décodeur Inverse Motor) |
| `tso_mini_lm/` | Auto-encodeur topologique zéro-gradient (idée pour embedding Rust) |
| `experiments/phase13_shakespeare.py` | Génération séquentielle Shakespeare — référence pour boucle auto-régressive |
| `src/phase4_benchmark.py` | MiniGPT de référence (PyTorch) pour comparaison |
| `src/phase12_tokenizer.py` | Intégration tokenizer BPE |
| `src/real_embedder.py` | Interface d'embedding texte → vecteur |
| `blueprint_minilm.md` | Blueprint architecturale de l'auto-encodeur TSO-MiniLM |
