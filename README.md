# TSO : Topographic Stabilization Operator (V3.1 — PoC final)

Une architecture neuromorphique événementielle basée sur la dissipation de friction cognitive.

TSO propose un changement de paradigme : le calcul n'est pas déclenché par le flux de données, mais par la nécessité de maintenir l'homéostasie interne. Le langage y est modélisé comme une action motrice visant à réduire une tension interne ($\Phi$) générée par des contradictions logiques. L'apprentissage séquentiel y est obtenu sans rétropropagation dans le temps (BPTT).

## Résultats Principaux (V3.1 — PoC final)

- **Sevrage du NLI :** Le SNN détecte lui-même les relations entre concepts (implication/contradiction) par dynamique électrique, sans DeBERTa externe.
- **Moteur Inverse Sémantique :** Sélection de mots parmi 1000+ par projection (50×384), sans matrice dense vocabulaire.
- **Génération Auto-Régressive :** Apprentissage de séquences syntaxiques (ex: "CHAT EST ANIMAL MAIS") par traces d'éligibilité multi-échelles, sans BPTT.
- **Efficacité :** 28x moins de FLOPs en multi-tâches (25M vs 708M) vs Transformer.
- **Mémoire :** 0% d'oubli catastrophique (plasticité locale pure).
- **Zéro-shot :** 100% de succès sur la résolution de paradoxes non vus à l'entraînement.

## Résultats Clés Phase 9-11

**Phase 9 — EMA linéaire :**
- **τ=20** : amnésie totale (0.6%, niveau du hasard)
- **τ=200** : 26.6% (26× le hasard) — la trace lente porte le crédit temporel
- Plafond à ~30% dû au bruit linéaire additif

**Phase 10 — Réservoir ESN non-linéaire :**
- Séquences courtes (SeqLen=5) : **45.3%** (+48% vs EMA) — la non-linéarité améliore la discrimination
- Séquences longues (SeqLen=20) : **4.1%** — les dynamiques récurrentes interfèrent avec la mémoire pure
- Le LIF binaire échoue totalement (hasard) — la binarisation détruit l'information discriminante

**Phase 11 — Skip Calcul Dynamique :**
- Coût SNN variable **2,9× plus élevé** pour une séquence paradoxale ($\Phi$=170) vs triviale ($\Phi$=0)
- Le calcul TSO suit la friction — les tokens triviaux consomment le minimum, les paradoxes déclenchent le Double Mapping
- Un Transformer déploie 100% des FLOPs sur 100% des tokens

**Phase 12 — Tokenizer BPE (GPT-2, 50 257 tokens) :**
- Projection SNN(50) → embedding(384) avec cosinus **0.9996** en 40 époques (règle d'Oja)
- Token cible dans le top-5 systématiquement parmi 50 257
- **19 200 paramètres** seulement (0.1% d'une couche softmax complète)

**Phase 13 — Prédiction Conceptuelle Shakespeare :**
- Chaque mot quantifié sur SOM 10×10 (100 concepts) via embeddings MiniLM
- Graphe de transitions conceptuelles appris par Hebbien local
- Friction Φ initiale : −1.32 (233% mieux que hasard), finale : −2.12 (314% mieux que hasard)
- **Amélioration de 60%** de la confiance pendant la lecture sans rétropropagation
- TSO ne prédit pas le mot exact, mais le CONCEPT attendu — les alternatives syntaxiques ("be"/"go"/"have" après "to") ne s'annulent plus

## Pipeline de Validation (13 Phases)

0. **Phase 0 :** Preuve géométrique du Double Mapping ($\Phi=0$).
1. **Phase 1 :** Consolidation des implications par R-STDP locale.
2. **Phase 2 :** Expansion topologique SNN (recrutement de neurones dormants).
3. **Phase 3 :** Pipeline complet NLP réel (MiniLM + SOM + Hebbien natif sur GPU).
4. **Phase 4 :** Benchmark vs Transformer (28x FLOPs, zero-shot 100%, 0% oubli).
5. **Phase 5 :** Décodeur local (Apprentissage de l'émission du mot "MAIS").
6. **Phase 6 :** Génération auto-régressive (Crédit temporel sans BPTT).
7. **Phase 7 :** Moteur Inverse (Scaling à 1000 mots par projection sémantique).
8. **Phase 8 :** Sevrage NLI (Critic natif par co-activation Hebbienne, plus de Transformer externe).
9. **Phase 9 :** Crédit Temporel Long-Distance (Copie sans BPTT, τ=200 → 26× hasard).
10. **Phase 10 :** Réservoir Non-Linéaire (ESN vs EMA, +48% sur séquences courtes).
11. **Phase 11 :** Skip Calcul Dynamique (FLOPs événementiels, Φ→2.9× le coût SNN).
12. **Phase 12 :** Tokenizer BPE Réel (GPT-2, 50k tokens, 0.1% des paramètres d'un softmax).
13. **Phase 13 :** Corpus Shakespeare — prédiction conceptuelle via graphe de transitions SOM. Φ chute à 314% mieux que hasard.

## Installation

```bash
pip install torch numpy sentence-transformers
```

## Utilisation

```bash
python src/phase0_simulation.py
python src/phase2_tso_expansion.py
python src/phase3_pipeline.py          # Pipeline NLP (GPU recommandé)
python src/phase4_benchmark.py         # 28x FLOPs
python src/phase5_decoder.py
python src/phase6_autoregressive.py
python src/phase7_scaling_decoder.py   # 1000 mots
python src/phase13_corpus.py           # Prédiction Conceptuelle Shakespeare
```

## Auteur

**Hamouda ALIAS** - Institut de Neuro-Cybernétique
