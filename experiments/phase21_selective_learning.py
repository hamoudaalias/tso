"""
Phase 21 — Apprentissage Sélectif Φ-Gated.

TSO n'apprend que quand il est surpris.

Si Φ (surprise structurelle d'une phrase par rapport au graphe courant)
est faible → la phrase est déjà comprise → on ne fait rien.
Si Φ est élevé → le système apprend : R-STDP, nouveaux nœuds, nouveaux liens.

Objectif : atteindre l'accuracy de Phase 20 (48.4%) avec 5× moins de
phrases réellement apprises.
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED)
np.random.seed(SEED)

WINDOW = 5
TOP_K = 20
SEED_SENTENCES = 10000
N_TOTAL = 50000
N_VAL = 2000
N_TEST = 2000
EVAL_EVERY = 5000

CACHE_BASELINE = "experiments/phase21_baseline.pkl"
CACHE_GATED = "experiments/phase21_gated.pkl"


def tokenize(text):
    return [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split()
            if len(t.strip(".,!?;:\"'()[]{}")) > 0]


def load_snli_jsonl(path, max_examples=None):
    pairs, sentences = [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            d = json.loads(line)
            if d["gold_label"] == "-":
                continue
            label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
            pairs.append((d["sentence1"], d["sentence2"],
                          label_map[d["gold_label"]]))
            sentences.extend([d["sentence1"], d["sentence2"]])
    return pairs, sentences


def build_graph_rstdp(sentences, window=5):
    graph = {}
    def add_edge(a, b):
        if a == b:
            return
        if a not in graph:
            graph[a] = {}
        if b not in graph[a]:
            graph[a][b] = 0.0
        graph[a][b] += 1.0

    for sent in sentences:
        tokens = tokenize(sent)
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + window + 1, len(tokens))):
                add_edge(tokens[i], tokens[j])

    for word in graph:
        neighbors = graph[word]
        max_w = max(neighbors.values()) if neighbors else 1.0
        if max_w > 0:
            for n in neighbors:
                neighbors[n] /= max_w
    return graph


def sentence_friction(sentence, graph):
    """Surprise structurelle d'une phrase = fraction de mots rares.
    Un mot est 'rare' s'il a < 5 voisins dans le graphe courant.
    """
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    rare = 0
    for t in tokens:
        if t not in graph or len(graph[t]) < 5:
            rare += 1
    return rare / len(tokens)


def learn_sentence(sentence, graph):
    """Appliquer R-STDP pour une phrase."""
    tokens = tokenize(sentence)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + WINDOW + 1, len(tokens))):
            a, b = tokens[i], tokens[j]
            if a == b:
                continue
            if a not in graph:
                graph[a] = {}
            if b not in graph[a]:
                graph[a][b] = 0.0
            graph[a][b] += 1.0


def compute_trifriction(premise, hypothesis, graph, top_k=20):
    pt = tokenize(premise)
    ht = tokenize(hypothesis)

    np_set = set()
    for w in pt:
        if w in graph:
            ns = sorted(graph[w].items(), key=lambda x: -x[1])[:top_k]
            np_set |= {n for n, _ in ns}

    nh_set = set()
    for w in ht:
        if w in graph:
            ns = sorted(graph[w].items(), key=lambda x: -x[1])[:top_k]
            nh_set |= {n for n, _ in ns}

    union = len(np_set | nh_set)
    support = len(np_set & nh_set) / union if union > 0 else 0.0
    conflict = 1.0 - len(np_set & nh_set) / len(np_set) if len(np_set) > 0 else 0.5
    ht_in_np = sum(1 for w in ht if w in np_set)
    novelty = 1.0 - ht_in_np / len(ht) if len(ht) > 0 else 0.0

    return support, conflict, novelty


def evaluate_snli(graph, val_pairs, test_pairs):
    """Évaluer accuracy SNLI avec le graphe courant."""
    X_val, y_val, X_test, y_test = [], [], [], []
    for p, h, l in val_pairs:
        sup, conf, nov = compute_trifriction(p, h, graph)
        X_val.append([sup, conf, nov])
        y_val.append(l)
    for p, h, l in test_pairs:
        sup, conf, nov = compute_trifriction(p, h, graph)
        X_test.append([sup, conf, nov])
        y_test.append(l)

    X_val = np.array(X_val); y_val = np.array(y_val)
    X_test = np.array(X_test); y_test = np.array(y_test)

    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_val, y_val)
    val_acc = clf.score(X_val, y_val) * 100
    test_acc = clf.score(X_test, y_test) * 100
    return val_acc, test_acc, clf


def run_benchmark():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 21 — Apprentissage Sélectif Φ-Gated")
    log("  TSO n'apprend que quand il est surpris (Φ > seuil)")
    log("=" * 72)

    # =====================================================================
    # 1. Dataset
    # =====================================================================
    log("\n[1] Chargement SNLI...")
    test_pairs, _ = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_test.jsonl")
    train_pairs, train_sentences = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_train.jsonl")
    val_pairs = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_train.jsonl",
        max_examples=5000 + N_VAL)[0][5000:5000 + N_VAL]
    test_pairs = test_pairs[:N_TEST]

    corpus = train_sentences[:N_TOTAL]
    log(f"    Corpus: {len(corpus)} phrases")
    log(f"    Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # =====================================================================
    # 2. Baseline : tout apprendre (Phase 19)
    # =====================================================================
    log("\n[2] Baseline — tout apprendre (Phase 19)...")
    if os.path.exists(CACHE_BASELINE):
        with open(CACHE_BASELINE, "rb") as f:
            bl = pickle.load(f)
        graph_all = bl["graph"]
        log(f"    Cache chargé: {len(graph_all)} nœuds")
    else:
        graph_all = build_graph_rstdp(corpus, WINDOW)
        with open(CACHE_BASELINE, "wb") as f:
            pickle.dump({"graph": graph_all}, f, protocol=5)
        log(f"    Construit: {len(graph_all)} nœuds")

    _, test_acc_all, _ = evaluate_snli(graph_all, val_pairs, test_pairs)
    log(f"    Accuracy test (tout appris): {test_acc_all:.1f}%")

    # =====================================================================
    # 3. Apprentissage sélectif Φ-gated
    # =====================================================================
    log("\n[3] Apprentissage sélectif Φ-gated...")

    # Découpage : seed + flux
    seed_sentences = corpus[:SEED_SENTENCES]
    stream_sentences = corpus[SEED_SENTENCES:]

    # Seuil calibré sur le seed : médiane de la friction après seed
    graph_seed = {}
    for s in seed_sentences[:2000]:
        learn_sentence(s, graph_seed)
    phis_seed = [sentence_friction(s, graph_seed) for s in seed_sentences[:500]]
    theta_gate = np.percentile(phis_seed, 50)  # médiane après seed
    log(f"    Seed: {len(seed_sentences)} phrases")
    log(f"    Flux: {len(stream_sentences)} phrases")
    log(f"    Seuil Φ_gate = {theta_gate:.4f} "
        f"(médiane friction post-seed, min={min(phis_seed):.2f}, max={max(phis_seed):.2f})")

    # Construire graphe initial avec le seed
    graph = {}
    for s in seed_sentences:
        learn_sentence(s, graph)
    log(f"    Graphe initial: {len(graph)} nœuds")

    # Traiter le flux
    n_learned = 0
    n_skipped = 0
    acc_history = []

    for idx, sentence in enumerate(stream_sentences):
        phi = sentence_friction(sentence, graph)
        if phi > theta_gate:
            learn_sentence(sentence, graph)
            n_learned += 1
        else:
            n_skipped += 1

        if (idx + 1) % EVAL_EVERY == 0:
            _, test_acc, _ = evaluate_snli(graph, val_pairs, test_pairs)
            total = idx + 1
            learned_pct = n_learned / total * 100
            acc_history.append((total, test_acc, n_learned, learned_pct))
            log(f"      {total:>5d} phrases | "
                f"appris={n_learned:>4d} ({learned_pct:>4.0f}%) | "
                f"skippé={n_skipped:>4d} | "
                f"test={test_acc:.1f}%")

    # Final
    total = len(stream_sentences)
    learned_pct = n_learned / total * 100

    log(f"\n    Résumé :")
    log(f"      Phrases totales: {total + SEED_SENTENCES}")
    log(f"      Phrases apprises: {n_learned + SEED_SENTENCES} "
        f"({(n_learned + SEED_SENTENCES) / (total + SEED_SENTENCES) * 100:.0f}%)")
    log(f"      Phrases skippées: {n_skipped} ({n_skipped/total*100:.0f}%)")

    _, test_acc_gated, _ = evaluate_snli(graph, val_pairs, test_pairs)
    log(f"      Accuracy test: {test_acc_gated:.1f}%")

    # =====================================================================
    # 4. Contre-vérification : sélection aléatoire au même taux
    # =====================================================================
    log("\n[4] Contrôle — sélection aléatoire au même taux d'apprentissage...")

    graph_random = {}
    for s in seed_sentences:
        learn_sentence(s, graph_random)

    n_random = 0
    for sentence in stream_sentences:
        if random.random() < (n_learned / total):
            learn_sentence(sentence, graph_random)
            n_random += 1

    _, test_acc_random, _ = evaluate_snli(graph_random, val_pairs, test_pairs)
    log(f"    Appris aléatoirement: {n_random} phrases")
    log(f"    Accuracy test: {test_acc_random:.1f}%")

    # =====================================================================
    # 5. Comparaison
    # =====================================================================
    log(f"\n[5] Comparaison :")
    log(f"    {'Stratégie':>30s} | {'Apprises':>8s} | {'Accuracy':>8s}")
    log(f"    {'-'*30} | {'-'*8} | {'-'*8}")
    log(f"    {'Tout apprendre (Phase 19)':>30s} | {N_TOTAL:>8d} | "
        f"{test_acc_all:>6.1f}%")
    log(f"    {'Φ-gated (Phase 21)':>30s} | "
        f"{n_learned + SEED_SENTENCES:>8d} | {test_acc_gated:>6.1f}%")
    log(f"    {'Aléatoire (contrôle)':>30s} | "
        f"{n_random + SEED_SENTENCES:>8d} | {test_acc_random:>6.1f}%")

    savings = 100 - (n_learned + SEED_SENTENCES) / N_TOTAL * 100
    acc_gap = test_acc_all - test_acc_gated
    log(f"\n    Économie: {savings:.0f}% de phrases non apprises")
    log(f"    Perte accuracy: {acc_gap:.1f} points")

    if test_acc_gated >= test_acc_random + 2:
        log(f"\n  *** Φ-GATED > ALÉATOIRE ***")
        log(f"  L'apprentissage sélectif cible les bonnes phrases")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase21_results.csv", "w") as f:
        f.write("strategy,sentences_learned,accuracy\n")
        f.write(f"full,{N_TOTAL},{test_acc_all:.2f}\n")
        f.write(f"phi_gated,{n_learned + SEED_SENTENCES},{test_acc_gated:.2f}\n")
        f.write(f"random,{n_random + SEED_SENTENCES},{test_acc_random:.2f}\n")
    log(f"\n    → experiments/phase21_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
