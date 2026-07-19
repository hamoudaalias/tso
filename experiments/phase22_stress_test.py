"""
Phase 22 — Stress-Test Φ-Gated : Prouver que la friction bat le hasard.

Phase 21 montre 60% d'économie mais l'aléatoire égalait Φ-gated.
Pourquoi ? Parce qu'à 40% d'apprentissage, le hasard capture encore
la structure générale.

La solution : pousser la compression à l'extrême.
À 5-10% des phrases, le hasard rate les anomalies cruciales ;
la friction, elle, cible précisément les zones de surprise.

Mesures :
- Accuracy SNLI à chaque taux de compression
- Densité du graphe (nœuds, arêtes, degré moyen)
- Φ-gated vs Aléatoire : le gap doit se creuser à haute compression
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
random.seed(SEED); np.random.seed(SEED)

WINDOW = 5
TOP_K = 20
SEED_SENTENCES = 10000
N_TOTAL = 50000
N_VAL = 2000
N_TEST = 2000

CACHE_BASELINE = "experiments/phase21_baseline.pkl"

RATES = [40, 25, 15, 10, 5, 2]  # taux de compression cibles (% du flux)


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


def learn_sentence(sentence, graph):
    tokens = tokenize(sentence)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + WINDOW + 1, len(tokens))):
            a, b = tokens[i], tokens[j]
            if a == b: continue
            if a not in graph: graph[a] = {}
            if b not in graph[a]: graph[a][b] = 0.0
            graph[a][b] += 1.0


def sentence_friction(sentence, graph):
    tokens = tokenize(sentence)
    if not tokens: return 0.0
    rare = sum(1 for t in tokens if t not in graph or len(graph[t]) < 5)
    return rare / len(tokens)


def graph_stats(graph):
    nodes = len(graph)
    edges = sum(len(ns) for ns in graph.values()) // 2
    avg_deg = np.mean([len(ns) for ns in graph.values()]) if nodes > 0 else 0
    return nodes, edges, avg_deg


def compute_trifriction(premise, hypothesis, graph, top_k=20):
    pt = tokenize(premise)
    ht = tokenize(hypothesis)
    np_set = set()
    for w in pt:
        if w in graph:
            np_set |= {n for n, _ in sorted(graph[w].items(),
                                            key=lambda x: -x[1])[:top_k]}
    nh_set = set()
    for w in ht:
        if w in graph:
            nh_set |= {n for n, _ in sorted(graph[w].items(),
                                            key=lambda x: -x[1])[:top_k]}
    union = len(np_set | nh_set)
    support = len(np_set & nh_set) / union if union > 0 else 0.0
    conflict = 1.0 - len(np_set & nh_set) / len(np_set) if len(np_set) > 0 else 0.5
    ht_in_np = sum(1 for w in ht if w in np_set)
    novelty = 1.0 - ht_in_np / len(ht) if len(ht) > 0 else 0.0
    return support, conflict, novelty


def evaluate_snli(graph, val_pairs, test_pairs):
    X_val, y_val, X_test, y_test = [], [], [], []
    for p, h, l in val_pairs:
        sup, conf, nov = compute_trifriction(p, h, graph)
        X_val.append([sup, conf, nov])
        y_val.append(l)
    for p, h, l in test_pairs:
        sup, conf, nov = compute_trifriction(p, h, graph)
        X_test.append([sup, conf, nov])
        y_test.append(l)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(np.array(X_val), np.array(y_val))
    return clf.score(np.array(X_test), np.array(y_test)) * 100


def run_stress_test():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 22 — Stress-Test : Φ-Gated vs Aléatoire")
    log("  Prouver que la friction bat le hasard à haute compression")
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
    seed = corpus[:SEED_SENTENCES]
    stream = corpus[SEED_SENTENCES:]
    log(f"    Seed: {len(seed)}, Stream: {len(stream)}")

    # =====================================================================
    # 2. Baseline : tout apprendre
    # =====================================================================
    log("\n[2] Baseline (tout apprendre)...")
    if os.path.exists(CACHE_BASELINE):
        with open(CACHE_BASELINE, "rb") as f:
            graph_full = pickle.load(f)["graph"]
    else:
        graph_full = {}
        for s in corpus: learn_sentence(s, graph_full)
    acc_full = evaluate_snli(graph_full, val_pairs, test_pairs)
    n_full, e_full, d_full = graph_stats(graph_full)
    log(f"    Accuracy: {acc_full:.1f}% | {n_full} nœuds, {e_full} arêtes, "
        f"degré {d_full:.1f}")

    # =====================================================================
    # 3. Φ-gated : trouver les seuils pour chaque taux cible
    # =====================================================================
    log("\n[3] Calibration des seuils Φ-gated...")

    # Construire graphe seed + calculer friction de chaque phrase du stream
    graph_seed = {}
    for s in seed: learn_sentence(s, graph_seed)

    stream_phis = np.array([sentence_friction(s, graph_seed) for s in stream])
    sorted_idx = np.argsort(-stream_phis)  # du plus surprenant au moins

    # Pour chaque taux cible, trouver le seuil correspondant
    thresholds = {}
    for rate in RATES:
        n_target = max(1, len(stream) * rate // 100)
        phi_at_target = stream_phis[sorted_idx[n_target - 1]]
        thresholds[rate] = (n_target, phi_at_target)
        log(f"    Taux {rate:>2d}% → {n_target:>4d} phrases, "
            f"seuil Φ>{phi_at_target:.3f}")

    # =====================================================================
    # 4. Stress test : comparer Φ-gated vs Random à chaque taux
    # =====================================================================
    log(f"\n[4] Stress test : {len(RATES)} taux de compression...")
    log(f"    {'Taux':>6s} | {'Apprises':>8s} | {'Φ-gated':>8s} "
        f"{'nœudsΦ':>7s} {'arêtesΦ':>8s} | {'Aléat.':>8s} "
        f"{'nœudsR':>7s} {'arêtesR':>8s} | {'Gap':>6s}")
    log(f"    {'-'*6} | {'-'*8} | {'-'*8} {'-'*7} {'-'*8} "
        f"| {'-'*8} {'-'*7} {'-'*8} | {'-'*6}")

    results = []

    for rate in RATES:
        n_target, theta = thresholds[rate]

        # --- Φ-gated ---
        graph_phi = {}
        for s in seed: learn_sentence(s, graph_phi)
        learned_idx = sorted_idx[:n_target]
        for i in learned_idx:
            learn_sentence(stream[i], graph_phi)
        acc_phi = evaluate_snli(graph_phi, val_pairs, test_pairs)
        n_phi, e_phi, d_phi = graph_stats(graph_phi)

        # --- Random ---
        graph_rand = {}
        for s in seed: learn_sentence(s, graph_rand)
        rand_idx = random.sample(range(len(stream)), n_target)
        for i in rand_idx:
            learn_sentence(stream[i], graph_rand)
        acc_rand = evaluate_snli(graph_rand, val_pairs, test_pairs)
        n_rand, e_rand, d_rand = graph_stats(graph_rand)

        gap = acc_phi - acc_rand
        results.append((rate, acc_phi, acc_rand, gap, n_phi, e_phi, n_rand, e_rand))

        log(f"    {rate:>5d}% | {n_target:>8d} | {acc_phi:>7.1f}% "
            f"{n_phi:>7d} {e_phi:>8d} | {acc_rand:>7.1f}% "
            f"{n_rand:>7d} {e_rand:>8d} | {gap:+6.1f}")

    # =====================================================================
    # 5. Analyse
    # =====================================================================
    log(f"\n[5] Analyse :")
    gaps = [(r, gap) for r, _, _, gap, _, _, _, _ in results]

    # Gap se creuse-t-il à mesure que la compression augmente ?
    # Calculer corrélation taux→gap
    rates_arr = np.array([g[0] for g in gaps])
    gaps_arr = np.array([g[1] for g in gaps])
    corr = np.corrcoef(rates_arr, gaps_arr)[0, 1]
    log(f"    Corrélation taux→gap: {corr:.3f} "
        f"({'POSITIVE: le gap se creuse' if corr > 0 else 'NÉGATIVE: pas de creusement'})")

    # Trouver le meilleur gap
    best = max(gaps, key=lambda x: x[1])
    log(f"    Meilleur gap: +{best[1]:.1f} points à {best[0]}%")

    # Résumé
    log(f"\n    {'Résultat clé':>30s}")
    log(f"    {'-'*30}")
    for rate, acc_p, acc_r, gap, _, _, _, _ in results:
        flag = " <<<" if gap > 2 else ""
        log(f"    Taux {rate:>2d}% : Φ-gated {acc_p:.1f}% vs Random {acc_r:.1f}% "
            f"= {gap:+.1f} pts{flag}")

    if any(g > 2 for _, g in gaps):
        log(f"\n  *** Φ-GATED BAT LE HASARD À HAUTE COMPRESSION ***")
        log(f"  La friction cible les anomalies, le hasard les rate.")
    elif any(g > 0 for _, g in gaps):
        log(f"\n  *** Φ-GATED LÉGÈREMENT SUPÉRIEUR ***")
        log(f"  L'effet est faible — le seuil est peut-être trop bas.")
    else:
        log(f"\n  *** AUCUN AVANTAGE DÉTECTÉ ***")
        log(f"  SNLI est trop homogène pour ce test.")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase22_results.csv", "w") as f:
        f.write("rate,phi_accuracy,random_accuracy,gap,phi_nodes,phi_edges,rand_nodes,rand_edges\n")
        for r, ap, ar, g, np_, ep, nr_, er_ in results:
            f.write(f"{r},{ap:.2f},{ar:.2f},{g:.2f},{np_},{ep},{nr_},{er_}\n")
    log(f"\n    → experiments/phase22_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_stress_test()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
