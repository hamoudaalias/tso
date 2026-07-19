"""
Phase 20 — Φ = (support, conflit, nouveauté) : friction à 3 composantes.

Le problème de Phase 19 : Jaccard est monotone, le Neutre est coincé
entre Entailment et Contradiction. La solution : décomposer Φ en :

  support   = Jaccard(N(P), N(H))         — contexte partagé → Entailment
  conflit   = voisins forts de P absents   — attentes violées → Contradiction
             de N(H)
  nouveauté = mots de H hors N(P)          — info nouvelle → Neutral

Une régression logistique sur [support, conflit, nouveauté] peut
apprendre le profil unique de chaque classe :
  - Entailment :    support↑, conflit↓, nouveauté↓
  - Neutral :       support→, conflit↓, nouveauté↑
  - Contradiction : support↓, conflit↑, nouveauté→
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CACHE = "experiments/topological_graph.pkl"
N_VAL = 2000
N_TEST = 2000
TOP_K = 20


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


def compute_trifriction(premise, hypothesis, graph, top_k=20):
    """Calcule les 3 composantes de friction.

    Retourne (support, conflit, nouveauté) où chaque composante ∈ [0,1].
    """
    pt = tokenize(premise)
    ht = tokenize(hypothesis)

    # N(P) : voisinage de la prémisse
    np_set = set()
    for w in pt:
        if w in graph:
            neighbors = sorted(graph[w].items(), key=lambda x: -x[1])[:top_k]
            np_set |= {n for n, _ in neighbors}

    # N(H) : voisinage de l'hypothèse
    nh_set = set()
    for w in ht:
        if w in graph:
            neighbors = sorted(graph[w].items(), key=lambda x: -x[1])[:top_k]
            nh_set |= {n for n, _ in neighbors}

    # Support : Jaccard entre les deux voisinages
    union = len(np_set | nh_set)
    support = len(np_set & nh_set) / union if union > 0 else 0.0

    # Conflit : fraction des voisins forts de P absents de N(H)
    # (attentes de la prémisse que l'hypothèse ne satisfait pas)
    if len(np_set) > 0:
        conflict = 1.0 - len(np_set & nh_set) / len(np_set)
    else:
        conflict = 0.5

    # Nouveauté : fraction des tokens de H hors de N(P)
    ht_in_np = sum(1 for w in ht if w in np_set)
    novelty = 1.0 - ht_in_np / len(ht) if len(ht) > 0 else 0.0

    return support, conflict, novelty


def run_benchmark():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 20 — Tri-Friction : Φ = (support, conflit, nouveauté)")
    log("  Rupture avec la friction monotone → Neutre libéré")
    log("=" * 72)

    # =====================================================================
    # 1. Graphe topologique (Phase 19)
    # =====================================================================
    log("\n[1] Chargement du graphe topologique R-STDP...")
    if not os.path.exists(CACHE):
        log("    ERREUR : lancer Phase 19 d'abord pour construire le graphe")
        return
    with open(CACHE, "rb") as f:
        data = pickle.load(f)
    graph = data["graph"]
    log(f"    {len(graph)} nœuds")

    # =====================================================================
    # 2. Dataset
    # =====================================================================
    log("\n[2] Chargement SNLI...")
    test_pairs, _ = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_test.jsonl")
    _, train_sentences = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_train.jsonl")
    val_pairs = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_train.jsonl",
        max_examples=5000 + N_VAL)[0][5000:5000 + N_VAL]
    test_pairs = test_pairs[:N_TEST]
    log(f"    Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # =====================================================================
    # 3. Calcul des 3 composantes de friction
    # =====================================================================
    log("\n[3] Calcul de (support, conflit, nouveauté)...")

    def extract(pairs):
        X, y = [], []
        for p, h, l in pairs:
            sup, conf, nov = compute_trifriction(p, h, graph, TOP_K)
            X.append([sup, conf, nov])
            y.append(l)
        return np.array(X), np.array(y)

    X_val, y_val = extract(val_pairs)
    X_test, y_test = extract(test_pairs)
    log(f"    Val: {X_val.shape}, Test: {X_test.shape}")

    # Profil par classe
    log(f"\n    Profil (support, conflit, nouveauté) par classe :")
    log(f"    {'Classe':>15s} {'support':>8s} {'conflit':>8s} {'nouveauté':>9s}")
    for l, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        m = X_val[y_val == l].mean(axis=0)
        log(f"    {name:>15s} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>9.4f}")

    # =====================================================================
    # 4. Fusion
    # =====================================================================
    log("\n[4] Fusion (LogisticRegression 3 features)...")
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_val, y_val)  # on calibre sur la val

    val_acc = clf.score(X_val, y_val) * 100
    test_acc = clf.score(X_test, y_test) * 100
    log(f"    Validation: {val_acc:.1f}%")
    log(f"    Test: {test_acc:.1f}%")

    log(f"\n    Poids de la fusion :")
    names = ["support", "conflit", "nouveauté"]
    for j, name in enumerate(["Entailment", "Neutral", "Contradiction"]):
        parts = [f"{names[i]}={clf.coef_[j][i]:+.4f}" for i in range(3)]
        log(f"      {name}: {'  '.join(parts)}")

    # =====================================================================
    # 5. Détail
    # =====================================================================
    y_pred = clf.predict(X_test)

    log(f"\n[5] Matrice de confusion (test) :")
    cm = confusion_matrix(y_test, y_pred)
    log(f"    {'':>15s} {'Entail':>8s} {'Neutral':>8s} {'Contra':>8s}")
    for i, name in enumerate(["Entailment", "Neutral", "Contradiction"]):
        log(f"    {name:>15s} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    for name, label in [("Entailment", 0), ("Neutral", 1), ("Contradiction", 2)]:
        mask = y_test == label
        acc = (y_pred[mask] == label).mean() * 100
        log(f"      {name}: {acc:.1f}% ({mask.sum()} ex.)")

    # =====================================================================
    # 6. Comparaison
    # =====================================================================
    log(f"\n[6] Comparaison :")
    log(f"    {'Modèle':>35s} | {'Accuracy':>8s} | {'Embedding':>12s}")
    log(f"    {'-'*35} | {'-'*8} | {'-'*12}")

    baselines = [
        ("Random", 33.3, "—"),
        ("TSO v1 vectoriel (Phase 18)", 44.0, "MiniLM"),
        ("TSO v2 topologique (Phase 19)", 41.5, "aucun"),
        ("TSO v2 tri-friction (Phase 20)", test_acc, "aucun"),
        ("BERT-base", 80.4, "WordPiece"),
    ]
    for name, acc, emb in baselines:
        log(f"    {name:>35s} | {acc:>6.1f}% | {emb:>12s}")

    delta_v1 = test_acc - 44.0
    delta_v2 = test_acc - 41.5
    log(f"\n    Delta vs TSO v1 (vectoriel): {delta_v1:+.1f} points")
    log(f"    Delta vs TSO v2 (Phase 19): {delta_v2:+.1f} points")

    if test_acc > 41.5:
        log(f"\n  *** TRI-FRICTION > TOPOLOGIE MONOTONE ***")
        log(f"  La décomposition de Φ libère le raisonnement")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase20_results.csv", "w") as f:
        f.write("model,accuracy,embedding\n")
        f.write(f"TSO_Trifriction,{test_acc:.2f},none\n")
        f.write(f"TSO_Topological_v2,{41.5:.2f},none\n")
        f.write(f"TSO_Vector_v1,{44.0:.2f},MiniLM\n")
        f.write(f"BERT,{80.4:.2f},WordPiece\n")
        f.write(f"Random,{33.3:.2f},none\n")
    log(f"\n    → experiments/phase20_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
