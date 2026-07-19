"""
Phase 25 — MultiNLI : TSO v3.1 sur matched + mismatched.
"""
import sys, os, time, pickle, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from datasets import load_dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CACHE_GRAPH = "experiments/multinli_graph.pkl"
CACHE_FEATURES = "experiments/multinli_features.pkl"
WINDOW = 5
TOP_K = 20
N_TRAIN = 100000
N_VAL = 5000


def tokenize(text):
    out = []
    for t in text.lower().split():
        t = t.strip(".,!?;:\"'()[]{}")
        if t:
            out.append(t)
    return out


def learn_sentence(sentence, graph):
    tokens = tokenize(sentence)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + WINDOW + 1, len(tokens))):
            a, b = tokens[i], tokens[j]
            if a == b: continue
            if a not in graph: graph[a] = {}
            graph[a][b] = graph[a].get(b, 0.0) + 1.0


def prepare_sorted_neighbors(graph):
    sn = {}
    for w in graph:
        sn[w] = {n for n, _ in sorted(graph[w].items(), key=lambda x: -x[1])[:TOP_K]}
    return sn


def compute_trifriction(premise, hypothesis, sorted_neighbors):
    pt = tokenize(premise)
    ht = tokenize(hypothesis)
    np_set = set()
    for w in pt:
        if w in sorted_neighbors:
            np_set |= sorted_neighbors[w]
    nh_set = set()
    for w in ht:
        if w in sorted_neighbors:
            nh_set |= sorted_neighbors[w]
    union = len(np_set | nh_set)
    support = len(np_set & nh_set) / union if union > 0 else 0.0
    conflict = 1.0 - len(np_set & nh_set) / len(np_set) if len(np_set) > 0 else 0.5
    ht_in_np = sum(1 for w in ht if w in np_set)
    novelty = 1.0 - ht_in_np / len(ht) if len(ht) > 0 else 0.0
    return np.array([support, conflict, novelty])


def extract_from_split(split_name, sorted_neighbors, n_max=None):
    ds = load_dataset('nyu-mll/multi_nli', split=split_name, streaming=True)
    X, y, genres = [], [], []
    for i, ex in enumerate(ds):
        if n_max and i >= n_max:
            break
        if ex['label'] not in [0, 1, 2]:
            continue
        X.append(compute_trifriction(ex['premise'], ex['hypothesis'], sorted_neighbors))
        y.append(ex['label'])
        genres.append(ex.get('genre', 'unknown'))
    return np.array(X), np.array(y), genres


def run_benchmark():
    print("=" * 72)
    print("  Phase 25 — MultiNLI : TSO v3.1 sur matched + mismatched")
    print("=" * 72)

    # 1. Graph
    print("\n[1] Graphe R-STDP...")
    if os.path.exists(CACHE_GRAPH):
        with open(CACHE_GRAPH, "rb") as f:
            graph = pickle.load(f)
        print(f"    {len(graph)} nœuds (cache)")
    else:
        graph = {}
        ds = load_dataset('nyu-mll/multi_nli', split='train', streaming=True)
        t0 = time.time()
        for i, ex in enumerate(ds):
            if i >= N_TRAIN: break
            learn_sentence(ex['premise'], graph)
            learn_sentence(ex['hypothesis'], graph)
            if (i + 1) % 20000 == 0:
                print(f"    {i+1}/{N_TRAIN} ({len(graph)} nœuds, {time.time()-t0:.1f}s)", flush=True)
        with open(CACHE_GRAPH, "wb") as f:
            pickle.dump(graph, f, protocol=5)
        print(f"    {len(graph)} nœuds en {time.time()-t0:.1f}s")

    # 2. Pre-sort neighbors
    print("\n[2] Pré-tri voisins...")
    t0 = time.time()
    sorted_neighbors = prepare_sorted_neighbors(graph)
    print(f"    {time.time()-t0:.1f}s")

    # 3. Extract features (cached)
    print("\n[3] Extraction features tri-friction...")
    if os.path.exists(CACHE_FEATURES):
        with open(CACHE_FEATURES, "rb") as f:
            data = pickle.load(f)
        Xm, ym, gm = data['matched']
        Xmm, ymm, gmm = data['mismatched']
        print(f"    Matched: {len(Xm)} | Mismatched: {len(Xmm)} (cache)")
    else:
        Xm, ym, gm = extract_from_split('validation_matched', sorted_neighbors, N_VAL)
        Xmm, ymm, gmm = extract_from_split('validation_mismatched', sorted_neighbors, N_VAL)
        data = {'matched': (Xm, ym, gm), 'mismatched': (Xmm, ymm, gmm)}
        with open(CACHE_FEATURES, "wb") as f:
            pickle.dump(data, f, protocol=5)
        print(f"    Matched: {len(Xm)} | Mismatched: {len(Xmm)}")

    # 4. Attractors
    print("\n[4] Attracteurs euclidiens...")
    attractors = {}
    for label in [0, 1, 2]:
        mask = ym == label
        attractors[label] = Xm[mask].mean(axis=0) if mask.sum() > 0 else np.array([0.5, 0.5, 0.5])
    for l, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        a = attractors[l]
        print(f"    {name:>15s} : [{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}]")

    def predict(X):
        preds = []
        for x in X:
            scores = {l: -np.linalg.norm(x - a) for l, a in attractors.items()}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

    # 5. Evaluation
    print("\n[5] Évaluation :")
    acc_m = (predict(Xm) == ym).mean() * 100
    acc_mm = (predict(Xmm) == ymm).mean() * 100
    rand = 100 / 3

    print(f"    {'Split':>20s} | {'Acc':>6s} | {'vs Rnd':>6s}")
    print(f"    {'-'*20} | {'-'*6} | {'-'*6}")
    print(f"    {'Matched':>20s} | {acc_m:>5.1f}% | {acc_m-rand:+>5.1f}%")
    print(f"    {'Mismatched':>20s} | {acc_mm:>5.1f}% | {acc_mm-rand:+>5.1f}%")
    print(f"    {'Random':>20s} | {rand:>5.1f}% | {'—':>6s}")
    print(f"\n    Gap: {abs(acc_m - acc_mm):.1f} points")

    # 6. Genre analysis
    print(f"\n[6] Par genre (mismatched) :")
    preds_mm = predict(Xmm)
    for genre in sorted(set(gmm)):
        mask = np.array(gmm) == genre
        if mask.sum() >= 10:
            a = (preds_mm[mask] == ymm[mask]).mean() * 100
            print(f"    {genre:>25s} | {a:>5.1f}% ({mask.sum()} ex.)")

    # 7. Comparison
    print(f"\n[7] Comparaison :")
    print(f"    {'Benchmark':>20s} | {'Acc':>6s} | {'Grad':>6s}")
    print(f"    {'-'*20} | {'-'*6} | {'-'*6}")
    print(f"    {'SNLI (Ph24)':>20s} | {44.2:>5.1f}% | {'none':>6s}")
    print(f"    {'MNLI matched':>20s} | {acc_m:>5.1f}% | {'none':>6s}")
    print(f"    {'MNLI mismatched':>20s} | {acc_mm:>5.1f}% | {'none':>6s}")

    if abs(acc_m - acc_mm) <= 3:
        print(f"\n  *** Généralisation validée (gap ≤ 3 pts) ***")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase25_results.csv", "w") as f:
        f.write(f"matched,{acc_m:.2f}\nmismatched,{acc_mm:.2f}\n")
    print(f"\n    → experiments/phase25_results.csv")
    print("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
