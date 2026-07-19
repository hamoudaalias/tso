"""
Phase 23 — TSO v3 : Attracteurs Topologiques (Zéro Gradient).

On tue la régression logistique globale.

La décision Entailment/Neutral/Contradiction émerge des attracteurs
topologiques du graphe. Chaque classe = une région de l'espace de
friction = un attracteur naturel.

Mécanisme :
  1. Chaque attracteur est le centroïde des vecteurs de friction
     [support, conflit, nouveauté] des exemples de sa classe.
  2. Pour une nouvelle paire, on projette son Φ dans l'espace
     des attracteurs.
  3. La classe gagnante = attracteur le plus proche (distance cosinus).
  4. Aucun gradient. Aucun poids appris. 3 prototypes.

Purement topologique. Purement local. 100% TSO.
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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
    pairs = []
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
    return pairs


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
    return np.array([support, conflict, novelty])


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


class TopologicalAttractorField:
    """Champ d'attracteurs topologiques — zéro gradient.

    Chaque attracteur est le centroïde des frictions d'une classe.
    La décision = attracteur le plus proche (similarité cosinus).
    """

    def __init__(self):
        self.attractors = {}  # label -> centroid vector [s, c, n]
        self.label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    def fit(self, X, y):
        """Calcule les attracteurs = centroïdes par classe."""
        for label in [0, 1, 2]:
            mask = y == label
            if mask.sum() > 0:
                self.attractors[label] = X[mask].mean(axis=0)
            else:
                self.attractors[label] = np.array([0.5, 0.5, 0.5])

    def predict(self, X):
        """Prédit par proximité d'attracteur (cosinus)."""
        preds = []
        for x in X:
            scores = {}
            for label, attr in self.attractors.items():
                scores[label] = cosine_sim(x, attr)
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

    def predict_with_scores(self, X):
        """Retourne prédictions + scores d'attracteurs."""
        results = []
        for x in X:
            scores = {}
            for label, attr in self.attractors.items():
                scores[self.label_names[label]] = cosine_sim(x, attr)
            pred = max(scores, key=scores.get)
            results.append((pred, scores))
        return results


def run_benchmark():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 23 — TSO v3 : Attracteurs Topologiques")
    log("  Zéro gradient. Zéro régression logistique.")
    log("  La décision émerge des attracteurs du graphe.")
    log("=" * 72)

    # =====================================================================
    # 1. Graphe + Dataset
    # =====================================================================
    log("\n[1] Chargement...")
    if not os.path.exists(CACHE):
        log("    ERREUR : lancer Phase 19 d'abord")
        return
    with open(CACHE, "rb") as f:
        graph = pickle.load(f)["graph"]
    log(f"    Graphe: {len(graph)} nœuds")

    test_pairs = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_test.jsonl")[:N_TEST]
    val_all = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_train.jsonl")
    random.shuffle(val_all)
    val_pairs = val_all[:N_VAL]
    train_pairs = val_all[N_VAL:N_VAL + 3000]  # pour calibrer les attracteurs
    log(f"    Train attracteurs: {len(train_pairs)} | "
        f"Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # =====================================================================
    # 2. Attracteurs topologiques (zéro gradient)
    # =====================================================================
    log("\n[2] Calcul des attracteurs topologiques...")
    t0 = time.time()

    X_train = np.array([compute_trifriction(p, h, graph)
                        for p, h, _ in train_pairs])
    y_train = np.array([l for _, _, l in train_pairs])

    field = TopologicalAttractorField()
    field.fit(X_train, y_train)

    log(f"    Attracteurs :")
    for label, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        attr = field.attractors[label]
        log(f"      {name:>15s} : [supp={attr[0]:.4f}, "
            f"conf={attr[1]:.4f}, nov={attr[2]:.4f}]")

    t_attr = time.time() - t0
    log(f"    Temps: {t_attr:.3f}s (0 gradient)")

    # =====================================================================
    # 3. Évaluation
    # =====================================================================
    log("\n[3] Évaluation (attracteurs seuls, 0 gradient)...")

    X_val = np.array([compute_trifriction(p, h, graph) for p, h, _ in val_pairs])
    y_val = np.array([l for _, _, l in val_pairs])
    X_test = np.array([compute_trifriction(p, h, graph) for p, h, _ in test_pairs])
    y_test = np.array([l for _, _, l in test_pairs])

    val_preds = field.predict(X_val)
    test_preds = field.predict(X_test)

    val_acc = (val_preds == y_val).mean() * 100
    test_acc = (test_preds == y_test).mean() * 100
    log(f"    Validation: {val_acc:.1f}%")
    log(f"    Test: {test_acc:.1f}%")

    # Détail par classe
    log(f"\n    Détail par classe :")
    cm = confusion_matrix(y_test, test_preds)
    log(f"    {'':>15s} {'Entail':>8s} {'Neutral':>8s} {'Contra':>8s}")
    for i, name in enumerate(["Entailment", "Neutral", "Contradiction"]):
        log(f"    {name:>15s} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")
    for name, label in [("Entailment", 0), ("Neutral", 1), ("Contradiction", 2)]:
        mask = y_test == label
        acc = (test_preds[mask] == label).mean() * 100
        log(f"      {name}: {acc:.1f}% ({mask.sum()} ex.)")

    # =====================================================================
    # 4. Scores d'attracteurs (exemples)
    # =====================================================================
    log(f"\n[4] Exemples de scores d'attracteurs :")
    results = field.predict_with_scores(X_test[:10])
    for i, (pred_name, scores) in enumerate(results):
        p, h, true = test_pairs[i]
        true_name = ["Entailment", "Neutral", "Contradiction"][true]
        s = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
        log(f"    [{i}] vrai={true_name} préd={pred_name} | {s}")

    # =====================================================================
    # 5. Comparaison
    # =====================================================================
    log(f"\n[5] Comparaison :")
    acc_lr = 48.4  # Phase 20 (LR)
    acc_random = 33.3

    log(f"    {'Approche':>30s} | {'Accuracy':>8s} | {'Gradient':>10s}")
    log(f"    {'-'*30} | {'-'*8} | {'-'*10}")
    log(f"    {'Random':>30s} | {acc_random:>6.1f}% | {'non':>10s}")
    log(f"    {'TSO v2 LR (Phase 20)':>30s} | {acc_lr:>6.1f}% | {'oui (LR)':>10s}")
    log(f"    {'TSO v3 attracteurs':>30s} | {test_acc:>6.1f}% | {'non':>10s}")
    log(f"    {'BERT-base':>30s} | {80.4:>6.1f}% | {'oui (BP)':>10s}")

    delta = test_acc - acc_lr
    log(f"\n    Delta vs TSO v2 (LR): {delta:+.1f} points")

    if test_acc >= acc_lr - 2:
        log(f"\n  *** ATTRACTEURS VALIDÉS ***")
        log(f"  Zéro gradient, zéro régression logistique.")
        log(f"  La topologie seule suffit à la décision.")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase23_results.csv", "w") as f:
        f.write("model,accuracy,gradient\n")
        f.write(f"TSO_v3_Attractors,{test_acc:.2f},none\n")
        f.write(f"TSO_v2_LR,{acc_lr:.2f},logistic\n")
        f.write(f"Random,{acc_random:.2f},none\n")
    log(f"\n    → experiments/phase23_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
