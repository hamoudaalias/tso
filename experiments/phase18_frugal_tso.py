"""
Phase 18 (Frugale) — TSO enrichi sans nouveau SOM.

Pas de nouvelle SOM. Pas de compression. Pas de parallélisation lourde.
On réutilise le cache de Phase 17 (144 concepts) et on ajoute 2 métriques
calculées en O(1) à partir des distributions de clusters déjà acquises :

  [Φ, H(hypothèse), n_clusters(hypothèse)]

  Φ = JS divergence (friction de base, Phase 17)
  H = entropie de la distribution de clusters (ambiguïté → Neutre)
  n_clusters = nombre de clusters distincts activés (largeur thématique → Neutre)

Fusion : LogisticRegression sur ces 3 features. Zéro entraînement SOM.
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CACHE = "experiments/snli_data.pkl"
N_TRAIN = 5000
N_VAL = 2000
N_TEST = 2000


def load_snli_jsonl(path, max_examples=None):
    pairs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            d = json.loads(line)
            if d["gold_label"] == "-":
                continue
            pairs.append((d["sentence1"], d["sentence2"],
                          {"entailment": 0, "neutral": 1, "contradiction": 2}[d["gold_label"]]))
    return pairs


def sentence_cluster_dist(sentence, word_to_idx, cluster_of_word, n_concepts):
    words = sentence.lower().split()
    idxs = [word_to_idx.get(w, -1) for w in words if w in word_to_idx]
    if not idxs:
        dist = np.zeros(n_concepts)
        dist[0] = 1.0
        return dist, []
    clusters = [cluster_of_word[i] for i in idxs]
    dist = np.zeros(n_concepts)
    for c in clusters:
        dist[c] += 1.0
    dist /= dist.sum() + 1e-8
    return dist, list(set(clusters))


def js_div(p, q):
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def entropy(dist):
    dist = np.clip(dist, 1e-10, 1.0)
    return float(-np.sum(dist * np.log(dist)))


def run_benchmark():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 18 (Frugale) — TSO enrichi, 0 SOM training")
    log("  Features : [Φ, H(hypothèse), n_clusters(hypothèse)]")
    log("  Fusion : LogisticRegression (3 params)")
    log("=" * 72)

    # =====================================================================
    # 1. Cache
    # =====================================================================
    log("\n[1] Chargement cache SOM (144 concepts, Phase 17)...")
    if not os.path.exists(CACHE):
        log("    ERREUR: cache snli_data.pkl introuvable. Lance Phase 17 d'abord.")
        return
    with open(CACHE, "rb") as f:
        c = pickle.load(f)
    vocab = c["vocab"]
    cluster_of_word = c["cluster_of_word"]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n_concepts = len(set(cluster_of_word))
    log(f"    {len(vocab)} mots, {n_concepts} concepts")

    # =====================================================================
    # 2. Dataset
    # =====================================================================
    log("\n[2] Chargement SNLI...")
    train_all = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_train.jsonl")
    test_all = load_snli_jsonl("/tmp/snli_1.0/snli_1.0_test.jsonl")
    random.shuffle(train_all)
    train_pairs = train_all[:N_TRAIN]
    val_pairs = train_all[N_TRAIN:N_TRAIN + N_VAL]
    test_pairs = test_all[:N_TEST]
    log(f"    Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # =====================================================================
    # 3. Features
    # =====================================================================
    log("\n[3] Calcul features [Φ, H, n_clusters]...")

    def extract(pairs):
        X, y = [], []
        for p, h, l in pairs:
            dp, _ = sentence_cluster_dist(p, word_to_idx, cluster_of_word, n_concepts)
            dh, clusters_h = sentence_cluster_dist(h, word_to_idx, cluster_of_word, n_concepts)

            f_phi = js_div(dp, dh)
            f_ent = entropy(dh)
            f_n = len(clusters_h) / n_concepts  # normalisé

            X.append([f_phi, f_ent, f_n])
            y.append(l)
        return np.array(X), np.array(y)

    X_train, y_train = extract(train_pairs)
    X_val, y_val = extract(val_pairs)
    X_test, y_test = extract(test_pairs)

    log(f"    Features: {X_train.shape}")

    # Statistiques
    log(f"\n    Profil feature moyen par classe :")
    log(f"    {'Classe':>15s} {'Φ':>8s} {'H':>8s} {'n_clusters':>10s}")
    for l, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        means = X_train[y_train == l].mean(axis=0)
        log(f"    {name:>15s} {means[0]:>8.4f} {means[1]:>8.4f} {means[2]:>10.4f}")

    # =====================================================================
    # 4. Fusion douce
    # =====================================================================
    log("\n[4] Fusion douce (LogisticRegression 3 features)...")
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_train, y_train)

    val_acc = clf.score(X_val, y_val) * 100
    test_acc = clf.score(X_test, y_test) * 100
    log(f"    Validation: {val_acc:.1f}%")
    log(f"    Test: {test_acc:.1f}%")

    log(f"\n    Poids de la fusion :")
    names = ["Φ (JS)", "H (entropie)", "n_clusters"]
    for j, name in enumerate(["Entailment", "Neutral", "Contradiction"]):
        line = f"      {name}: "
        for i, fn in enumerate(names):
            line += f"{fn}={clf.coef_[j][i]:+.4f}  "
        log(line)

    # =====================================================================
    # 5. Détail test
    # =====================================================================
    y_pred = clf.predict(X_test)

    label_names = ["Entailment", "Neutral", "Contradiction"]

    log(f"\n[5] Matrice de confusion (test) :")
    cm = confusion_matrix(y_test, y_pred)
    log(f"    {'':>15s} {'Entail':>8s} {'Neutral':>8s} {'Contra':>8s}")
    for i, name in enumerate(label_names):
        log(f"    {name:>15s} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    for name, label in zip(label_names, [0, 1, 2]):
        mask = y_test == label
        acc = (y_pred[mask] == label).mean() * 100
        log(f"      {name}: {acc:.1f}% ({mask.sum()} ex.)")

    # =====================================================================
    # 6. Comparaison
    # =====================================================================
    random_acc = 100 / 3
    phase17_acc = 40.5
    bert_acc = 80.4

    log(f"\n[6] Comparaison :")
    log(f"    {'Modèle':>32s} | {'Accuracy':>8s}")
    log(f"    {'-'*32} | {'-'*8}")
    log(f"    {'Random':>32s} | {random_acc:>6.1f}%")
    log(f"    {'TSO plat (Phase 17, seul Φ)':>32s} | {phase17_acc:>6.1f}%")
    log(f"    {'TSO frugal (Φ+H+n_clusters)':>32s} | {test_acc:>6.1f}%")
    log(f"    {'BERT-base fine-tuné':>32s} | {bert_acc:>6.1f}%")

    delta = test_acc - phase17_acc
    emoji = "🔼" if delta > 0 else "🔽"
    log(f"\n    Delta vs Phase 17 : {emoji} {abs(delta):+.1f} points")

    # Gain sur Neutre
    neutral_mask = y_test == 1
    neutral_acc_18 = (y_pred[neutral_mask] == 1).mean() * 100
    log(f"\n    Classe NEUTRE : {neutral_acc_18:.1f}% (Phase 17: 35.9%)")

    if test_acc > phase17_acc:
        log(f"\n  *** AMÉLIORATION VALIDÉE ***")
        log(f"  Features gratuites suffisent — pas de nouveau SOM")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase18_results.csv", "w") as f:
        f.write("model,accuracy\n")
        f.write(f"TSO_Frugal,{test_acc:.2f}\n")
        f.write(f"TSO_Flat,{phase17_acc:.2f}\n")
        f.write(f"BERT,{bert_acc:.2f}\n")
        f.write(f"Random,{random_acc:.2f}\n")
    log(f"\n    → experiments/phase18_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
