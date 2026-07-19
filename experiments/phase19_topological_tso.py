"""
Phase 19 — TSO Topologique Pur (Zéro Embedding).

Aucun MiniLM. Aucun SentenceTransformer. Aucun espace vectoriel.
Le sens émerge uniquement de la topologie du graphe construit par R-STDP.

Principe :
  - Lecture du corpus : fenêtre glissante Δt=5, R-STDP local
  - Chaque cooccurrence renforce le lien entre deux mots
  - Le graphe résultant = carte conceptuelle purement structurelle
  - Φ = 1 - Jaccard(N(P), N(H)) où N(X) = voisinage topologique

Pas de backprop, pas d'embedding, pas de pré-entraînement externe.
"""
import sys, os, time, pickle, random, json, builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CACHE_GRAPH = "experiments/topological_graph.pkl"
WINDOW_SIZE = 5
TOP_K_NEIGHBORS = 20
N_TRAIN_SENTENCES = 50000  # phrases SNLI train à lire
N_VAL = 2000
N_TEST = 2000
MAX_VOCAB = 10000


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
            sentences.append(d["sentence1"])
            sentences.append(d["sentence2"])
    return pairs, sentences


def tokenize(text):
    tokens = text.lower().split()
    return [t.strip(".,!?;:\"'()[]{}") for t in tokens if len(t.strip(".,!?;:\"'()[]{}")) > 0]


def build_rstdp_graph(sentences, window=5):
    """Construit un graphe topologique par R-STDP local.
    Chaque cooccurrence dans une fenêtre glissante renforce le lien.
    """
    graph = {}  # word -> {neighbor: weight}

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

    # Normaliser les poids : pour chaque mot, diviser par le max
    for word in graph:
        neighbors = graph[word]
        if not neighbors:
            continue
        max_w = max(neighbors.values())
        if max_w > 0:
            for n in neighbors:
                neighbors[n] /= max_w

    return graph


def extract_neighborhoods(graph, top_k=TOP_K_NEIGHBORS):
    """Pour chaque mot, garde les top_k voisins les plus forts."""
    hoods = {}
    for word, neighbors in graph.items():
        sorted_n = sorted(neighbors.items(), key=lambda x: -x[1])
        hoods[word] = {n for n, w in sorted_n[:top_k]}
    return hoods


def sentence_neighborhood(tokens, hoods):
    """Union des voisinages de tous les tokens d'une phrase."""
    nh = set()
    for t in tokens:
        if t in hoods:
            nh |= hoods[t]
    return nh


def jaccard_distance(set_a, set_b):
    if not set_a and not set_b:
        return 0.5
    union = len(set_a | set_b)
    if union == 0:
        return 0.5
    return 1.0 - len(set_a & set_b) / union


def run_benchmark():
    log = lambda msg: builtins.print(msg, flush=True)

    log("=" * 72)
    log("  Phase 19 — TSO Topologique Pur")
    log("  Aucun embedding. Aucun MiniLM. Rien que le graphe.")
    log("  R-STDP local + distance de Jaccard topologique")
    log("=" * 72)

    # =====================================================================
    # 1. Dataset
    # =====================================================================
    log("\n[1] Chargement SNLI...")
    train_all, train_sentences = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_train.jsonl")
    test_pairs, _ = load_snli_jsonl(
        "/tmp/snli_1.0/snli_1.0_test.jsonl")

    random.shuffle(train_all)
    train_pairs = train_all[:N_TRAIN_SENTENCES // 2]
    val_pairs = train_all[N_TRAIN_SENTENCES // 2:
                          N_TRAIN_SENTENCES // 2 + N_VAL]
    test_pairs = test_pairs[:N_TEST]

    # Limiter les phrases pour la construction du graphe
    corpus = train_sentences[:N_TRAIN_SENTENCES]
    log(f"    Corpus R-STDP: {len(corpus)} phrases")
    log(f"    Train pairs: {len(train_pairs)}")
    log(f"    Val pairs: {len(val_pairs)}")
    log(f"    Test pairs: {len(test_pairs)}")

    # =====================================================================
    # 2. Construction du graphe topologique par R-STDP
    # =====================================================================
    log("\n[2] Construction du graphe R-STDP...")

    if os.path.exists(CACHE_GRAPH):
        with open(CACHE_GRAPH, "rb") as f:
            data = pickle.load(f)
        graph = data["graph"]
        hoods = data["neighborhoods"]
        log(f"    Cache chargé: {len(graph)} nœuds")
    else:
        t0 = time.time()
        graph = build_rstdp_graph(corpus, window=WINDOW_SIZE)
        t_graph = time.time() - t0
        log(f"    {len(graph)} nœuds, "
            f"{sum(len(n) for n in graph.values()) // 2} arêtes "
            f"en {t_graph:.1f}s")

        t0 = time.time()
        hoods = extract_neighborhoods(graph)
        t_hood = time.time() - t0
        log(f"    Voisinages extraits en {t_hood:.1f}s")

        with open(CACHE_GRAPH, "wb") as f:
            pickle.dump({"graph": graph, "neighborhoods": hoods},
                        f, protocol=5)
        log(f"    Cache sauvegardé")

    # Stats du graphe
    degrees = [len(n) for n in graph.values()]
    log(f"    Degré moyen: {np.mean(degrees):.1f}")
    log(f"    Degré max: {max(degrees)}")

    # =====================================================================
    # 3. Calcul de Φ sur validation set
    # =====================================================================
    log("\n[3] Calcul de Φ topologique sur validation...")

    def phi_pair(premise, hypothesis, hoods):
        pt = tokenize(premise)
        ht = tokenize(hypothesis)
        nh_p = sentence_neighborhood(pt, hoods)
        nh_h = sentence_neighborhood(ht, hoods)
        return jaccard_distance(nh_p, nh_h)

    t0 = time.time()
    val_phis = np.array([phi_pair(p, h, hoods) for p, h, _ in val_pairs])
    val_labels = np.array([l for _, _, l in val_pairs])
    log(f"    {len(val_phis)} Φ en {time.time()-t0:.2f}s")

    # Profil Φ par classe
    log(f"\n    Φ moyen par classe :")
    for l, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        mask = val_labels == l
        log(f"      {name}: {val_phis[mask].mean():.4f} (σ={val_phis[mask].std():.4f})")

    # =====================================================================
    # 4. Calibration des seuils
    # =====================================================================
    log("\n[4] Calibration des seuils...")

    best_low, best_high = 0.3, 0.6
    best_acc = 0.0

    for low in np.linspace(val_phis.min(), np.percentile(val_phis, 50), 30):
        for high in np.linspace(np.percentile(val_phis, 50), val_phis.max(), 30):
            if low >= high:
                continue
            preds = np.zeros_like(val_labels)
            preds[val_phis < low] = 0
            preds[val_phis > high] = 2
            preds[(val_phis >= low) & (val_phis <= high)] = 1
            acc = (preds == val_labels).mean() * 100
            if acc > best_acc:
                best_acc = acc
                best_low, best_high = low, high

    log(f"    θ_low={best_low:.4f}, θ_high={best_high:.4f}")
    log(f"    Acc validation: {best_acc:.1f}%")

    # =====================================================================
    # 5. Test set
    # =====================================================================
    log("\n[5] Évaluation test...")
    t0 = time.time()
    test_phis = np.array([phi_pair(p, h, hoods) for p, h, _ in test_pairs])
    test_labels = np.array([l for _, _, l in test_pairs])
    log(f"    {len(test_phis)} Φ en {time.time()-t0:.2f}s")

    test_preds = np.zeros_like(test_labels)
    test_preds[test_phis < best_low] = 0
    test_preds[test_phis > best_high] = 2
    test_preds[(test_phis >= best_low) & (test_phis <= best_high)] = 1

    test_acc = (test_preds == test_labels).mean() * 100

    log(f"    Accuracy test: {test_acc:.1f}%")

    # Matrice de confusion
    log(f"\n    Matrice de confusion :")
    log(f"    {'':>15s} {'Entail':>8s} {'Neutral':>8s} {'Contra':>8s}")
    for i, name in enumerate(["Entailment", "Neutral", "Contradiction"]):
        row = [(test_labels == i) & (test_preds == j) for j in range(3)]
        log(f"    {name:>15s} {sum(row[0]):>8d} {sum(row[1]):>8d} {sum(row[2]):>8d}")

    for name, label in [("Entailment", 0), ("Neutral", 1), ("Contradiction", 2)]:
        mask = test_labels == label
        if mask.sum() > 0:
            acc = (test_preds[mask] == label).mean() * 100
            log(f"      {name}: {acc:.1f}% ({mask.sum()} ex.)")

    # =====================================================================
    # 6. Comparaison
    # =====================================================================
    random_acc = 100 / 3
    tso_vec_acc = 44.0  # Phase 18
    bert_acc = 80.4

    log(f"\n[6] Comparaison :")
    log(f"    {'Modèle':>35s} | {'Accuracy':>8s}")
    log(f"    {'-'*35} | {'-'*8}")
    log(f"    {'Random':>35s} | {random_acc:>6.1f}%")
    log(f"    {'TSO topologique pur (Phase 19)':>35s} | {test_acc:>6.1f}%")
    log(f"    {'TSO vectoriel (Phase 18)':>35s} | {tso_vec_acc:>6.1f}%")
    log(f"    {'BERT-base fine-tuné':>35s} | {bert_acc:>6.1f}%")

    delta = test_acc - random_acc
    gap_to_vec = test_acc - tso_vec_acc
    log(f"\n    Bat le hasard de {delta:.1f} points")
    log(f"    Delta vs TSO vectoriel: {gap_to_vec:+.1f} points")
    log(f"    Aucun embedding pré-entraîné utilisé.")

    if test_acc > random_acc + 5:
        log(f"\n  *** LA TOPOLOGIE FONCTIONNE ***")
        log(f"  Le sens émerge du graphe R-STDP seul")

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/phase19_results.csv", "w") as f:
        f.write("model,accuracy,embedding\n")
        f.write(f"TSO_Topological,{test_acc:.2f},none\n")
        f.write(f"TSO_Vector,{tso_vec_acc:.2f},MiniLM\n")
        f.write(f"BERT,{bert_acc:.2f},WordPiece\n")
        f.write(f"Random,{random_acc:.2f},none\n")
    log(f"\n    → experiments/phase19_results.csv")
    log("=" * 72)


if __name__ == "__main__":
    t0 = time.time()
    run_benchmark()
    builtins.print(f"\nDurée : {time.time()-t0:.1f}s", flush=True)
