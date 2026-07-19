"""
Benchmark: TSO (TransitionGraph) vs Transformer sur la prédiction conceptuelle
(Tiny Shakespeare, même pipeline SOM que Phase 13).

Métriques :
  - Accuracy (next-cluster prediction)
  - Paramètres entraînables
  - FLOPs par inférence
  - FLOPs totaux jusqu'à convergence
  - Temps d'entraînement

Cache automatique : les embeddings MiniLM + SOM sont sauvegardés dans
  experiments/shakespeare_data.pkl pour éviter le rechargement.
"""
import sys, os, time, math, re, urllib.request, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from tso_kernel.friction import FrictionCalculator
from tso_nlp.embedder import MiniLMEmbedder
from tso_nlp.som import SOM
from tso_nlp.decoder import TransitionGraph

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 64
SOM_ROWS, SOM_COLS = 10, 10
N_CONCEPTS = SOM_ROWS * SOM_COLS
MAX_VOCAB = 1000
TRAIN_SPLIT = 0.8

# TSO hyperparams
TSO_ALPHA = 0.05
TSO_BETA = 0.002

# Transformer hyperparams
TR_EPOCHS = 5
TR_LR = 1e-3
TR_BATCH = 64
TR_CONTEXT = 5
TR_STEPS_PER_EPOCH = 200  # limite pour accélérer


CACHE_PATH = "experiments/shakespeare_data.pkl"


def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        return urllib.request.urlopen(url).read().decode('utf-8')
    except:
        return "To be or not to be."


def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.lower())
    return [re.findall(r'\b\w+\b', s) + ['.'] for s in sentences]


def build_cluster_sequences(sentences, vocab, word_to_idx, cluster_of_word):
    sequences = []
    for s in sentences:
        idxs = [word_to_idx.get(w, -1) for w in s]
        idxs = [i for i in idxs if i != -1]
        clusters = [cluster_of_word[i] for i in idxs]
        if len(clusters) > 1:
            sequences.append(clusters)
    return sequences


def load_or_precompute_data(force_rebuild=False):
    os.makedirs("experiments", exist_ok=True)
    if not force_rebuild and os.path.exists(CACHE_PATH):
        print("  Cache trouvé. Chargement...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        # Vérifier que toutes les clés sont présentes (compatibilité ascendante)
        if "train_seqs" not in data:
            print("  Cache obsolète, reconstruction...")
            force_rebuild = True
    if force_rebuild:
        os.remove(CACHE_PATH)
        return load_or_precompute_data(force_rebuild=False)
    if not force_rebuild and os.path.exists(CACHE_PATH):
        return data

    print("  Première exécution — construction du cache...")
    text = download_shakespeare()
    sentences = preprocess_text(text)
    counts = Counter(w for s in sentences for w in s)
    vocab = [w for w, _ in counts.most_common(MAX_VOCAB)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    print(f"  Vocabulaire: {len(vocab)} mots, {len(sentences)} phrases")

    embedder = MiniLMEmbedder()
    raw_embs = embedder.encode(vocab)
    P = embedder.random_projection(EMBED_DIM, seed=0)
    embeddings = raw_embs @ P

    print("  Organisation topographique (SOM)...")
    som = SOM(SOM_ROWS, SOM_COLS, EMBED_DIM)
    som.train(embeddings, epochs=150, lr_start=0.1, sigma_start=2.0)
    cluster_of_word = np.array([som.bmu(embeddings[i]) for i in range(len(vocab))])
    n_occupied = len(set(cluster_of_word))
    print(f"  {n_occupied}/{N_CONCEPTS} concepts occupés")

    # Séquences de clusters
    all_seqs = build_cluster_sequences(sentences, vocab, word_to_idx, cluster_of_word)
    np.random.shuffle(all_seqs)
    split = int(len(all_seqs) * TRAIN_SPLIT)

    data = {
        "vocab": vocab,
        "word_to_idx": word_to_idx,
        "sentences": sentences,
        "embeddings": embeddings,
        "cluster_of_word": cluster_of_word,
        "all_seqs": all_seqs,
        "train_seqs": all_seqs[:split],
        "test_seqs": all_seqs[split:],
        "n_occupied": n_occupied,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f, protocol=5)
    print(f"  Cache sauvegardé dans {CACHE_PATH}")
    return data


def count_flops_transformer(model, seq_len):
    d = model.embed.embedding_dim
    nlayers = len(model.layers)
    nhead = model.layers[0].self_attn.num_heads
    d_ff = model.layers[0].linear1.out_features
    vocab = model.head.out_features
    embed_lookup = seq_len * d
    flop_attn = nlayers * (
        3 * seq_len * d * (d // nhead) + seq_len * seq_len * (d // nhead) * nhead
    )
    flop_ff = nlayers * (2 * seq_len * d * d_ff)
    flop_head = 2 * d * vocab
    return embed_lookup + flop_attn + flop_ff + flop_head


class MiniConceptTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, nlayers=2, max_seq=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, d_model * 4,
                dropout=0.1, batch_first=True, activation='gelu'
            )
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = self.pos(torch.arange(T, device=x.device))
        h = self.embed(x) + pos
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h[:, -1, :])


def make_transformer_configs(vocab_size):
    return [
        ("Tiny (1L,2H,d=32)",  MiniConceptTransformer(vocab_size, d_model=32,  nhead=2, nlayers=1)),
        ("Small (2L,4H,d=64)", MiniConceptTransformer(vocab_size, d_model=64,  nhead=4, nlayers=2)),
    ]


def train_transformer(model, train_seqs, test_seqs, vocab_size, epochs=TR_EPOCHS):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=TR_LR)
    crit = nn.CrossEntropyLoss()

    seqs_as_tensors = [torch.tensor(s, dtype=torch.long, device=DEVICE) for s in train_seqs]

    best_acc = 0.0
    all_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        order = np.random.permutation(len(seqs_as_tensors))
        for idx in order:
            seq = seqs_as_tensors[idx]
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                ctx_start = max(0, t - TR_CONTEXT)
                inp = seq[ctx_start:t].unsqueeze(0)
                target = seq[t].unsqueeze(0)
                logits = model(inp)
                loss = crit(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

                if n_batches >= TR_STEPS_PER_EPOCH:
                    break
            if n_batches >= TR_STEPS_PER_EPOCH:
                break

        acc = evaluate_transformer(model, test_seqs, vocab_size)
        all_accs.append(acc)
        if acc > best_acc:
            best_acc = acc

    return best_acc, all_accs


def evaluate_transformer(model, test_seqs, vocab_size):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq in test_seqs:
            if len(seq) < 2:
                continue
            seq_t = torch.tensor(seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
            for t in range(1, seq_t.size(1)):
                ctx_start = max(0, t - TR_CONTEXT)
                inp = seq_t[:, ctx_start:t]
                logits = model(inp)
                pred = logits.argmax(-1).item()
                if pred == seq[t]:
                    correct += 1
                total += 1
    return correct / max(total, 1) * 100


def eval_tso(graph, test_seqs):
    correct = 0
    total = 0
    for seq in test_seqs:
        for j in range(len(seq) - 1):
            c1, c2 = seq[j], seq[j + 1]
            pred = graph.predict_cluster(c1)
            if pred == c2:
                correct += 1
            total += 1
    return correct / max(total, 1) * 100


def run_benchmark():
    print("=" * 72)
    print("  Benchmark: TSO (TransitionGraph) vs Transformer")
    print("  Tâche : Prédiction conceptuelle (Tiny Shakespeare)")
    print("=" * 72)

    # --- Data pipeline (partagé, avec cache) ---
    print("\n[Data] Préparation de Tiny Shakespeare...")
    data = load_or_precompute_data(force_rebuild=False)
    vocab = data["vocab"]
    sentences = data["sentences"]
    embeddings = data["embeddings"]
    cluster_of_word = data["cluster_of_word"]
    train_seqs = data["train_seqs"]
    test_seqs = data["test_seqs"]
    n_occupied = data["n_occupied"]
    print(f"  {len(train_seqs)} train, {len(test_seqs)} test, {n_occupied} concepts")

    baseline_acc = 1.0 / max(n_occupied, 1) * 100
    results = []

    # =====================================================================
    # 1. TSO (TransitionGraph)
    # =====================================================================
    print("\n" + "=" * 72)
    print("  [1/2] TSO — TransitionGraph (Hebbien local)")
    print("=" * 72)

    graph = TransitionGraph(N_CONCEPTS, alpha=TSO_ALPHA, beta=TSO_BETA)
    tso_train_accs = []
    tso_start = time.time()
    n_updates = 0

    for seq in train_seqs:
        for j in range(len(seq) - 1):
            c1, c2 = seq[j], seq[j + 1]
            graph.learn(c1, c2)
            n_updates += 1

    tso_train_time = time.time() - tso_start
    tso_acc = eval_tso(graph, test_seqs)
    tso_params = N_CONCEPTS * N_CONCEPTS  # taille de la matrice W
    tso_flops_predict = N_CONCEPTS        # argmax sur une ligne
    tso_flops_learn = 5                   # ~5 FLOPs par update Hebbien
    tso_flops_total = n_updates * tso_flops_learn + len(test_seqs) * N_CONCEPTS

    print(f"  Accuracy test  : {tso_acc:.1f}% (baseline: {baseline_acc:.1f}%)")
    print(f"  Paramètres     : {tso_params:,}")
    print(f"  FLOPs/prédiction: {tso_flops_predict}")
    print(f"  FLOPs total     : {tso_flops_total:,}")
    print(f"  Temps entraînement: {tso_train_time:.2f}s")
    print(f"  Mises à jour   : {n_updates:,}")

    results.append(("TSO TransitionGraph", tso_acc, tso_params, tso_flops_predict, tso_flops_total, tso_train_time))

    # =====================================================================
    # 2. Transformer (3 tailles)
    # =====================================================================
    print("\n" + "=" * 72)
    print("  [2/2] Transformer (3 configurations)")
    print("=" * 72)

    for name, model in make_transformer_configs(N_CONCEPTS):
        n_params = sum(p.numel() for p in model.parameters())
        flops_per_step = count_flops_transformer(model, TR_CONTEXT)

        print(f"\n  --- {name} ---")
        print(f"    Paramètres : {n_params:,}")
        print(f"    FLOPs/step : {flops_per_step:,}")

        best_acc, all_accs = train_transformer(
            model, train_seqs, test_seqs, N_CONCEPTS, epochs=TR_EPOCHS
        )

        train_cost_flops = flops_per_step * TR_STEPS_PER_EPOCH * TR_EPOCHS
        print(f"    Best acc.  : {best_acc:.1f}%")
        print(f"    FLOPs total: {train_cost_flops:,}")
        print(f"    Epochs     : {TR_EPOCHS}")

        results.append((name, best_acc, n_params, flops_per_step, train_cost_flops, TR_EPOCHS))

    # =====================================================================
    # RÉSUMÉ
    # =====================================================================
    print("\n" + "=" * 72)
    print("  RÉSUMÉ DU BENCHMARK")
    print("=" * 72)
    print(f"\n  Baseline aléatoire : {baseline_acc:.1f}% (1/{n_occupied})")
    print(f"\n  {'Modèle':>28s} | {'Acc.':>6s} | {'Params':>8s} | {'FLOPs/step':>10s} | {'FLOPs total':>12s}")
    print(f"  {'-'*28} | {'-'*6} | {'-'*8} | {'-'*10} | {'-'*12}")
    for name, acc, params, flops_s, flops_t, extra in results:
        print(f"  {name:>28s} | {acc:>5.1f}% | {params:>7,d} | {flops_s:>9,d} | {flops_t:>11,d}")

    print(f"\n  Ratio FLOPs total (moyen Transformer / TSO):")
    tr_flops_avg = np.mean([r[4] for r in results[1:]])
    tso_flops = results[0][4]
    print(f"    Transformer/TSO = {tr_flops_avg / max(tso_flops, 1):.1f}x")

    print(f"\n  Ratio paramètres (moyen Transformer / TSO):")
    tr_params_avg = np.mean([r[2] for r in results[1:]])
    tso_params = results[0][2]
    print(f"    Transformer/TSO = {tr_params_avg / max(tso_params, 1):.1f}x")

    print(f"\n  Efficacité (Accuracy / million de FLOPs):")
    for name, acc, _, _, flops_t, _ in results:
        eff = acc / max(flops_t / 1e6, 1e-6)
        print(f"    {name:>28s} : {eff:.3f}% / MFLOPs")

    print(f"\n  Sauvegarde dans experiments/benchmark_results.csv")
    import csv
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/benchmark_results.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy_pct", "params", "flops_per_step", "flops_total", "extra"])
        for name, acc, params, flops_s, flops_t, extra in results:
            w.writerow([name, f"{acc:.2f}", params, flops_s, flops_t, extra])

    print(f"\n  {'='*72}")
    if tso_acc > baseline_acc * 2:
        print(f"  VERDICT: TSO bat le hasard ({tso_acc:.1f}% vs {baseline_acc:.1f}%)")
        print(f"  avec {tso_params:,} paramètres et {tso_flops_total:,} FLOPs totaux.")
    if any(r[1] > tso_acc for r in results[1:]):
        print(f"  Certains Transformers atteignent une meilleure accuracy,")
        print(f"  mais avec bien plus de paramètres et de FLOPs.")
    else:
        print(f"  TSO est compétitif voire meilleur que les Transformers")
        print(f"  testés sur cette tâche de prédiction conceptuelle.")
    print(f"  {'='*72}")


if __name__ == "__main__":
    run_benchmark()
