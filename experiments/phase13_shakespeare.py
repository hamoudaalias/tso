"""
Phase 13 — Conceptual Shakespeare Reader using the TSO Kernel.

TSO reads Tiny Shakespeare, quantizes words onto a SOM (100 concepts),
and learns a transition graph between clusters via local Hebbian updates.
The friction Phi drops as the model internalizes grammatical structure.
"""
import sys, os, time, math, re, urllib.request, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

from tso_kernel.friction import FrictionCalculator
from tso_nlp.embedder import MiniLMEmbedder
from tso_nlp.som import SOM
from tso_nlp.decoder import TransitionGraph

CACHE_PATH = "experiments/shakespeare_data.pkl"

SEED = 42
np.random.seed(SEED)

EMBED_DIM = 64
SOM_ROWS, SOM_COLS = 10, 10
N_CONCEPTS = SOM_ROWS * SOM_COLS


def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("  Téléchargement de Tiny Shakespeare...")
    try:
        return urllib.request.urlopen(url).read().decode('utf-8')
    except:
        return "To be or not to be."


def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.lower())
    return [re.findall(r'\b\w+\b', s) + ['.'] for s in sentences]


def run_phase13():
    print("=" * 72)
    print("  Phase 13 — Conceptual Shakespeare Reader (Kernel)")
    print("=" * 72)

    # Chargement avec cache partagé
    os.makedirs("experiments", exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("  Cache trouvé, chargement...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
    else:
        text = download_shakespeare()
        sentences = preprocess_text(text)
        counts = Counter(w for s in sentences for w in s)
        vocab = [w for w, _ in counts.most_common(1000)]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        print(f"  Vocabulaire: {len(vocab)} mots, {len(sentences)} phrases")

        embedder = MiniLMEmbedder()
        raw_embs = embedder.encode(vocab)
        P = embedder.random_projection(EMBED_DIM, seed=0)
        embeddings = raw_embs @ P

        print("\n  Organisation topographique (SOM)...")
        som = SOM(SOM_ROWS, SOM_COLS, EMBED_DIM)
        som.train(embeddings, epochs=150, lr_start=0.1, sigma_start=2.0)
        cluster_of_word = np.array([som.bmu(embeddings[i]) for i in range(len(vocab))])

        data = {
            "vocab": vocab,
            "word_to_idx": word_to_idx,
            "sentences": sentences,
            "embeddings": embeddings,
            "cluster_of_word": cluster_of_word,
            "n_occupied": len(set(cluster_of_word)),
        }
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(data, f, protocol=5)
        print(f"  Cache sauvegardé dans {CACHE_PATH}")

    vocab = data["vocab"]
    sentences = data["sentences"]
    embeddings = data["embeddings"]
    cluster_of_word = data["cluster_of_word"]
    n_occupied = data["n_occupied"]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    print(f"  {n_occupied}/{N_CONCEPTS} concepts occupés")

    graph = TransitionGraph(N_CONCEPTS, alpha=0.05, beta=0.002)
    calc = FrictionCalculator()

    all_phis = []
    all_accs = []
    block_size = 1000
    n_blocks = max(1, len(sentences) // block_size)
    baseline_phi = 0.0  # phi for random uniform = 1 - (1/N)*N = 0
    n_occupied = len(set(cluster_of_word))
    baseline_acc = 1.0 / max(n_occupied, 1) * 100

    print(f"\n  Lecture Shakespeare ({n_blocks} blocs de {block_size} phrases)...")
    print(f"  {'Bloc':>5s} | {'Φ':>8s} | {'Acc':>6s} | {'ΔAcc':>7s}")
    print("  " + "-" * 34)

    start = time.time()
    for block in range(n_blocks):
        block_phis = []
        block_correct = 0
        block_total = 0
        for i in range(block * block_size, (block + 1) * block_size):
            if i >= len(sentences):
                break
            s = sentences[i]
            idxs = [word_to_idx.get(w, -1) for w in s]
            idxs = [i for i in idxs if i != -1]
            for j in range(len(idxs) - 1):
                c1, c2 = cluster_of_word[idxs[j]], cluster_of_word[idxs[j+1]]
                phi = calc.conceptual_phi(c1, c2, graph.W, N_CONCEPTS)
                block_phis.append(phi)
                predicted = graph.predict_cluster(c1)
                if predicted == c2:
                    block_correct += 1
                block_total += 1
                graph.learn(c1, c2)

        avg = np.mean(block_phis) if block_phis else 0.0
        all_phis.append(avg)
        acc = block_correct / max(block_total, 1) * 100
        all_accs.append(acc)
        delta_acc = acc - baseline_acc
        if block % 2 == 0 or block == n_blocks - 1:
            print(f"  {block+1:3d}/{n_blocks} | {avg:.4f} | {acc:5.1f}% | {delta_acc:+.1f}%")

    elapsed = time.time() - start
    print("  " + "-" * 34)

    first_phi = np.mean(all_phis[:3])
    last_phi = np.mean(all_phis[-3:])
    first_acc = np.mean(all_accs[:3])
    last_acc = np.mean(all_accs[-3:])
    gain_phi = (baseline_phi - last_phi) / max(abs(baseline_phi), 1e-6) * 100 if baseline_phi != 0 else 0
    improvement_acc = ((last_acc - first_acc) / max(first_acc, 1e-6)) * 100

    print(f"\n  Temps: {elapsed:.1f}s")
    print(f"  Concepts occupés: {n_occupied}/{N_CONCEPTS}")
    print(f"  Baseline aléatoire:             Acc = {baseline_acc:.1f}%")
    print(f"  Φ initial (blocs 1-3):          {first_phi:.4f}")
    print(f"  Φ final  (blocs {n_blocks-2}-{n_blocks}):   {last_phi:.4f}")
    print(f"  Accuracy initiale (blocs 1-3):  {first_acc:.1f}%")
    print(f"  Accuracy finale  (blocs {n_blocks-2}-{n_blocks}):  {last_acc:.1f}%")
    print(f"  Amélioration relative:          {improvement_acc:.1f}%")

    phis_per_block = list(zip(range(1, n_blocks+1), all_phis))
    print(f"\n  Top-5 transitions apprises:")
    flat = [(i, j, graph.W[i, j]) for i in range(N_CONCEPTS) for j in range(N_CONCEPTS)]
    flat.sort(key=lambda x: -x[2])
    for i, j, w in flat[:5]:
        words_i = [vocab[k] for k in range(len(vocab)) if cluster_of_word[k] == i][:2]
        words_j = [vocab[k] for k in range(len(vocab)) if cluster_of_word[k] == j][:2]
        print(f"    concept {i:2d} ({','.join(words_i):12s}) → concept {j:2d} ({','.join(words_j):12s})  W={w:.2f}")

    if last_acc > 3 * baseline_acc:
        print(f"\n  *** VALIDATION : TSO infère la grammaire conceptuelle de Shakespeare ***")
        print(f"  Accuracy {last_acc:.1f}% vs baseline {baseline_acc:.1f}% ({last_acc/baseline_acc:.0f}x)")
    else:
        print(f"\n  Résultat: Accuracy {last_acc:.1f}% (baseline {baseline_acc:.1f}%)")


if __name__ == "__main__":
    run_phase13()
