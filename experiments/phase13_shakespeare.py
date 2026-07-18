"""
Phase 13 — Conceptual Shakespeare Reader using the TSO Kernel.

TSO reads Tiny Shakespeare, quantizes words onto a SOM (100 concepts),
and learns a transition graph between clusters via local Hebbian updates.
The friction Phi drops as the model internalizes grammatical structure.
"""
import sys, os, time, math, re, urllib.request
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

from tso_kernel.friction import FrictionCalculator
from tso_nlp.embedder import MiniLMEmbedder
from tso_nlp.som import SOM
from tso_nlp.decoder import TransitionGraph

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
    print(f"  {len(set(cluster_of_word))}/{N_CONCEPTS} concepts occupés")

    graph = TransitionGraph(N_CONCEPTS, alpha=0.05, beta=0.002)
    calc = FrictionCalculator()

    all_phis = []
    block_size = 1000
    n_blocks = max(1, len(sentences) // block_size)
    baseline = 1.0 - 1.0 / N_CONCEPTS

    print(f"\n  Lecture Shakespeare ({n_blocks} blocs de {block_size} phrases)...")
    print(f"  {'Bloc':>5s} | {'Φ':>8s} | {'Δ vs baseline':>12s}")
    print("  " + "-" * 32)

    start = time.time()
    for block in range(n_blocks):
        block_phis = []
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
                graph.learn(c1, c2)

        avg = np.mean(block_phis) if block_phis else 0.0
        all_phis.append(avg)
        imp = (baseline - avg) / baseline * 100
        if block % 2 == 0 or block == n_blocks - 1:
            print(f"  {block+1:3d}/{n_blocks} | {avg:.4f} | {imp:+.2f}%")

    elapsed = time.time() - start
    print("  " + "-" * 32)

    first = np.mean(all_phis[:3])
    last = np.mean(all_phis[-3:])
    gain_initial = (baseline - first) / baseline * 100
    gain_final = (baseline - last) / baseline * 100
    improvement = (first - last) / abs(first) * 100

    print(f"\n  Temps: {elapsed:.1f}s")
    print(f"  Baseline aléatoire:             {baseline:.4f}")
    print(f"  Φ initial (blocs 1-3):          {first:.4f} ({gain_initial:.0f}% mieux que hasard)")
    print(f"  Φ final  (blocs {n_blocks-2}-{n_blocks}):   {last:.4f} ({gain_final:.0f}% mieux que hasard)")
    print(f"  Amélioration relative:          {improvement:.1f}%")

    phis_per_block = list(zip(range(1, n_blocks+1), all_phis))
    print(f"\n  Top-5 transitions apprises:")
    flat = [(i, j, graph.W[i, j]) for i in range(N_CONCEPTS) for j in range(N_CONCEPTS)]
    flat.sort(key=lambda x: -x[2])
    for i, j, w in flat[:5]:
        words_i = [vocab[k] for k in range(len(vocab)) if cluster_of_word[k] == i][:2]
        words_j = [vocab[k] for k in range(len(vocab)) if cluster_of_word[k] == j][:2]
        print(f"    concept {i:2d} ({','.join(words_i):12s}) → concept {j:2d} ({','.join(words_j):12s})  W={w:.2f}")

    if gain_final > 100:
        print("\n  *** VALIDATION : TSO infère la grammaire conceptuelle de Shakespeare ***")
    else:
        print("\n  Résultat: tendance positive mais modérée.")


if __name__ == "__main__":
    run_phase13()
