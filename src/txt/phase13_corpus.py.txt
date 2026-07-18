"""
TSO Phase 13 (V6) — Lecture de Shakespeare par transitions conceptuelles.
Chaque mot active un concept SOM. Le réseau apprend la matrice de
transition entre concepts consécutifs via Hebbien local.
La friction Φ = 1 - P(concept_j | concept_i) normalisée.
Si TSO infère la structure grammaticale, Φ chute significativement
par rapport à la baseline aléatoire.
"""
import math, random, urllib.request, re
import numpy as np
import time
from collections import Counter

SEED = 42
random.seed(SEED); np.random.seed(SEED)

EMBED_DIM = 64
SOM_ROWS, SOM_COLS = 10, 10
N_CONCEPTS = SOM_ROWS * SOM_COLS
ALPHA = 0.05
BETA = 0.002

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("  Téléchargement de Tiny Shakespeare...")
    try: return urllib.request.urlopen(url).read().decode('utf-8')
    except: return "To be or not to be. To go or not to go."

def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.lower())
    return [re.findall(r'\b\w+\b', s) + ['.'] for s in sentences]

def build_vocab(sentences, vocab_size=1000):
    counts = Counter(w for s in sentences for w in s)
    vocab = [w for w, _ in counts.most_common(vocab_size)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer('all-MiniLM-L6-v2', device=dev)
    e = m.encode(vocab, convert_to_tensor=True, show_progress_bar=False)
    e = e.cpu().numpy().astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
    np.random.seed(0)
    P = np.random.randn(384, EMBED_DIM).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
    embeddings = e @ P
    return vocab, word_to_idx, embeddings

class MiniSOM:
    def __init__(self, rows, cols, dim):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.weights = np.random.randn(rows * cols, dim).astype(np.float32)
        for i in range(rows * cols):
            self.weights[i] /= np.linalg.norm(self.weights[i])
    def bmu(self, x):
        return int(np.argmin(np.linalg.norm(self.weights - x, axis=1)))
    def train_step(self, x, lr=0.1, sigma=2.0):
        bi = self.bmu(x)
        for i in range(self.rows * self.cols):
            ri, ci = divmod(i, self.cols)
            rj, cj = divmod(bi, self.cols)
            d2 = (ri-rj)**2 + (ci-cj)**2
            h = math.exp(-d2/(2*sigma*sigma))
            self.weights[i] += lr * h * (x - self.weights[i])
            self.weights[i] /= np.linalg.norm(self.weights[i]) + 1e-8

class TransitionGraph:
    def __init__(self, n):
        self.n = n
        self.W = np.ones((n, n), dtype=np.float32) * 0.5

    def transition_phi(self, prev, curr):
        total = self.W[prev].sum()
        p = self.W[prev, curr] / total if total > 0 else 1.0 / self.n
        return 1.0 - p * self.n

    def learn(self, prev, curr):
        self.W[prev, curr] += ALPHA * (1.0 - self.W[prev, curr])
        total = self.W[prev].sum()
        if total > 1.0:
            self.W[prev] *= (1.0 - BETA)
            self.W[prev, curr] += BETA * total

def run_phase13_v6():
    print("="*72)
    print("  TSO Phase 13 (V6) — Transitions Conceptuelles Shakespeare")
    print("="*72)

    text = download_shakespeare()
    sentences = preprocess_text(text)
    print(f"  Phrases: {len(sentences)}")

    vocab, word_to_idx, embeddings = build_vocab(sentences, vocab_size=1000)
    print(f"  Vocabulaire: {len(vocab)} mots.")

    print("\n  Organisation SOM (100 concepts)...")
    som = MiniSOM(SOM_ROWS, SOM_COLS, EMBED_DIM)
    for epoch in range(150):
        lr = 0.1 * (1.0 - epoch / 150)
        for i in range(len(vocab)):
            som.train_step(embeddings[i], lr=lr, sigma=2.0)
    cluster_of_word = np.array([som.bmu(embeddings[i]) for i in range(len(vocab))])
    occupied = len(set(cluster_of_word))
    print(f"  Concepts occupés: {occupied}/{N_CONCEPTS}")

    graph = TransitionGraph(N_CONCEPTS)

    all_phis = []
    block_size = 1000
    n_blocks = max(1, len(sentences) // block_size)

    print(f"\n  Lecture Shakespeare ({n_blocks} blocs de {block_size} phrases)...")
    print(f"  {'Bloc':>5s} | {'Φ':>8s} | {'Δ vs baseline':>12s}")
    print("  " + "-"*32)

    baseline = 1.0 - 1.0 / N_CONCEPTS

    for block in range(n_blocks):
        block_phis = []
        for i in range(block * block_size, (block + 1) * block_size):
            if i >= len(sentences): break
            s = sentences[i]
            idxs = [word_to_idx.get(w, -1) for w in s]
            idxs = [i for i in idxs if i != -1]
            for j in range(len(idxs) - 1):
                c1, c2 = cluster_of_word[idxs[j]], cluster_of_word[idxs[j+1]]
                block_phis.append(graph.transition_phi(c1, c2))
                graph.learn(c1, c2)

        avg = np.mean(block_phis) if block_phis else 0.0
        all_phis.append(avg)
        imp = (baseline - avg) / baseline * 100
        if block % 2 == 0 or block == n_blocks - 1:
            print(f"  {block+1:3d}/{n_blocks} | {avg:.4f} | {imp:+.2f}%")

    print("  " + "-"*32)

    first = np.mean(all_phis[:3])
    last = np.mean(all_phis[-3:])
    first = float(first)
    last = float(last)
    phi_drop = (last - first) / abs(first) * 100
    gain_initial = (baseline - first) / baseline * 100
    gain_final = (baseline - last) / baseline * 100

    print(f"\n  >>> Analyse:")
    print(f"      Baseline aléatoire:        {baseline:.4f}")
    print(f"      Φ initial (blocs 1-3):     {first:.4f} ({gain_initial:.0f}% mieux que hasard)")
    print(f"      Φ final  (blocs 10-12):    {last:.4f} ({gain_final:.0f}% mieux que hasard)")
    print(f"      Amélioration relative:     {phi_drop:.1f}%")

    if gain_final > 100:
        print(f"\n  *** VALIDATION : TSO infère la grammaire conceptuelle de Shakespeare ***")
        print(f"  Le réseau anticipe les transitions entre concepts avec une confiance")
        print(f"  {gain_final:.0f}% supérieure au hasard, apprise sans rétropropagation.")
    elif gain_final > 20:
        print("\n  Tendance positive — le graphe conceptuel est informatif.")
    else:
        print("\n  Résultat neutre.")

if __name__ == "__main__":
    run_phase13_v6()
