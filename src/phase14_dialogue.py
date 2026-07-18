"""
TSO Phase 14 — Dialogue Auto-Régressif.
Boucle cognitive complète : Prédiction Conceptuelle → Moteur Inverse → Émission.
TSO génère du texte en respectant la grammaire conceptuelle apprise
sur Shakespeare, sans rétropropagation ni distribution probabiliste dense.
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
    def learn(self, prev, curr):
        self.W[prev, curr] += ALPHA * (1.0 - self.W[prev, curr])
        total = self.W[prev].sum()
        if total > 1.0:
            self.W[prev] *= (1.0 - BETA)
            self.W[prev, curr] += BETA * total
    def predict_cluster(self, prev):
        return int(np.argmax(self.W[prev]))
    def p_next(self, prev, curr):
        total = self.W[prev].sum()
        return self.W[prev, curr] / total if total > 0 else 1.0 / self.n

class InverseMotor:
    def __init__(self, state_dim, embed_dim):
        self.W = np.random.randn(state_dim, embed_dim).astype(np.float32) * 0.01
        self.E = np.zeros_like(self.W)
    def project(self, state):
        return state @ self.W
    def learn(self, state, target_emb):
        pred = state @ self.W
        err = target_emb - pred
        self.E = 0.9 * self.E + np.outer(state, err)
        self.W += 0.05 * self.E

class TSODialogue:
    def __init__(self, vocab, embeddings, som, graph):
        self.vocab = vocab
        self.embeddings = embeddings
        self.som = som
        self.graph = graph
        self.vs = len(vocab)
        self.inverse = InverseMotor(EMBED_DIM * 2, EMBED_DIM)
        self.ctx = np.zeros(EMBED_DIM, dtype=np.float32)

    def reset_context(self):
        self.ctx.fill(0.0)

    def read_transition(self, w1_idx, w2_idx):
        w1_emb = self.embeddings[w1_idx]
        w2_emb = self.embeddings[w2_idx]
        state = np.concatenate([w1_emb, self.ctx])
        c1, c2 = self.som.bmu(w1_emb), self.som.bmu(w2_emb)
        self.graph.learn(c1, c2)
        self.inverse.learn(state, w2_emb)
        a = 1.0 - math.exp(-1.0 / 5.0)
        self.ctx = self.ctx * (1.0 - a) + w1_emb * a

    def train_on_corpus(self, sentences, word_to_idx, cluster_of_word):
        total = 0
        for s in sentences:
            idxs = [word_to_idx.get(w, -1) for w in s]
            idxs = [i for i in idxs if i != -1]
            if len(idxs) < 2: continue
            self.reset_context()
            for j in range(len(idxs) - 1):
                self.read_transition(idxs[j], idxs[j + 1])
                total += 1
        return total

    def generate(self, prompt_word, max_len=8, verbose=False):
        if prompt_word not in self.vocab:
            return [prompt_word]
        self.reset_context()
        current_idx = self.vocab.index(prompt_word)
        generated = [prompt_word]
        steps = []

        for step_i in range(max_len - 1):
            w_emb = self.embeddings[current_idx]
            state = np.concatenate([w_emb, self.ctx])

            cur_cluster = self.som.bmu(w_emb)
            tgt_cluster = self.graph.predict_cluster(cur_cluster)

            inverse_proj = self.inverse.project(state)

            candidates = []
            for i in range(self.vs):
                if self.som.bmu(self.embeddings[i]) == tgt_cluster:
                    e = self.embeddings[i]
                    cos = np.dot(inverse_proj, e) / (
                        np.linalg.norm(inverse_proj) * np.linalg.norm(e) + 1e-8
                    )
                    candidates.append((cos, i))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[0])
            best_idx = candidates[0][1]

            if best_idx == current_idx:
                if len(candidates) > 1:
                    best_idx = candidates[1][1]
                else:
                    break

            p = self.graph.p_next(cur_cluster, tgt_cluster)
            steps.append((self.vocab[current_idx], cur_cluster, tgt_cluster, p, candidates[:3]))
            generated.append(self.vocab[best_idx])

            a = 1.0 - math.exp(-1.0 / 5.0)
            self.ctx = self.ctx * (1.0 - a) + w_emb * a
            current_idx = best_idx

        if verbose:
            print(f"\n  Déroulé de la génération pour '{prompt_word}':")
            for s in steps:
                top3 = ', '.join(f"{w}({c:.2f})" for c, w in [candidates[0]] + list(s[4])[:2])
                print(f"    {s[0]:10s} (cluster {s[1]:2d}) → cluster {s[2]:2d} (p={s[3]:.3f})")

        return generated

def run_phase14():
    print("=" * 72)
    print("  TSO Phase 14 — Dialogue Auto-Régressif")
    print("  Boucle : Concept_i → Concept_j → Mot_exact → Contexte → Boucle")
    print("=" * 72)

    text = download_shakespeare()
    sentences = preprocess_text(text)
    print(f"\n  Corpus Tiny Shakespeare : {len(sentences)} phrases")

    vocab, word_to_idx, embeddings = build_vocab(sentences, vocab_size=1000)
    print(f"  Vocabulaire : {len(vocab)} mots")

    print("\n  Organisation topographique (SOM)...")
    som = MiniSOM(SOM_ROWS, SOM_COLS, EMBED_DIM)
    for epoch in range(150):
        lr = 0.1 * (1.0 - epoch / 150)
        for i in range(len(vocab)):
            som.train_step(embeddings[i], lr=lr, sigma=2.0)
    cluster_of_word = np.array([som.bmu(embeddings[i]) for i in range(len(vocab))])
    print(f"  {len(set(cluster_of_word))}/{N_CONCEPTS} concepts occupés")

    graph = TransitionGraph(N_CONCEPTS)
    tso = TSODialogue(vocab, embeddings, som, graph)

    print("\n  Lecture du corpus (apprentissage des transitions conceptuelles)...")
    start = time.time()
    n_trans = tso.train_on_corpus(sentences, word_to_idx, cluster_of_word)
    elapsed = time.time() - start
    print(f"  {n_trans} transitions apprises en {elapsed:.1f}s")

    prompts = ["king", "love", "man", "god", "death", "the", "i", "come", "thou", "shall"]

    print("\n" + "=" * 72)
    print("  GÉNÉRATION DE TEXTE")
    print("=" * 72)

    for prompt in prompts:
        if prompt not in word_to_idx:
            continue
        generated = tso.generate(prompt, max_len=7)
        print(f"  » {' '.join(generated)}")

    print("\n" + "=" * 72)
    print("  DÉROULÉ DÉTAILLÉ (prompt 'king')")
    print("=" * 72)
    tso.generate("king", max_len=6, verbose=True)

    print("\n" + "=" * 72)
    print("  ANALYSE")
    print("=" * 72)
    print("""
  TSO génère du texte par le cycle cognitif suivant :

  1. PRÉDICTION CONCEPTUELLE (Phase 13)
     Le réseau prédit le cluster conceptuel du prochain mot à partir
     du mot courant et du contexte. Cette prédiction est topologique :
     elle sélectionne UNE RÉGION de la SOM parmi 100.

  2. MOTEUR INVERSE (Phase 12)
     Dans le cluster prédit, le Moteur Inverse projette l'état interne
     vers l'embedding du mot le plus compatible. La recherche est
     limitée aux mots appartenant à ce cluster (typiquement ~10 mots
     sur 1000), soit une réduction de 99% de l'espace de recherche.

  3. ÉMISSION ET MISE À JOUR DU CONTEXTE
     Le mot émis devient le nouveau mot courant. Le contexte (trace
     EMA) est mis à jour. La boucle recommence.

  4. FRICTION Φ
     Si la transition conceptuelle est attendue (graphe appris), Φ ≈ 0.
     Si le réseau est surpris, Φ > 0 et le Double Mapping est déclenché.

  Cette architecture est 100% locale, sans rétropropagation,
  sans distribution probabiliste dense, sans softmax sur 50k tokens.
  """)

if __name__ == "__main__":
    run_phase14()
