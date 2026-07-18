"""
Inverse Motor and Conceptual Decoder.

Bridge between the SNN state space and the word embedding space.
"""
import numpy as np
import math


class InverseMotor:
    """
    Projects SNN state into word embedding space (Phase 7, 12).

    Uses Oja's rule for local, Hebbian learning:
        W += eta * outer(state, target - W @ state)
    """
    def __init__(self, state_dim, embed_dim, eta=0.05):
        self.W = np.random.randn(state_dim, embed_dim).astype(np.float32) * 0.01
        self.E = np.zeros_like(self.W)
        self.eta = eta

    def project(self, state):
        return state @ self.W

    def learn(self, state, target_emb):
        pred = state @ self.W
        err = target_emb - pred
        self.E = 0.9 * self.E + np.outer(state, err)
        self.W += self.eta * self.E

    def select_word(self, state, embeddings, cluster_mask):
        proj = self.project(state)
        norms = np.linalg.norm(proj) * np.linalg.norm(embeddings, axis=1) + 1e-8
        cosines = (embeddings @ proj) / norms
        cosines[~cluster_mask] = -1.0
        best = int(np.argmax(cosines))
        return best, float(cosines[best])


class ConceptualDecoder:
    """
    Predicts the next conceptual cluster from current state (Phase 13).
    """
    def __init__(self, state_dim, embed_dim, n_concepts):
        self.W = np.random.randn(state_dim, embed_dim).astype(np.float32) * 0.1
        self.E = np.zeros_like(self.W)
        self.eta = 0.05

    def predict_cluster(self, state, som):
        proj = state @ self.W
        return som.bmu(proj)

    def learn(self, state, actual_cluster_emb):
        pred = state @ self.W
        err = actual_cluster_emb - pred
        self.E = 0.9 * self.E + np.outer(state, err)
        self.W += self.eta * self.E


class TransitionGraph:
    """
    Learns transition probabilities between conceptual clusters (Phase 13 V6).
    """
    def __init__(self, n_concepts, alpha=0.05, beta=0.002):
        self.n = n_concepts
        self.W = np.ones((n_concepts, n_concepts), dtype=np.float32) * 0.5
        self.alpha = alpha
        self.beta = beta

    def learn(self, prev, curr):
        self.W[prev, curr] += self.alpha * (1.0 - self.W[prev, curr])
        total = self.W[prev].sum()
        if total > 1.0:
            self.W[prev] *= (1.0 - self.beta)
            self.W[prev, curr] += self.beta * total

    def predict_cluster(self, prev):
        return int(np.argmax(self.W[prev]))

    def p_next(self, prev, curr):
        total = self.W[prev].sum()
        return self.W[prev, curr] / total if total > 0 else 1.0 / self.n

    def phi(self, prev, curr):
        p = self.p_next(prev, curr)
        return 1.0 - p * self.n
