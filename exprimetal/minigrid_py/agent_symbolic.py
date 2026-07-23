"""
TSO Agent with symbolic MiniGrid encoder + count-based exploration + Graph.
"""
import numpy as np
from tso_pyo3 import (
    LIFState, AttractorField, EpisodicMemory,
    ContextBuffer, Graph,
)
from symbolic_encoder import SymbolicEncoder


class TSOSymbolicAgent:
    def __init__(self, epsilon=0.7, alpha=0.85):
        self.enc = SymbolicEncoder()
        dim = self.enc.dim
        self.lif = LIFState(dim, alpha)
        self.field = AttractorField(dim, 2, 4, 0.05)
        self.episodic = EpisodicMemory(200)
        self.ctx = ContextBuffer(50)
        self.graph = Graph()
        self.q = {}
        self.visit = {}
        self.epsilon = epsilon
        self.lr = 0.2
        self.gamma = 0.95
        self.explore_bonus = 0.5
        self.n_actions = 7
        self.prev = None
        self.prev_a = None
        self.prev_node = None
        self._step = 0

    def reset(self):
        self.lif = LIFState(self.enc.dim, 0.85)
        self.ctx = ContextBuffer(50)
        self.prev = None
        self.prev_a = None
        self.prev_node = None
        self._step = 0

    def featurize(self, obs):
        raw = self.enc.encode(obs)
        self.lif.step(raw.tolist(), False)
        return np.array(self.lif.get_state())

    def act(self, vec):
        self.prev = vec.copy()
        self._step += 1
        nid = self.graph.add_node(vec.tolist())
        self.prev_node = nid

        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
            self.prev_a = a
            return a

        k = self._key(vec)
        scores = []
        for a in range(self.n_actions):
            n = self.visit.get((k, a), 0)
            bonus = self.explore_bonus / np.sqrt(max(n, 1))
            q = self.q.get((k, a), 0.0) + bonus
            scores.append((q, a))
        best_a = max(scores, key=lambda x: x[0])[1]
        self.prev_a = best_a
        return best_a

    def learn(self, vec, a, r, nv, done):
        k, nk = self._key(vec), self._key(nv)
        self.visit[(k, a)] = self.visit.get((k, a), 0) + 1
        bonus = self.explore_bonus / np.sqrt(self.visit[(k, a)])
        intrinsic_r = r + bonus

        mnq = max(
            (self.q.get((nk, na), 0.0) for na in range(self.n_actions)),
            default=0.0,
        ) if not done else 0.0

        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (intrinsic_r + self.gamma * mnq - oq)

        if r > 0:
            if self.field.n_classes() < 1:
                self.field.add_class(vec.tolist())
            self.field.add_prototype(vec.tolist(), 0)

        if self._step > 1 and abs(self._step - 0) > 0:
            nn = self.graph.add_node(nv.tolist())
            w = 2 if r > 0 else (-1 if r < 0 else 1)
            self.graph.add_edge(self.prev_node, nn, w)

        self.ctx.push(abs(hash(k)) % 10000)

    def store_episode(self, vecs):
        ids = [abs(hash(tuple(np.round(v[:4] * 10).astype(int)))) % 10000 for v in vecs]
        self.episodic.store(ids)

    def decay(self, factor=0.995):
        self.epsilon = max(self.epsilon * factor, 0.05)

    def _key(self, v):
        return tuple(np.round(v[:4] * 10).astype(int))
