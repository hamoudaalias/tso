"""
TSO Agent for MiniGrid Memory-S7.
Specialized for one-shot object matching.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tso_pyo3 import LIFState, AttractorField, EpisodicMemory, ContextBuffer


STATE_DIM = 48
OBJ_IDS = {0: "unseen", 1: "empty", 2: "wall", 3: "floor",
           4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 10: "agent"}
COLORS = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}


class Encoder:
    def __init__(self, dim=STATE_DIM):
        self.dim = dim
        rng = np.random.RandomState(42)
        self.proj = rng.randn(7 * 7 * 3 + 4, dim).astype(np.float32) * 0.1

    def encode(self, obs):
        img = obs["image"].astype(np.float32).flatten() / 10.0
        d = np.zeros(4, dtype=np.float32)
        d[obs["direction"]] = 1.0
        x = np.concatenate([img, d]).astype(np.float64)
        return x @ self.proj


class TSOAgent:
    def __init__(self, dim=STATE_DIM):
        self.enc = Encoder(dim)
        self.lif = LIFState(dim, 0.85)
        self.field = AttractorField(dim, 2, 3, 0.04)
        self.episodic = EpisodicMemory(100)
        self.ctx = ContextBuffer(20)
        self.q = {}
        self.epsilon = 0.5
        self.lr = 0.2
        self.gamma = 0.95
        self.prev = None
        self.prev_a = None
        self.n_actions = 7
        self.target_proto = None
        self.step = 0

    def reset(self):
        self.lif = LIFState(self.dim, 0.85)
        self.ctx = ContextBuffer(20)
        self.prev = None
        self.prev_a = None
        self.target_proto = None
        self.step = 0

    def _feat(self, obs):
        raw = self.enc.encode(obs)
        self.lif.step(raw.tolist(), False)
        return np.array(self.lif.get_state())

    def act(self, s, env):
        self.prev = s.copy()
        self.step += 1
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
            self.prev_a = a
            return int(a)
        k = self._key(s)
        best_a, best_q = 0, -1e9
        for a in range(self.n_actions):
            q = self.q.get((k, a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        self.prev_a = best_a
        return best_a

    def learn(self, s, a, r, ns, done):
        k, nk = self._key(s), self._key(ns)
        mnq = max((self.q.get((nk, na), 0.0) for na in range(self.n_actions)), default=0.0) if not done else 0.0
        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (r + self.gamma * mnq - oq)

        if r > 0.5:
            c = self.field.add_class(s.tolist()) if self.field.n_classes() < 1 else 0
            if not self.field.n_classes():
                self.field.add_class(s.tolist())
            self.field.add_prototype(s.tolist(), 0)
        elif r < -0.5 and self.field.n_classes() <= 1:
            self.field.add_class(s.tolist())
            if self.field.n_classes() <= 1:
                self.field.add_class(s.tolist())
            c = self.field.n_classes() - 1
            self.field.add_prototype(s.tolist(), c)

    def store_ep(self, seq):
        self.episodic.store([abs(hash(str(s))) % 10000 for s in seq])

    def _key(self, s):
        return tuple(np.round(s[:6] * 10).astype(int))

    def decay(self):
        self.epsilon = max(self.epsilon * 0.997, 0.05)
