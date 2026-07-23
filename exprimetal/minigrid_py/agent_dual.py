"""
TSO Agent for MemoryS7 — DualLIF (slow/fast) for one-shot target memory.
- Slow LIF (α=0.98) maintains target prototype across the hallway.
- Fast LIF (α=0.5) tracks immediate observations.
- Alignment = weighted combo of slow + fast drives action.
- LVQ1 stores the target prototype one-shot at step 0.
"""
import numpy as np
from tso_pyo3 import (
    DualLIFState, AttractorField,
    EpisodicMemory, ContextBuffer, Graph,
)
from symbolic_encoder import SymbolicEncoder


class TSOAgent:
    def __init__(self, epsilon=0.7):
        self.enc = SymbolicEncoder()
        self.dim = self.enc.dim
        # DualLIF: slow (long-term target), fast (immediate)
        self.lif = DualLIFState(self.dim, 0.98, 0.5)
        self.field = AttractorField(self.dim, 2, 4, 0.06)
        self.episodic = EpisodicMemory(100)
        self.ctx = ContextBuffer(50)
        self.graph = Graph()
        self.q = {}
        self.visit = {}
        self.epsilon = epsilon
        self.lr = 0.15
        self.gamma = 0.95
        self.explore_bonus = 0.5
        self.n_actions = 7
        self.prev = None
        self.prev_a = None
        self.prev_node = None
        self._step = 0
        self._target_stored = False

    def reset(self):
        self.lif = DualLIFState(self.dim, 0.98, 0.5)
        self.ctx = ContextBuffer(50)
        self.prev = None
        self.prev_a = None
        self.prev_node = None
        self._step = 0
        self._target_stored = False

    def featurize(self, obs):
        raw = self.enc.encode(obs)
        self.lif.step(raw.tolist(), False)
        # slow (α=0.98) maintains target memory, fast tracks current view
        slow = np.array(self.lif.get_slow_state())
        fast = np.array(self.lif.get_fast_state())
        # state = combo — slow-dominated for one-shot retention
        return 0.7 * slow + 0.3 * fast

    def maybe_store_target(self, vec):
        if not self._target_stored:
            if self.field.n_classes() < 1:
                self.field.add_class(vec.tolist())
            else:
                self.field.add_prototype(vec.tolist(), 0)
            self._target_stored = True

    def act(self, vec):
        self.prev = vec.copy()
        self._step += 1
        self.maybe_store_target(vec)
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
        self.prev_a = max(scores, key=lambda x: x[0])[1]
        return self.prev_a

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
            self.field.add_prototype(vec.tolist(), 0)

        if self._step > 1:
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
