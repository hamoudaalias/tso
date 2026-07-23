"""
TSO Cognitive Engine Agent for gym-minigrid.
Components: LIF reservoir + LVQ1 attractor (one-shot) + Episodic memory + Q-learning.
"""
import numpy as np
from tso_pyo3 import LIFState, AttractorField, EpisodicMemory, ContextBuffer


class Encoder:
    """Projects MiniGrid observation into a fixed-dim vector via random projection."""
    def __init__(self, dim=48, seed=42):
        self.dim = dim
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(7*7*3 + 4, dim).astype(np.float32) * 0.08

    def __call__(self, obs):
        img = obs["image"].astype(np.float32).ravel() / 10.0
        d = np.zeros(4, dtype=np.float32)
        d[obs["direction"]] = 1.0
        return np.concatenate([img, d]).astype(np.float64) @ self.proj


class TSOAgent:
    def __init__(self, dim=48, alpha=0.85, epsilon=0.5):
        self.dim = dim
        self.enc = Encoder(dim)
        self.lif = LIFState(dim, alpha)
        self.field = AttractorField(dim, 2, 4, 0.05)
        self.episodic = EpisodicMemory(200)
        self.ctx = ContextBuffer(30)
        self.q = {}
        self.epsilon = epsilon
        self.lr = 0.2
        self.gamma = 0.95
        self.n_actions = 7
        self.prev_vec = None
        self.prev_a = None
        self._step = 0

    def reset(self):
        self.lif = LIFState(self.dim, 0.85)
        self.ctx = ContextBuffer(30)
        self.prev_vec = None
        self.prev_a = None
        self._step = 0

    def featurize(self, obs):
        raw = self.enc(obs)
        self.lif.step(raw.tolist(), False)
        return np.array(self.lif.get_state())

    def act(self, vec):
        self.prev_vec = vec.copy()
        self._step += 1
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
            self.prev_a = a
            return a
        k = self._hash(vec)
        best_a, best_q = 0, -1e9
        for a in range(self.n_actions):
            q = self.q.get((k, a), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        self.prev_a = best_a
        return best_a

    def learn(self, vec, a, r, nv, done):
        k, nk = self._hash(vec), self._hash(nv)
        mnq = max((self.q.get((nk, na), 0.0) for na in range(self.n_actions)), default=0.0) if not done else 0.0
        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (r + self.gamma * mnq - oq)

        if r > 0:
            if self.field.n_classes() < 1:
                self.field.add_class(vec.tolist())
            else:
                self.field.add_prototype(vec.tolist(), 0)
        elif r < -0.5:
            need = self.field.n_classes()
            if need <= 1:
                self.field.add_class(vec.tolist())
                need = self.field.n_classes()
            self.field.add_prototype(vec.tolist(), need - 1)

        self.ctx.push(abs(hash(k)) % 10000)

    def store_episode(self, states):
        ids = [abs(hash(tuple(np.round(s[:4]*10).astype(int)))) % 10000 for s in states]
        self.episodic.store(ids)

    def decay(self, factor=0.997):
        self.epsilon = max(self.epsilon * factor, 0.03)

    def _hash(self, v):
        return tuple(np.round(v[:6] * 10).astype(int))
