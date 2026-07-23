"""
TSO — Agent combiné : SymbolicEncoder (navigation) + Alignement objet (one-shot).
- LVQ1 stocke le prototype de la cible vue au 1er step.
- L'alignement cosine entre chaque objet visible et le prototype guide le matching.
- L'état inclut l'alignement max comme feature supplémentaire.
"""
import numpy as np
from tso_pyo3 import LIFState, AttractorField
from symbolic_encoder import SymbolicEncoder


class ComboAgent:
    def __init__(self, epsilon=0.7):
        self.enc = SymbolicEncoder()
        self.lif = LIFState(self.enc.dim, 0.85)
        self.field = AttractorField(3, 2, 4, 0.08)
        self.q = {}
        self.visit = {}
        self.epsilon = epsilon
        self.lr = 0.2
        self.gamma = 0.95
        self.explore_bonus = 0.5
        self.n_actions = 7
        self._target = None
        self._step = 0
        self.prev = None
        self.prev_a = None

    def reset(self):
        self.lif = LIFState(self.enc.dim, 0.85)
        self._target = None
        self._step = 0
        self.prev = None
        self.prev_a = None

    def _obj_vec(self, t, c, s):
        return np.array([t / 10.0, c / 6.0, s / 3.0], dtype=np.float64)

    def _best_alignment(self, obs):
        img = obs["image"]
        best, best_tcs = -1.0, None
        for y in range(7):
            for x in range(7):
                t, c, s = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if t not in (4, 5, 6, 8):
                    continue
                ov = self._obj_vec(t, c, s)
                if self._target is None:
                    self._target = ov
                    return 1.0, (t, c, s)
                # cosine
                d = ov @ self._target
                n = np.linalg.norm(ov) * np.linalg.norm(self._target)
                align = d / n if n > 1e-12 else 0.0
                if align > best:
                    best = align
                    best_tcs = (t, c, s)
        return best, best_tcs

    def featurize(self, obs, align):
        nav = self.enc.encode(obs)
        self.lif.step(nav.tolist(), False)
        state = np.array(self.lif.get_state())
        return np.concatenate([state, [align]])

    def act(self, obs):
        self._step += 1
        align, _ = self._best_alignment(obs)
        state = self.featurize(obs, align)
        self.prev = state.copy()

        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
            self.prev_a = a
            return a

        k = tuple(np.round(state[:6] * 10).astype(int))
        best_a, best_q = 0, -1e9
        for na in range(self.n_actions):
            n = self.visit.get((k, na), 0)
            bonus = self.explore_bonus / np.sqrt(max(n, 1))
            q = self.q.get((k, na), 0.0) + bonus
            if q > best_q:
                best_q, best_a = q, na
        self.prev_a = best_a
        return best_a

    def learn(self, state, a, r, ns, done):
        k = tuple(np.round(state[:6] * 10).astype(int))
        nk = tuple(np.round(ns[:6] * 10).astype(int))
        self.visit[(k, a)] = self.visit.get((k, a), 0) + 1
        bonus = self.explore_bonus / np.sqrt(self.visit[(k, a)])
        intrinsic_r = r + bonus + 0.05 * max(state[-1], 0)

        mx = max((self.q.get((nk, na), 0.0) for na in range(self.n_actions)), default=0.0) if not done else 0.0
        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (intrinsic_r + self.gamma * mx - oq)

        if r > 0 and self._target is not None:
            self.field.add_class(self._target.tolist())

    def decay(self, factor=0.995):
        self.epsilon = max(self.epsilon * factor, 0.03)
