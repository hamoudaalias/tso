"""
TSO — Alignement Objet-par-Objet (Inverse Motor).
La cible est stockée comme prototype one-shot dans l'attracteur LVQ1.
Chaque objet visible est comparé (cosine) contre le prototype.
L'agent navigue vers le meilleur alignement.
"""
import numpy as np
from tso_pyo3 import AttractorField


OBJECT_KEYS = {4: "door", 5: "key", 6: "ball", 8: "goal"}


class AlignAgent:
    def __init__(self, epsilon=0.5):
        self.field = AttractorField(3, 2, 2, 0.1)
        self.q = {}
        self.visit = {}
        self.epsilon = epsilon
        self.lr = 0.15
        self.gamma = 0.9
        self.explore_bonus = 0.5
        self.n_actions = 7
        self._target_proto = None
        self._step = 0
        self.prev = None
        self.prev_a = None

    def reset(self):
        self._target_proto = None
        self._step = 0
        self.prev = None
        self.prev_a = None

    def _obj_vec(self, t, c, s):
        return np.array([t / 10.0, c / 6.0, s / 3.0], dtype=np.float64)

    def _extract_objects(self, obs):
        img = obs["image"]
        objs = []
        for y in range(7):
            for x in range(7):
                t, c, s = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if t not in (4, 5, 6, 8):
                    continue
                dx, dy = (x - 3) / 3.5, (y - 3) / 3.5
                objs.append((t, c, s, dx, dy))
        objs.sort(key=lambda o: abs(o[3]) + abs(o[4]))
        return objs

    def _cosine(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(a @ b / (na * nb))

    def act(self, obs):
        self._step += 1
        objs = self._extract_objects(obs)

        # Phase 1: stocker le premier objet comme cible one-shot
        if self._target_proto is None and objs:
            t, c, s, _, _ = objs[0]
            self._target_proto = self._obj_vec(t, c, s)

        # Alignement: chaque objet vs cible
        best_align, best_idx = -1.0, -1
        for i, (t, c, s, dx, dy) in enumerate(objs):
            ov = self._obj_vec(t, c, s)
            align = self._cosine(ov, self._target_proto) if self._target_proto is not None else 0.0
            if align > best_align:
                best_align, best_idx = align, i

        dx, dy = (0.0, 0.0)
        if best_idx >= 0:
            _, _, _, dx, dy = objs[best_idx]
            if best_align > 0.85 and abs(dx) < 0.3:
                a = 5  # toggle/pickup
                self.prev = np.array([best_align, dx, dy, 5.0])
                self.prev_a = a
                return a

        state = np.array([best_align, dx, dy, float(self._target_proto is not None)])
        self.prev = state.copy()

        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
            self.prev_a = a
            return a

        k = tuple(np.round(state * 10).astype(int))
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
        k = tuple(np.round(state * 10).astype(int))
        nk = tuple(np.round(ns * 10).astype(int))
        self.visit[(k, a)] = self.visit.get((k, a), 0) + 1
        bonus = self.explore_bonus / np.sqrt(self.visit[(k, a)])
        intrinsic_r = r + bonus + 0.1 * max(state[0], 0)

        mx = max((self.q.get((nk, na), 0.0) for na in range(self.n_actions)), default=0.0) if not done else 0.0
        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (intrinsic_r + self.gamma * mx - oq)

        # LVQ1: stocker les états réussis
        if r > 0 and self._target_proto is not None:
            self.field.add_class(self._target_proto.tolist())

    def decay(self, factor=0.995):
        self.epsilon = max(self.epsilon * factor, 0.03)
