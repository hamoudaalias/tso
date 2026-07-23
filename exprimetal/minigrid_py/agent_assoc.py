"""
TSO — Mémoire associative (k-NN cosinus) pour le matching one-shot.
- LIF: contexte d'action / navigation (SymbolicEncoder)
- AssociativeMemory: stocke la cible (objet en one-hot type+color+state)
- Décision: l'objet visible avec la meilleure similarité cosinus → cible
"""
import numpy as np
from tso_pyo3 import LIFState, AssociativeMemory, AttractorField
from symbolic_encoder import SymbolicEncoder


TYPE_ONEHOT = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}


class AssocAgent:
    def __init__(self, epsilon=0.7):
        self.enc = SymbolicEncoder()
        self.lif = LIFState(self.enc.dim, 0.85)
        self.mem = AssociativeMemory()
        self.field = AttractorField(7, 2, 2, 0.1)
        self.q = {}
        self.visit = {}
        self.epsilon = epsilon
        self.lr = 0.2
        self.gamma = 0.95
        self.explore_bonus = 0.5
        self.n_actions = 7
        self._step = 0
        self._target_stored = False
        self._correct_type = None
        self.prev = None
        self.prev_a = None

    def reset(self):
        self.lif = LIFState(self.enc.dim, 0.85)
        self._step = 0
        self._target_stored = False
        self._correct_type = None
        self.prev = None
        self.prev_a = None

    def _obj_vec(self, t, c, s):
        oh = [0.0] * 5
        idx = TYPE_ONEHOT.get(t, -1)
        if idx >= 0:
            oh[idx] = 1.0
        return np.array(oh + [c / 6.0, s / 3.0], dtype=np.float64)

    def _extract_objects(self, obs):
        img = obs["image"]
        objs = []
        for y in range(7):
            for x in range(7):
                t, c, s = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if t not in (5, 6):
                    continue
                dx, dy = (x - 3) / 3.5, (y - 3) / 3.5
                objs.append((self._obj_vec(t, c, s), dx, dy, t, c))
        objs.sort(key=lambda o: abs(o[1]) + abs(o[2]))
        return objs

    def _store_target(self, obs):
        objs = self._extract_objects(obs)
        if objs:
            ov, _, _, t, c = objs[0]
            self.mem.store(ov.tolist(), t)
            self._target_stored = True
            return True
        return False

    def _best_match(self, objs):
        best_sim, best_obj = -1.0, None
        for ov, dx, dy, t, c in objs:
            result = self.mem.recall_with_sim(ov.tolist())
            if result is not None:
                data, sim = result
                if sim > best_sim:
                    best_sim, best_obj = sim, (ov, dx, dy, t, c)
        return best_sim, best_obj

    def featurize(self, obs):
        nav = self.enc.encode(obs)
        self.lif.step(nav.tolist(), False)
        return np.array(self.lif.get_state())

    def act(self, obs):
        self._step += 1
        objs = self._extract_objects(obs)

        if not self._target_stored and objs:
            self._store_target(obs)

        # État de navigation
        nav_state = self.featurize(obs)

        # Meilleur match via AssociativeMemory
        best_sim, best_obj = self._best_match(objs)
        sim_val = best_sim if best_sim >= 0 else 0.0

        # Assembler l'état complet
        state = np.concatenate([nav_state, [sim_val, float(best_obj is not None)]])
        self.prev = state.copy()

        # Toggle automatique si objet correspondant juste devant
        if best_sim is not None and best_sim > 0.95 and best_obj is not None:
            dx = best_obj[1]
            if abs(dx) < 0.3:
                self.prev_a = 5
                return 5

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
        sim = state[-2] if len(state) >= 7 else 0.0
        intrinsic_r = r + bonus + 0.05 * max(sim, 0)

        mx = max((self.q.get((nk, na), 0.0) for na in range(self.n_actions)), default=0.0) if not done else 0.0
        oq = self.q.get((k, a), 0.0)
        self.q[(k, a)] = oq + self.lr * (intrinsic_r + self.gamma * mx - oq)

        if r > 0 and self._target_stored:
            self.field.add_class(state[:7].tolist())

    def decay(self, factor=0.995):
        self.epsilon = max(self.epsilon * factor, 0.03)
