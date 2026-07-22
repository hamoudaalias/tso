"""
TSO Agent for Snail Jumper — pure Python port of tso-engine logic.
LIF reservoir + LVQ1 (cosine distance prototypes) — no backprop.
"""
import numpy as np
import math

class LIFState:
    def __init__(self, dim, alpha=0.8):
        self.state = np.zeros(dim)
        self.alpha = alpha

    def step(self, x, negate=False):
        e = -x if negate else x
        self.state = self.alpha * self.state + (1 - self.alpha) * e

class AttractorField:
    def __init__(self, dim, n_classes=2, k=4, lr=0.05):
        self.prototypes = []
        for _ in range(n_classes):
            class_ps = []
            for _ in range(k):
                v = np.random.uniform(-1, 1, dim)
                v = v / max(np.linalg.norm(v), 1e-12)
                class_ps.append(v)
            self.prototypes.append(class_ps)
        self.lr = lr

    def cosine_dist(self, a, b):
        dot = np.dot(a, b)
        na = max(np.dot(a, a), 1e-12) ** 0.5
        nb = max(np.dot(b, b), 1e-12) ** 0.5
        return 1.0 - (dot / (na * nb))

    def predict(self, state):
        best_class = 0
        best_dist = float('inf')
        for c, protos in enumerate(self.prototypes):
            for p in protos:
                d = self.cosine_dist(state, p)
                if d < best_dist:
                    best_dist = d
                    best_class = c
        return best_class, best_dist

    def add_class(self, example):
        v = example / max(np.linalg.norm(example), 1e-12)
        self.prototypes.append([v])
        return len(self.prototypes) - 1

    def add_prototype(self, example, class_idx):
        v = example / max(np.linalg.norm(example), 1e-12)
        while len(self.prototypes) <= class_idx:
            self.prototypes.append([])
        self.prototypes[class_idx].append(v)

    def n_classes(self):
        return len(self.prototypes)

    def clone(self):
        import copy
        return copy.deepcopy(self)

class TSOAgent:
    def __init__(self, embed_dim=8, alpha=0.8):
        self.lif = LIFState(embed_dim, alpha)
        self.field = AttractorField(embed_dim)
        self.fitness = 0
        self.embed_dim = embed_dim

    def encode(self, game_state):
        v = np.zeros(self.embed_dim)
        v[0] = game_state['player_x'] / game_state['screen_w']
        if game_state['obstacles']:
            nearest = game_state['obstacles'][0]
            v[1] = nearest['x'] / game_state['screen_w']
            v[2] = nearest['y'] / game_state['screen_h']
            v[3] = (nearest['x'] - game_state['player_x']) / game_state['screen_w']
        v[4] = len(game_state['obstacles']) / 10.0
        for i, obs in enumerate(game_state['obstacles'][:3]):
            v[5 + i] = min(obs['y'] / game_state['screen_h'], 1.0)
        return v

    def decide(self, game_state):
        state = self.encode(game_state)
        self.lif.step(state)
        cls, _ = self.field.predict(self.lif.state)
        return cls == 0  # class 0 = left, class 1 = right

    def learn(self, game_state, alive):
        state = self.encode(game_state)
        if alive:
            self.field.add_prototype(state, 0)
        else:
            if self.field.n_classes() <= 1:
                self.field.add_class(state)
            self.field.add_prototype(state, 1)
