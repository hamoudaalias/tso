"""
TSO Agent v3 — version allégée.
Composants : V1 cortex → DualLIF → Cerebellum + AssociativeMemory + ActionMotor.
"""
import math
import numpy as np
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory, Cerebellum
from retina_v3 import encode_v1


def make_action_embeddings(dim):
    vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float64)
        np.random.seed(i * 7 + 13)
        idxs = np.random.choice(dim, min(6, dim), replace=False)
        v[idxs] = 0.2
        vecs.append(v.tolist())
    return vecs


class TSOAgentV3:
    def __init__(self, v1_field, dim=64, n_actions=15, lr=0.05, noise_std=0.3):
        self.v1 = v1_field
        self.dim = dim
        self.n_actions = n_actions
        self.lif = DualLIFState(dim, 0.95, 0.5)
        self.cb = Cerebellum(dim, n_actions, lr, noise_std)
        self.mem = AssociativeMemory()
        self.motor = ActionMotor(0.7)
        self.action_vecs = make_action_embeddings(dim)

    def act(self, rgb, explore=True):
        vec = encode_v1(self.v1, rgb)
        self.lif.step(vec, False)
        concept = self.lif.get_slow_state()
        rc = self.mem.recall_with_sim(concept)
        bonuses = [0.0] * self.n_actions
        if rc is not None and rc[1] > 0.35:
            bonuses[rc[0]] = 0.5
        cb_a = self.cb.forward(concept)
        bonuses[cb_a] += 0.3
        if explore and np.random.random() < 0.1:
            return np.random.randint(self.n_actions)
        action, _ = self.motor.select_with_bonus(self.lif, self.action_vecs, bonuses)
        return action

    def learn(self, concept, action, reward):
        learn_signal = reward + 0.1 * (reward - 0.0)
        self.cb.learn(concept, action, learn_signal)
        if reward > 0:
            self.mem.store(concept, action)

    def reset(self):
        self.lif = DualLIFState(self.dim, 0.95, 0.5)
