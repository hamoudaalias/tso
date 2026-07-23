"""
TSO Agent Procgen — utilise WorkingMemory, ActionMotor, EpisodicMemory, etc.
"""
import math
import numpy as np
from tso_pyo3 import (
    WorkingMemory, ActionMotor, DualLIFState,
    Graph, AttractorField, EpisodicMemory, ContextBuffer, AssociativeMemory
)

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def view_to_vector(view, n_types=6):
    """Convertit une vue 5×5 en vecteur one-hot 5×5×n_types."""
    h, w = view.shape
    vec = np.zeros((h * w * n_types,), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            val = int(view[y, x])
            idx = (y * w + x) * n_types + min(val, n_types - 1)
            vec[idx] = 1.0
    return vec


def dir_to_vec(d):
    v = np.zeros(4, dtype=np.float64)
    v[d % 4] = 1.0
    return v


class TSOProcgenAgent:
    """
    Agent TSO adapté aux environnements Procgen.
    Combine WorkingMemory + ActionMotor + EpisodicMemory + Graph.
    """
    def __init__(self, dim=125, n_actions=3, lr=0.01):
        self.dim = dim
        self.n_actions = n_actions
        self.lr = lr

        self.wm = WorkingMemory(dim // 5, 0.95, 0.4)
        self.motor = ActionMotor(0.7)
        self.lif = DualLIFState(dim, 0.9, 0.5)
        self.graph = Graph()
        self.attractor = AttractorField(dim, 2, 2, lr)  # danger / safe
        self.episodic = EpisodicMemory(50)
        self.context = ContextBuffer(20)

        self.prev_state = None
        self.prev_action = None
        self.stored_target = False
        self.danger_initialized = False

        self._init_action_embeddings()

    def _init_action_embeddings(self):
        self.action_vecs = []
        for i in range(self.n_actions):
            v = np.zeros(self.dim, dtype=np.float64)
            v[i * (self.dim // self.n_actions):(i + 1) * (self.dim // self.n_actions)] = 0.2
            self.action_vecs.append(v.tolist())

    def encode(self, view):
        return view_to_vector(view, n_types=max(6, int(view.max()) + 1))

    def observe_and_store(self, view):
        vec = self.encode(view)
        self.lif.step(vec.tolist(), False)
        return vec

    def store_target(self, view, data=1):
        vec = self.encode(view)
        self.wm.store(vec.tolist(), data)
        self.stored_target = True

    def recall_target(self, view):
        vec = self.encode(view)
        return self.wm.recall(vec.tolist())

    def has_target(self):
        return self.wm.has_target()

    def select_action(self, view, bonuses=None, explore_eps=0.1):
        if np.random.random() < explore_eps:
            return np.random.randint(self.n_actions), 0.0
        vec = self.encode(view)
        if bonuses is not None:
            return self.motor.select_with_bonus(self.lif, self.action_vecs, bonuses)
        return self.motor.select(self.lif, self.action_vecs)

    def record_transition(self, from_view, to_view, reward):
        fv = self.encode(from_view)
        tv = self.encode(to_view)
        self.graph.add_transition(fv.tolist(), tv.tolist(), reward)

    def learn_danger(self, view, is_danger):
        vec = self.encode(view)
        if not self.danger_initialized:
            self.attractor.add_class(vec.tolist())
            if not is_danger:
                self.attractor.add_prototype(vec.tolist(), 0)
                self.attractor.add_prototype(vec.tolist(), 1)
            self.danger_initialized = True
        label = 1 if is_danger else 0
        self.attractor.train_step(vec.tolist(), label)

    def predict_danger(self, view):
        vec = self.encode(view)
        label, dist = self.attractor.predict_with_distance(vec.tolist())
        return label, dist

    def reset_memory(self):
        self.wm.reset()
        self.lif = DualLIFState(self.dim, 0.9, 0.5)
        self.context = ContextBuffer(20)
        self.stored_target = False

    def store_context(self, token):
        self.context.push(token)

    def recall_context(self):
        ctx = self.context.as_slice()
        return self.episodic.recall(ctx)

    def store_episode(self):
        ctx = self.context.as_slice()
        if len(ctx) > 1:
            self.episodic.store(ctx)
