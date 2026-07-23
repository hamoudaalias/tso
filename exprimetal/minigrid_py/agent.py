import numpy as np
from tso_pyo3 import (
    LIFState,
    AttractorField,
    Graph,
    EpisodicMemory,
    ContextBuffer,
    AssociativeMemory,
)

OBJECT_IDS = {
    "unseen": 0, "empty": 1, "wall": 2, "floor": 3,
    "door": 4, "key": 5, "ball": 6, "box": 7,
    "goal": 8, "lava": 9, "agent": 10,
}

COLOR_IDS = {
    "red": 0, "green": 1, "blue": 2, "purple": 3,
    "yellow": 4, "grey": 5,
}

STATE_DIM = 64
IMG_CHANNELS = 3


class MinigridEncoder:
    def __init__(self, state_dim=STATE_DIM):
        self.state_dim = state_dim
        input_dim = 7 * 7 * IMG_CHANNELS + 4 + 32
        rng = np.random.RandomState(42)
        self.proj = rng.randn(input_dim, state_dim).astype(np.float32) / np.sqrt(input_dim)

    def encode(self, obs):
        img = obs["image"].astype(np.float32).flatten() / 10.0
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[obs["direction"]] = 1.0
        mission = obs.get("mission", "")
        mission_vec = self._encode_mission(mission)
        state_in = np.concatenate([img, dir_onehot, mission_vec])
        return (state_in @ self.proj).astype(np.float64)

    def _encode_mission(self, mission):
        v = np.zeros(32, dtype=np.float32)
        for i, ch in enumerate(mission.encode("utf-8", errors="replace")[:32]):
            v[i] = ch / 255.0
        return v


class TSOAgent:
    def __init__(self, state_dim=STATE_DIM, lif_alpha=0.85, epsilon=0.5):
        self.encoder = MinigridEncoder(state_dim)
        self.lif = LIFState(state_dim, lif_alpha)
        self.field = AttractorField(state_dim, 2, 4, 0.06)
        self.episodic = EpisodicMemory(200)
        self.context = ContextBuffer(30)
        self.graph = Graph()
        self.assoc = AssociativeMemory()
        self.q_table = {}
        self.epsilon = epsilon
        self.lr = 0.15
        self.gamma = 0.9
        self.step_count = 0
        self.prev_state_vec = None
        self.prev_action = None
        self.prev_state_node = None
        self.state_dim = state_dim
        self.goal_class = None

    def reset(self):
        self.lif = LIFState(self.state_dim, 0.85)
        self.context = ContextBuffer(30)
        self.step_count = 0
        self.prev_state_vec = None
        self.prev_action = None
        self.prev_state_node = None
        self.goal_class = None

    def encode(self, obs):
        raw = self.encoder.encode(obs)
        self.lif.step(raw.tolist(), False)
        return np.array(self.lif.get_state())

    def act(self, state_vec, env):
        self.prev_state_vec = state_vec.copy()
        self.step_count += 1

        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
            self.prev_action = action
            return action

        state_key = self._state_key(state_vec)
        best_action = 0
        best_q = -1e9
        for a in range(env.action_space.n):
            q = self.q_table.get((state_key, a), 0.0)
            if q > best_q:
                best_q = q
                best_action = a

        self.prev_action = best_action
        return best_action

    def learn(self, state_vec, action, reward, next_state_vec, done, info=None):
        state_key = self._state_key(state_vec)
        next_key = self._state_key(next_state_vec)

        max_next_q = 0.0
        if not done:
            for a in range(7):
                nq = self.q_table.get((next_key, a), 0.0)
                if nq > max_next_q:
                    max_next_q = nq

        old_q = self.q_table.get((state_key, action), 0.0)
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state_key, action)] = new_q

        self.learn_state(state_vec, reward)

        if self.prev_state_node is not None:
            next_node = self.graph.node_count()
            self.graph.add_node(next_state_vec.tolist())
            weight = 1 if reward >= -0.1 else (-1 if reward < 0 else 2)
            self.graph.add_edge(self.prev_state_node, next_node, weight)

        self.prev_state_node = self.graph.node_count() - 1
        if self.prev_state_node < 0:
            self.prev_state_node = 0

        self.context.push(abs(hash(state_key)) % 10000)

    def learn_state(self, state_vec, reward):
        if reward > 0.5:
            if self.goal_class is None:
                self.goal_class = self.field.add_class(state_vec.tolist())
            else:
                self.field.add_prototype(state_vec.tolist(), self.goal_class)
        elif reward < -0.5:
            cls = self.field.n_classes()
            if cls <= 1:
                self.field.add_class(state_vec.tolist())
            dc = self.field.n_classes() - 1
            self.field.add_prototype(state_vec.tolist(), dc)

    def recall_episodic(self):
        ctx = self.context.as_slice()
        if len(ctx) < 2:
            return None
        return self.episodic.recall(ctx)

    def store_episode(self, sequence):
        self.episodic.store(sequence)

    def _state_key(self, vec):
        return tuple(np.round(vec[:8] * 10).astype(int))

    def decay_epsilon(self, factor=0.99):
        self.epsilon = max(self.epsilon * factor, 0.02)

    def get_prototypes(self):
        return self.field.get_prototypes()
