"""
TSO Core — pure math orchestration.

TSOCore manages clusters, edges, plasticity, and friction.
No NLP dependencies — works entirely on NumPy.

V2 adds adaptive homeostasis: dynamic threshold, inertia gate
against premature expansion, and soft Double Mapping support.
"""
import numpy as np
from collections import deque
from .neurons import LIFCluster
from .friction import FrictionCalculator
from .plasticity import RSTDPPlasticity


class TSOCore:
    """
    Topographic Stabilization Operator — Core Engine.

    Attributes:
        clusters: list of LIFCluster objects
        edges: list of (i, j, w, strength) tuples
        friction: FrictionCalculator instance
        plasticity: RSTDPPlasticity instance
        phi_history: list of phi values over time

    V2 additions:
        activity_history: sliding window for adaptive threshold
        slow_variance: sliding window for inertia gate
        dynamic_theta_c: noise-adapted trigger threshold
    """

    def __init__(self, max_clusters=100, d=5, gamma=0.5, epsilon=0.3,
                 history_size=50, base_theta_c=0.5, inertia_threshold=0.1):
        self.d = d
        self.max_clusters = max_clusters
        self.gamma = gamma
        self.epsilon = epsilon
        self.base_theta_c = base_theta_c
        self.inertia_threshold = inertia_threshold

        self.clusters = []
        self.labels = {}
        self.edges = []
        self.friction = FrictionCalculator(gamma=gamma, epsilon=epsilon)
        self.plasticity = RSTDPPlasticity(max_clusters, d)
        self.phi_history = []
        self.time = 0.0

        # Adaptive homeostasis (V2)
        self.activity_history = deque(maxlen=history_size)
        self.slow_trace_history = deque(maxlen=history_size)
        self.dynamic_theta_c = base_theta_c

    def add_cluster(self, label=""):
        if len(self.clusters) >= self.max_clusters:
            return -1
        idx = len(self.clusters)
        self.clusters.append(LIFCluster(self.d, label))
        self.labels[label] = idx
        return idx

    def add_edge(self, i, j, w, strength=1.0):
        self.edges.append((i, j, w, strength))

    def update_homeostasis(self, current_activity, current_slow_trace):
        """
        Adaptive threshold and inertia gate update.

        1. Dynamic theta_c rises with network noise (prevents
           false-positive contradiction detection during semantic storms).
        2. Slow trace variance measures network stability for inertia gate.
        """
        self.activity_history.append(current_activity)
        self.slow_trace_history.append(current_slow_trace)

        if len(self.activity_history) > 10:
            std_act = float(np.std(self.activity_history))
            self.dynamic_theta_c = self.base_theta_c + (std_act * 0.5)

    def check_inertia_gate(self):
        """
        Block expansion if the network is still unstable.

        Returns True only if the slow trace variance is below
        the inertia threshold, indicating stable consolidation.
        """
        if len(self.slow_trace_history) < 10:
            return False
        variance = float(np.var(self.slow_trace_history))
        return variance < self.inertia_threshold

    def should_trigger_expansion(self, current_phi, min_implication):
        """
        Autonomous Native Critic with three guards:

        1. phi must exceed dynamic (noise-adapted) threshold
        2. implications must be solidly consolidated (>= gamma)
        3. network must be stable (inertia gate)
        """
        if current_phi < self.dynamic_theta_c:
            return False
        if min_implication < self.gamma:
            return False
        if not self.check_inertia_gate():
            return False
        return True

    def step(self, I_ext, dt=0.5):
        rates = np.zeros(len(self.clusters), dtype=np.float32)
        total_spikes = 0

        for ci, c in enumerate(self.clusters):
            s = c.step(I_ext[ci], dt=dt)
            rates[ci] = c.rate
            total_spikes += s

        phi = self.friction.compute_phi(rates, self.edges)
        self.phi_history.append(phi)
        self.time += dt

        if len(self.phi_history) > 1:
            dphi = phi - self.phi_history[-2]
            self.plasticity.reward_modulate(phi, dphi)

        mean_activity = float(np.mean(rates))
        slow_trace = float(np.mean([c.rate for c in self.clusters]))
        self.update_homeostasis(mean_activity, slow_trace)

        return phi, rates, total_spikes

    def reset(self):
        for c in self.clusters:
            c.reset()
        self.phi_history = []
        self.activity_history.clear()
        self.slow_trace_history.clear()
        self.dynamic_theta_c = self.base_theta_c
        self.time = 0.0

    def phi_gradient(self, window=10):
        if len(self.phi_history) < window:
            return 0.0
        return (self.phi_history[-1] - self.phi_history[-window]) / window
