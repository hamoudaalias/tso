"""
TSO Core — pure math orchestration.

TSOCore manages clusters, edges, plasticity, and friction.
No NLP dependencies — works entirely on NumPy.
"""
import numpy as np
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
    """

    def __init__(self, max_clusters=100, d=5, gamma=0.5, epsilon=0.3):
        self.d = d
        self.max_clusters = max_clusters
        self.gamma = gamma
        self.epsilon = epsilon

        self.clusters = []
        self.labels = {}
        self.edges = []
        self.friction = FrictionCalculator(gamma=gamma, epsilon=epsilon)
        self.plasticity = RSTDPPlasticity(max_clusters, d)
        self.phi_history = []
        self.time = 0.0

    def add_cluster(self, label=""):
        if len(self.clusters) >= self.max_clusters:
            return -1
        idx = len(self.clusters)
        self.clusters.append(LIFCluster(self.d, label))
        self.labels[label] = idx
        return idx

    def add_edge(self, i, j, w, strength=1.0):
        self.edges.append((i, j, w, strength))

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

        return phi, rates, total_spikes

    def reset(self):
        for c in self.clusters:
            c.reset()
        self.phi_history = []
        self.time = 0.0

    def phi_gradient(self, window=10):
        if len(self.phi_history) < window:
            return 0.0
        return (self.phi_history[-1] - self.phi_history[-window]) / window
