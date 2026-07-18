"""
R-STDP plasticity and multi-scale eligibility traces.
"""
import numpy as np
import math


class EligibilityTrace:
    """Multi-timescale eligibility trace (Phase 9)."""
    def __init__(self, shape, tau_fast=5.0, tau_slow=50.0):
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.E_fast = np.zeros(shape, dtype=np.float32)
        self.E_slow = np.zeros(shape, dtype=np.float32)

    def update(self, pre, post, dt=0.5):
        a_fast = 1.0 - math.exp(-dt / self.tau_fast)
        a_slow = 1.0 - math.exp(-dt / self.tau_slow)
        hebb = np.outer(pre, post)
        self.E_fast += a_fast * (hebb - self.E_fast)
        self.E_slow += a_slow * (hebb - self.E_slow)

    def get(self, beta=0.3):
        return beta * self.E_fast + (1.0 - beta) * self.E_slow

    def decay(self, dt=0.5):
        self.E_fast *= math.exp(-dt / self.tau_fast)
        self.E_slow *= math.exp(-dt / self.tau_slow)

    def reset(self):
        self.E_fast.fill(0.0)
        self.E_slow.fill(0.0)


class RSTDPPlasticity:
    """
    Reward-modulated STDP with eligibility traces.
    Supports implication (exc) and exclusion (inh) edges.
    """
    def __init__(self, n_clusters, d, alpha_p=0.05, alpha_n=0.02):
        self.n = n_clusters
        self.d = d
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n

        self.W_exc = np.zeros((n_clusters, d, d), dtype=np.float32)
        self.W_inh = np.zeros((n_clusters, d, d), dtype=np.float32)
        self.trace = EligibilityTrace((d, d))
        self.spike_history = []

    def pre_post(self, pre_idx, post_idx):
        hebb = np.zeros((self.d, self.d), dtype=np.float32)
        hebb += self.trace.get(beta=0.3)
        self.W_exc[pre_idx] += self.alpha_p * hebb
        self.W_exc[pre_idx] = np.clip(self.W_exc[pre_idx], 0.0, 1.0)

    def reward_modulate(self, phi, delta_phi):
        if delta_phi < 0:
            scale = 1.0
        else:
            scale = 0.1
        self.W_exc *= (1.0 - self.alpha_n * scale)

    def reset(self):
        self.W_exc.fill(0.0)
        self.W_inh.fill(0.0)
        self.trace.reset()
