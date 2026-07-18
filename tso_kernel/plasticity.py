"""
R-STDP plasticity, multi-scale eligibility traces, and Friction-Gated
Consolidation.

V2: Friction-Gated Consolidation prevents representation collapse.
Before applying LTP, checks semantic similarity via target embeddings.
If cos < 0 (exclusive), applies LTD instead to block indirect cascades
such as Chat→Animal→Chien creating a spurious Chat↔Chien edge.
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
    Reward-modulated STDP with Friction-Gated Consolidation.

    Maintains a flat weight matrix W[n_clusters, n_clusters] representing
    implication strength between concept clusters. The gate checks target
    embedding similarity before allowing LTP: exclusive concepts (cos < 0)
    can never form an implication edge, even through indirect cascades.

    Attributes:
        W: weight matrix (n, n), row = pre, col = post
        z_targets: dict mapping cluster_idx -> semantic embedding
        eligibility: scalar per pre-cluster, tracks recent co-activation
    """
    def __init__(self, n_clusters, alpha_p=0.05, alpha_n=0.02, inhib_factor=0.05):
        self.n = n_clusters
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.inhib_factor = inhib_factor

        self.W = np.zeros((n_clusters, n_clusters), dtype=np.float32)
        self.z_targets = {}
        self.eligibility = np.zeros(n_clusters, dtype=np.float32)
        self.decay = math.exp(-1.0 / 20.0)

    def register_target(self, idx, z_vec):
        """Register a cluster's semantic target embedding for gate checks."""
        self.z_targets[idx] = z_vec.copy()

    def consolidate(self, pre_idx, post_idx, rate_pre=1.0, rate_post=1.0):
        """
        Friction-Gated consolidation step.

        If pre and post have target embeddings with cos < 0 (exclusive),
        LTD is applied to prevent cascade collapse.
        Otherwise, standard Hebbian LTP is applied.
        """
        el = self.eligibility[pre_idx] * self.decay + rate_pre * rate_post
        self.eligibility[pre_idx] = el

        if pre_idx in self.z_targets and post_idx in self.z_targets:
            a = self.z_targets[pre_idx]
            b = self.z_targets[post_idx]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            if sim < 0:
                self.W[pre_idx, post_idx] -= self.inhib_factor * el
                self.W[pre_idx, post_idx] = max(0.0, self.W[pre_idx, post_idx])
                return

        self.W[pre_idx, post_idx] += self.alpha_p * el
        self.W[pre_idx, post_idx] = min(1.5, max(0.0, self.W[pre_idx, post_idx]))

    def reward_modulate(self, phi, delta_phi):
        """Global reward signal scales down weights when phi rises."""
        scale = 1.0 if delta_phi < 0 else 0.1
        self.W *= (1.0 - self.alpha_n * scale)
        self.W = np.clip(self.W, 0.0, 1.5)

    def reset(self):
        self.W.fill(0.0)
        self.eligibility.fill(0.0)
