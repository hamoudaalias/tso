import numpy as np
from .hierarchy import HierarchicalGraph


class FrictionRouter:
    """
    Routage événementiel par friction (skip-attention).

    Φ bas  → flux linéaire, pas d'attention calculée.
    Φ haut → alignement géométrique (double-mapping local).
    """

    def __init__(self, hierarchy: HierarchicalGraph, friction_threshold=0.3):
        self.hierarchy = hierarchy
        self.threshold = friction_threshold
        self.phi_history = []
        self.conflict_history = []
        self.n_triggers = 0

    def compute_local_friction(self, alpha_state, beta_state):
        if alpha_state is None or beta_state is None:
            return 0.0
        a = np.asarray(alpha_state, dtype=np.float32)
        b = np.asarray(beta_state, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        cos_sim = np.dot(a, b) / (na * nb)
        return float(1.0 - cos_sim)

    def route(self, sentence_tokens, surprise=None):
        self.hierarchy.active_alpha.clear()
        self.hierarchy.active_beta.clear()
        self.phi_history.clear()
        skip_triggered = False

        for token in sentence_tokens:
            self.hierarchy.process_token(token)

            beta_vec = np.zeros(len(self.hierarchy.beta_nodes), dtype=np.float32)
            for bid in self.hierarchy.active_beta:
                beta_vec[bid] = 1.0

            phi = self.compute_local_friction(
                list(self.hierarchy.alpha_edges.values()),
                beta_vec,
            )
            self.phi_history.append(phi)

            if phi > self.threshold:
                skip_triggered = True

        # Ph27 : conflit cross-window + surprise running_mean
        gamma_vec = self.hierarchy.compute_cross_window_friction(sentence_tokens)
        conflict = float(gamma_vec[1])

        # Ph28 : recrutement β forcé par conflit γ
        if conflict > self.threshold:
            skip_triggered = True
            self.n_triggers += 1
            n = len(sentence_tokens)
            if n >= 2:
                mid = n // 2
                self.hierarchy.force_beta_node(sentence_tokens[:mid], sentence_tokens[mid:])

        # Surprise-based routing : si h est loin de running_mean, on route
        if surprise is not None and surprise > self.threshold:
            skip_triggered = True
            self.n_triggers += 1

        self.conflict_history.append(conflict)

        beta_out = np.zeros(self.hierarchy.max_beta, dtype=np.float32)
        for bid in self.hierarchy.active_beta:
            beta_out[bid] = 1.0

        return {
            "beta": beta_out,
            "gamma": gamma_vec,
            "phi_max": max(self.phi_history) if self.phi_history else 0.0,
            "conflict": conflict,
            "surprise": surprise if surprise is not None else 0.0,
            "skip_triggered": skip_triggered,
        }
