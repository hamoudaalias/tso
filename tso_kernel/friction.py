"""
Friction computation — the thermodynamic core of TSO.

Phi measures topological surprise: the distance between the predicted
conceptual state and the observed one.
"""
import numpy as np
import math


class FrictionCalculator:
    """
    Computes Phi = sum of violations across implication/exclusion edges.

    An implication edge (w=+1) expects correlated firing: if dot < gamma
    there is tension. An exclusion edge (w=-1) expects anti-correlation:
    if dot > epsilon there is tension.
    """
    def __init__(self, gamma=0.5, epsilon=0.3):
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_phi(self, rates, edges):
        phi = 0.0
        for i, j, w, strength in edges:
            dot = rates[i] * rates[j]
            if w > 0:
                phi += strength * max(0.0, self.gamma - dot)
            elif w < 0:
                phi += strength * max(0.0, dot - self.epsilon)
        return phi

    def conceptual_phi(self, prev_cluster, curr_cluster, transition_matrix, n_concepts):
        """
        Phase 13 style: phi = 1 - p(curr|prev) * N.
        """
        total = transition_matrix[prev_cluster].sum()
        if total < 1e-6:
            return 1.0
        p = transition_matrix[prev_cluster, curr_cluster] / total
        return 1.0 - p * n_concepts

    def topological_phi(self, pred_bmu, actual_bmu, som_shape):
        """
        Phase 13 V3 style: phi = Euclidean distance on SOM grid.
        """
        rows, cols = som_shape
        ri, ci = divmod(pred_bmu, cols)
        rj, cj = divmod(actual_bmu, cols)
        return math.sqrt((ri - rj)**2 + (ci - cj)**2) / (rows + cols)
