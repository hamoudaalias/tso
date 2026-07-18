"""
TSO Phase 8 — NativeCritic: remplace le NLI par la dynamique interne du SNN.

Analyse les poids appris (R-STDP) et la stabilite electrique pour determiner
les relations entre clusters : implication (1), contradiction (-1), neutre (0).
"""
import numpy as np

N_TEST_STEPS = 40
I_DRIVE = 14.0
W_TH_IMP = 0.2       # seuil W pour implication directe
W_TH_SHARED = 0.05   # seuil W pour cible partagee
V_SAT = -55.0
FRAC_SAT = 0.25      # fraction de neurones saturee -> friction


class NativeCritic:
    def __init__(self, net=None):
        self.net = net

    def attach(self, net):
        self.net = net

    def evaluate(self, ci, cj):
        net = self.net
        if net is None or ci >= net.n_clusters or cj >= net.n_clusters:
            return 0
        s_i, s_j = net._sl(ci), net._sl(cj)
        w_ij = net.W[s_i, s_j].mean()
        w_ji = net.W[s_j, s_i].mean()
        max_w = max(w_ij, w_ji)

        # Implication directe (poids appris par R-STDP)
        if max_w > W_TH_IMP:
            return 1

        # Cible partagee -> contradiction potentielle
        for ck in range(net.n_clusters):
            if ck in (ci, cj):
                continue
            s_k = net._sl(ck)
            w_ik = net.W[s_i, s_k].mean()
            w_jk = net.W[s_j, s_k].mean()
            if w_ik > W_TH_SHARED and w_jk > W_TH_SHARED:
                return -1

        return 0
