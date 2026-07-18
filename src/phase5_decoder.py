"""
TSO Phase 5 — Le Décodeur Auto-Régressif Local.
TSO apprend a ecrire "MAIS" pour dissiper sa tension.
"""
import math, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

GAMMA, EPSILON, THETA_T, THETA_C = 0.15, 0.08, 0.02, 0.15
D = 5
ETA_MOTOR = 0.05

VOCAB = ["OK", "IMP", "CONTR", "MAIS"]
IDX_OK, IDX_IMP, IDX_CONTR, IDX_MAIS = 0, 1, 2, 3


class MotorCortex:
    def __init__(self, n_max_clusters, vocab_size=len(VOCAB)):
        self.vocab_size = vocab_size
        self.W_motor = np.random.uniform(-0.01, 0.01, (n_max_clusters * D, vocab_size))
        self.eligibility_motor = np.zeros_like(self.W_motor)
        self.epsilon = 0.3

    def propose_word(self, cluster_rates, n_active_clusters):
        state = cluster_rates[:n_active_clusters * D]
        # Epsilon-greedy: exploration vs exploitation
        if np.random.rand() < self.epsilon:
            word_idx = np.random.randint(self.vocab_size)
        else:
            logits = state @ self.W_motor[:n_active_clusters * D, :]
            word_idx = int(np.argmax(logits))
        return word_idx, state

    def update_traces(self, state, word_idx):
        self.eligibility_motor.fill(0.0)
        # Trace normalisee par la norme de l'etat
        norm = np.linalg.norm(state) + 1e-8
        self.eligibility_motor[:len(state), word_idx] = state / norm

    def consolidate(self, M_signal):
        self.W_motor += ETA_MOTOR * M_signal * self.eligibility_motor
        # Normaliser les poids pour eviter l'emballement
        for i in range(self.vocab_size):
            norm = np.linalg.norm(self.W_motor[:, i]) + 1e-8
            self.W_motor[:, i] /= norm
        self.eligibility_motor.fill(0.0)


def run_phase5():
    print("=" * 72)
    print("  TSO Phase 5 — Le Decodeur Local")
    print("  Apprentissage du mot 'MAIS' par R-STDP et renforcement")
    print("=" * 72)

    n_clusters = 3
    # Chaque cluster a D=5 neurones, donc 15 dimensions
    rates = np.tile(np.array([15.0, 12.0, 14.0]), D)

    motor = MotorCortex(n_max_clusters=10)

    print("\n  Etat initial: Chat et Chien co-actives (Contradiction d'exclusion).")
    print("  Objectif: L'Actor doit apprendre a emettre 'MAIS' (IDX 3)\n")

    success_history = []

    for epoch in range(50):
        word_idx, state = motor.propose_word(rates, n_clusters)
        word_str = VOCAB[word_idx]

        delta_phi = 0.0
        M_signal = 0.0

        if word_idx == IDX_MAIS:
            delta_phi = 0.15
            M_signal = 1.0
        elif word_idx == IDX_CONTR:
            delta_phi = 0.05
            M_signal = 0.3
        else:
            delta_phi = 0.0
            M_signal = 0.0

        motor.update_traces(state, word_idx)
        motor.consolidate(M_signal)

        success_history.append(word_idx == IDX_MAIS)
        success_rate = np.mean(success_history[-10:])

        if epoch % 10 == 0 or epoch == 49:
            print(f"  Epoch {epoch+1:2d} | Mot: {word_str:4s} | "
                  f"DeltaPhi={delta_phi:.2f} | M={M_signal:.1f} | "
                  f"Succes MAIS(10): {success_rate:.0%}")

    print("\n  >>> Analyse des poids moteurs finaux :")
    for i, word in enumerate(VOCAB):
        mean_w = np.mean(motor.W_motor[:n_clusters * D, i])
        print(f"    Poids moyen pour '{word}': {mean_w:.4f}")

    if np.mean(success_history[-20:]) > 0.8:
        print("\n  *** PHASE 5 VALIDEE ***")
        print("  Le reseau a appris sans backpropagation que face a une friction")
        print("  d'exclusion, l'emission du mot 'MAIS' est l'action optimale.")
    else:
        print("\n  Convergence partielle — augmenter les iterations ou le taux")


if __name__ == "__main__":
    run_phase5()
