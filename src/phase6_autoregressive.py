"""
TSO Phase 6 — Generation Auto-Regressive Locale.
Apprentissage par traces lentes avec recompenses par pas de temps.
"""
import math, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

ETA_MOTOR = 0.05
TAU_SLOW = 20.0

VOCAB = ["CHAT", "CHIEN", "ANIMAL", "EST", "MAIS", "PAS", "OK", "FIN"]
IDX = {w: i for i, w in enumerate(VOCAB)}
TARGET = [IDX["CHAT"], IDX["EST"], IDX["ANIMAL"], IDX["MAIS"]]


class AutoRegressiveActor:
    def __init__(self, state_dim=12, vocab_size=len(VOCAB)):
        self.vocab_size = vocab_size
        self.W_motor = np.random.uniform(-0.2, 0.2, (state_dim, vocab_size))
        self.E_slow = np.zeros_like(self.W_motor)

    def reset(self):
        self.E_slow.fill(0.0)

    def propose_word(self, state):
        if np.random.rand() < 0.15:
            return np.random.randint(self.vocab_size)
        logits = state @ self.W_motor
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()
        return np.random.choice(self.vocab_size, p=probs)

    def update_traces(self, state, word_idx):
        self.E_slow *= math.exp(-1.0 / TAU_SLOW)
        self.E_slow[:, word_idx] += state

    def consolidate(self, M):
        self.W_motor += ETA_MOTOR * M * self.E_slow
        self.W_motor = np.clip(self.W_motor, -3.0, 3.0)
        self.reset()


def run_phase6():
    print("=" * 72)
    print("  TSO Phase 6 — Generation Auto-Regressive")
    print("  Recompenses par pas de temps + trace lente")
    print("=" * 72)

    # Etat initial: collision Chat/Chien
    state = np.zeros(8)
    state[[IDX["CHAT"], IDX["CHIEN"]]] = 1.0
    state /= np.linalg.norm(state)

    actor = AutoRegressiveActor()

    print(f"\n  Cible: CHAT EST ANIMAL MAIS\n")

    best_seq = None
    best_score = 0
    success_epoch = None

    for epoch in range(300):
        actor.reset()
        s = state.copy()
        generated = []
        total_M = 0.0

        for step in range(4):
            # Etat = contexte semantique + encodage de position (one-hot)
            pos_enc = np.eye(4)[step] * 0.5  # [0.5,0,0,0], [0,0.5,0,0], ...
            s_with_pos = np.concatenate([s, pos_enc])
            w = actor.propose_word(s_with_pos)
            generated.append(w)
            actor.update_traces(s_with_pos, w)
            step_M = 0.5 if w == TARGET[step] else 0.0
            total_M += step_M
            # Mise a jour de l'etat semantique
            s = s.copy()
            s[w] += 0.3
            s /= np.linalg.norm(s)

        # Bonus sequence complete
        if generated == TARGET:
            total_M = 3.0
            success_epoch = epoch

        actor.consolidate(total_M)

        mots = sum(1 for i in range(4) if generated[i] == TARGET[i])
        if mots > best_score:
            best_score = mots
            best_seq = generated

        if success_epoch is not None:
            seq_str = " ".join([VOCAB[i] for i in generated])
            print(f"  Epoch {epoch+1:3d} | {seq_str:20s} | M={total_M:.1f} | "
                  f"score={mots}/4 *** SUCCES ***")
            break

        if epoch % 30 == 0:
            seq_str = " ".join([VOCAB[i] for i in generated])
            print(f"  Epoch {epoch+1:3d} | {seq_str:20s} | M={total_M:.1f} | score={mots}/4")

    print(f"\n  >>> Meilleure sequence: {' '.join([VOCAB[i] for i in best_seq])} "
          f"(score={best_score}/4)")
    for mot in ["MAIS", "CHAT", "EST", "ANIMAL"]:
        w = actor.W_motor[:, IDX[mot]]
        print(f"    '{mot}': mean={w.mean():+.4f} max={w.max():+.4f}")

    if success_epoch is not None:
        print(f"\n  *** PHASE 6 VALIDEE (epoch {success_epoch+1}) ***")
        print("  TSO a appris une sequence de 4 mots par traces lentes")
        print("  sans BPTT (retropropagation temporelle).")
    else:
        print(f"\n  Meilleur score: {best_score}/4 — continuer l'entrainement")


if __name__ == "__main__":
    run_phase6()
