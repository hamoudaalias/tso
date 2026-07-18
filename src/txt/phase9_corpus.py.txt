"""
TSO Phase 9 (V5) — Copie Longue Distance avec sortie one-hot.
Le moteur apprend une matrice W_motor (state_dim x vocab_size) par trace Hebbienne.
La sortie one-hot est plus discriminante que la projection embedding.
"""
import math, random
import numpy as np
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED)

EMBED_DIM = 32
BOS, EOS = 0, 1
RESERVED = 2

ETA_MOTOR = 0.05
TAU_SLOW = 20.0
TAU_CONTEXT = 200.0
EPSILON = 0.0
N_EPOCHS = 200


def make_vocab(vocab_size):
    emb = np.random.randn(vocab_size, EMBED_DIM)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def make_corpus(n_seqs, seq_len, vocab_size):
    """Copie : le dernier token = premier token."""
    corpus = []
    for _ in range(n_seqs):
        first = random.randint(RESERVED, vocab_size - 1)
        middle = [random.randint(RESERVED, vocab_size - 1)
                  for _ in range(seq_len - 2)]
        corpus.append([BOS, first] + middle + [first, EOS])
    return corpus


class ContextBuffer:
    def __init__(self, tau=TAU_CONTEXT):
        self.tau = tau
        self.buf = np.zeros(EMBED_DIM)
        self.wsum = 0.0
    def reset(self):
        self.buf.fill(0.0)
        self.wsum = 0.0
    def update(self, word_emb):
        d = math.exp(-1.0 / self.tau)
        self.buf = self.buf * d + word_emb
        self.wsum = self.wsum * d + 1.0
    def get(self):
        return self.buf / (self.wsum + 1e-8)


class SeqActor:
    def __init__(self, vocab_size, use_context=False):
        self.vs = vocab_size
        self.use_context = use_context
        ctx_dim = EMBED_DIM if use_context else 0
        state_dim = EMBED_DIM + ctx_dim
        self.W = np.random.randn(state_dim, vocab_size) * 0.01
        self.E = np.zeros_like(self.W)
        self.ctx = ContextBuffer(TAU_CONTEXT) if use_context else None

    def reset(self):
        self.E.fill(0.0)
        if self.ctx is not None:
            self.ctx.reset()

    def build_state(self, word_emb):
        if self.ctx is not None:
            return np.concatenate([word_emb, self.ctx.get()])
        return word_emb

    def context_only_state(self):
        if self.ctx is not None:
            z = np.zeros(EMBED_DIM)
            return np.concatenate([z, self.ctx.get()])
        return np.zeros(EMBED_DIM)

    def propose(self, state):
        logits = state @ self.W
        return int(np.argmax(logits[RESERVED:])) + RESERVED

    def store_trace(self, state, target_idx):
        oh = np.zeros(self.vs)
        oh[target_idx] = 1.0
        self.E += np.outer(state, oh)

    def update_context(self, word_emb):
        if self.ctx is not None:
            self.ctx.update(word_emb)

    def consolidate(self, M):
        self.W += ETA_MOTOR * M * self.E
        self.E.fill(0.0)


def run_experiment(label, vocab_emb, corpus, vocab_size, use_context=False):
    actor = SeqActor(vocab_size, use_context)
    success_epoch = None
    last_10 = []

    for epoch in range(N_EPOCHS):
        random.shuffle(corpus)
        correct = 0
        total = 0

        for seq in corpus:
            actor.reset()
            final_state = None
            final_target = None

            for pos in range(len(seq) - 1):
                idx = seq[pos]
                next_idx = seq[pos + 1]
                w_emb = vocab_emb[idx]
                state = actor.build_state(w_emb)
                actor.update_context(w_emb)

                if pos == len(seq) - 3:
                    cstate = actor.context_only_state()
                    final_state = cstate
                    final_target = next_idx
                    pred = actor.propose(cstate)
                    total += 1
                    if pred == next_idx:
                        correct += 1

            if final_state is not None:
                actor.store_trace(final_state, final_target)
                actor.consolidate(1.0)

        acc = correct / max(total, 1)
        last_10.append(acc)
        last_10 = last_10[-10:]

        if success_epoch is None and np.mean(last_10) > 0.70:
            success_epoch = epoch + 1

        if epoch % 20 == 0 or (success_epoch is not None
                               and epoch < success_epoch + 2):
            print(f"    Epoch {epoch+1:3d} | acc={acc:.3f} "
                  f"| avg10={np.mean(last_10):.3f}"
                  f"{' *** SUCCES ***' if success_epoch == epoch+1 else ''}")

    final = np.mean(last_10)
    print(f"  >> acc={final:.3f} "
          f"{f'CONVERGE epoch {success_epoch}' if success_epoch else 'ECHEC'}")
    return success_epoch, final


def main():
    start = time.time()
    print("=" * 72)
    print("  TSO Phase 9 (V5) — Copie Longue Distance (sortie one-hot)")
    print("=" * 72)

    for vocab_size, seq_len in [(30, 5), (30, 20), (100, 5), (100, 20)]:
        vocab_emb = make_vocab(vocab_size)
        corpus = make_corpus(800, seq_len, vocab_size)
        print(f"\n--- Vocab={vocab_size}, SeqLen={seq_len} ---")
        for label, ctx in [("tau=20 seul", False),
                           ("+ctx tau=200", True)]:
            ep, acc = run_experiment(
                label, vocab_emb, corpus, vocab_size, ctx)
            print(f"  {label:20s}: acc={acc:.3f} {'OK' if ep else 'ECHEC'}")

    print(f"\n  Temps: {time.time() - start:.1f}s\n")


if __name__ == "__main__":
    main()
