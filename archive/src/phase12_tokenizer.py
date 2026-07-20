"""
TSO Phase 12 — Branchement d'un vrai Tokenizer BPE (GPT-2).
TSO apprend à écrire le bon token parmi 50 257 sous-mots BPE via le Moteur Inverse.
Apprentissage par Hebbien direct (cible connue), comme la Phase 7.
Règle Delta locale pour stabiliser la convergence.
"""
import math, random
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNN_DIM = 50
EMBED_DIM = 384
ETA_INV = 0.05

class BPESemanticInverseDecoder:
    def __init__(self, vocab_embeddings):
        self.vocab_embs = vocab_embeddings.cpu().numpy()
        self.W_proj = np.random.randn(SNN_DIM, EMBED_DIM) * 0.01

    def embed(self, snn_state):
        return snn_state @ self.W_proj

    def top_tokens(self, snn_state, tokenizer, top_n=10):
        proj = self.embed(snn_state)
        proj_n = proj / (np.linalg.norm(proj) + 1e-8)
        embs_n = self.vocab_embs / (np.linalg.norm(self.vocab_embs, axis=1, keepdims=True) + 1e-8)
        cos = embs_n @ proj_n
        top_idx = np.argsort(cos)[-top_n:][::-1]
        return [(int(i), tokenizer.decode([int(i)]), float(cos[i])) for i in top_idx]

    def target_rank(self, snn_state, target_id):
        proj = self.embed(snn_state)
        proj_n = proj / (np.linalg.norm(proj) + 1e-8)
        embs_n = self.vocab_embs / (np.linalg.norm(self.vocab_embs, axis=1, keepdims=True) + 1e-8)
        cos = embs_n @ proj_n
        return int(np.where(np.argsort(cos)[::-1] == target_id)[0][0]) + 1

    def hebbian_step(self, snn_state, target_embedding, eta=ETA_INV):
        out = self.embed(snn_state)
        # Delta rule: W += eta * outer(s, t - out)  (Predictive Coding)
        # Minimizes ||t - W s||^2 without global backprop
        s = snn_state[:, None]
        self.W_proj += eta * (s * target_embedding[None, :] - s * out[None, :])

def run_phase12():
    print("="*72)
    print("  TSO Phase 12 — Vrai Tokenizer BPE (GPT-2) & Moteur Inverse")
    print("="*72)

    print("\n  Chargement du tokenizer GPT-2 (50 257 tokens)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("  Calcul des embeddings sémantiques pour 50 257 tokens...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    vocab_strings = [tokenizer.decode([i]) for i in range(vocab_size)]
    vocab_embeddings = embedder.encode(vocab_strings, convert_to_tensor=True, show_progress_bar=True)

    target_string = " but"
    target_token_id = tokenizer.encode(target_string, add_special_tokens=False)[0]
    target_embed = vocab_embeddings[target_token_id].cpu().numpy()
    target_embed = target_embed / (np.linalg.norm(target_embed) + 1e-8)
    print(f"  Token cible: '{target_string}' (ID: {target_token_id})")

    decoder = BPESemanticInverseDecoder(vocab_embeddings)

    snn_state = np.random.randn(SNN_DIM)
    snn_state = snn_state / (np.linalg.norm(snn_state) + 1e-8)

    print(f"\n  Apprentissage Hebbien (règle Delta) : SNN(50) → embedding(384) → '{target_string}'")
    print(f"  parmi {vocab_size} tokens BPE.\n")

    for epoch in range(40):
        decoder.hebbian_step(snn_state, target_embed)
        rank = decoder.target_rank(snn_state, target_token_id)
        proj = decoder.embed(snn_state)
        proj_n = proj / (np.linalg.norm(proj) + 1e-8)
        cos_actual = float(proj_n @ target_embed)
        if epoch < 5 or epoch % 5 == 4:
            top = decoder.top_tokens(snn_state, tokenizer, 5)
            top_str = " | ".join(f"{t[1].strip()}({t[2]:.3f})" for t in top)
            print(f"  Epoch {epoch+1:2d} | cos={cos_actual:.4f} | "
                  f"rang(cible)={rank}/50257")
            print(f"           top5: {top_str}")

    rank = decoder.target_rank(snn_state, target_token_id)
    print(f"\n  Rang final de '{target_string}' : {rank}/{vocab_size}")
    if rank <= 5:
        print(f"\n  *** PHASE 12 VALIDÉE ***")
        print(f"  Le Moteur Inverse projette l'état SNN(50) → embedding(384) → ")
        print(f"  token BPE '{target_string}' dans le top-{max(rank,5)} sur 50 257.")
        print(f"  Paramètres: {SNN_DIM}×{EMBED_DIM} = {SNN_DIM*EMBED_DIM} ({100*SNN_DIM*EMBED_DIM/(vocab_size*EMBED_DIM):.1f}% de la couche softmax d'un LLM)")
    print()

if __name__ == "__main__":
    run_phase12()
