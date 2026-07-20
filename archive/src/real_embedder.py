"""
TSO Phase 8 — Encodeur NLP réel (MiniLM uniquement, NLI supprime).
"""
import numpy as np
from sentence_transformers import SentenceTransformer, util

class RealEmbedder:
    def __init__(self, device=None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"  [RealEmbedder] Chargement de MiniLM sur {device}...")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self._cache = {}

    def embed(self, text):
        if text not in self._cache:
            self._cache[text] = self.st_model.encode(text, normalize_embeddings=True)
        return self._cache[text].copy()

    def embed_many(self, texts):
        return self.st_model.encode(texts, normalize_embeddings=True)

def quick_test():
    print("Test RealEmbedder...")
    emb = RealEmbedder()
    v_cat = emb.embed("cat")
    v_dog = emb.embed("dog")
    v_car = emb.embed("car")
    print(f"  Dimensions: {v_cat.shape}")
    print(f"  cos(cat, dog) = {float(util.cos_sim(v_cat, v_dog)[0][0]):.4f}")
    print(f"  cos(cat, car) = {float(util.cos_sim(v_cat, v_car)[0][0]):.4f}")
    print("Test OK.")


if __name__ == "__main__":
    quick_test()
