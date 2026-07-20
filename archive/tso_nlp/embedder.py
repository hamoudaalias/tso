"""
Thin wrapper around Sentence-Transformers and PyTorch.

Provides MiniLM embeddings (384d) with optional GPU acceleration.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class MiniLMEmbedder:
    """MiniLM-L6-v2 embedder with caching."""
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.cache = {}

    def encode(self, texts, normalize=True):
        results = []
        uncached = []
        for t in texts:
            if t in self.cache:
                results.append(self.cache[t])
            else:
                uncached.append(t)

        if uncached:
            embs = self.model.encode(
                uncached, convert_to_tensor=True, show_progress_bar=False
            )
            if normalize:
                embs = embs / (torch.norm(embs, dim=1, keepdim=True) + 1e-8)
            embs = embs.cpu().numpy().astype(np.float32)
            for t, e in zip(uncached, embs):
                self.cache[t] = e
                results.append(e)

        return np.array(results) if results else np.empty((0, 384), dtype=np.float32)

    def random_projection(self, target_dim, seed=0):
        rng = np.random.RandomState(seed)
        P = rng.randn(384, target_dim).astype(np.float32)
        P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
        return P
