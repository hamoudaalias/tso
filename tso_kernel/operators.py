"""
Geometric operators for TSO.

Double Mapping, topological expansion, and the Inverse Motor projection.
All operations are local, purely geometric — no global gradient.
"""
import numpy as np


class TopographicOperator:
    """
    Static methods for the core TSO geometric transforms.
    """

    @staticmethod
    def double_mapping(vectors, concepts, context, d):
        """
        Strict Double Mapping (Lemma 1).
        Projects conflicting concepts into fully disjoint subspaces.
        Dot product between exclusive concepts is exactly zero.
        """
        k = len(concepts)
        nd = k * d
        new_vecs = {}

        for c in concepts:
            zp = np.zeros(nd, dtype=np.float32)
            i = concepts.index(c)
            zp[i*d:(i+1)*d] = vectors[c]
            new_vecs[c] = zp

        ctx_zp = np.zeros(nd, dtype=np.float32)
        for i in range(k):
            ctx_zp[i*d:(i+1)*d] = vectors[context]
        new_vecs[context] = ctx_zp

        return new_vecs, nd

    @staticmethod
    def soft_double_mapping(vectors, concepts, context, d, alpha=0.1):
        """
        Soft Double Mapping with residual shared space (Lemma 1 v2).

        Instead of strict orthogonality, a residual coefficient alpha
        preserves shared features between exclusive concepts. This avoids
        "semantic lobotomy" where concepts with common traits (e.g.
        Chat and Chien both being mammals) lose all similarity.
        """
        k = len(concepts)
        nd = k * d
        new_vecs = {}

        for idx, c in enumerate(concepts):
            zp = np.zeros(nd, dtype=np.float32)
            zp[idx*d:(idx+1)*d] = vectors[c]
            for other_idx in range(k):
                if other_idx != idx:
                    zp[other_idx*d:(other_idx+1)*d] = vectors[c] * alpha
            new_vecs[c] = zp

        ctx_zp = np.zeros(nd, dtype=np.float32)
        for idx in range(k):
            ctx_zp[idx*d:(idx+1)*d] = vectors[context]
        new_vecs[context] = ctx_zp

        return new_vecs, nd

    @staticmethod
    def inverse_motor(state, W_proj, embeddings, cluster_mask):
        """
        Project SNN state to embedding space and select best word
        within the target cluster.

        Args:
            state: SNN state vector
            W_proj: projection matrix (state_dim x embed_dim)
            embeddings: word embedding matrix (vocab_size x embed_dim)
            cluster_mask: boolean array (vocab_size,) for the target cluster

        Returns:
            best_idx: index of best matching word
            best_cos: cosine similarity score
        """
        proj = state @ W_proj
        norms = np.linalg.norm(proj) * np.linalg.norm(embeddings, axis=1) + 1e-8
        cosines = (embeddings @ proj) / norms
        cosines[~cluster_mask] = -1.0
        best_idx = int(np.argmax(cosines))
        return best_idx, float(cosines[best_idx])

    @staticmethod
    def expand_topology(W_exc, W_inh, dormant_threshold=0.01):
        """
        Phase 2: detect edge saturation and flag dormant neurons
        for recruitment. Returns indices of saturated edges.
        """
        n = W_exc.shape[0]
        saturated = []
        for i in range(n):
            for j in range(n):
                if abs(W_exc[i, j]) > 1.0 - dormant_threshold:
                    saturated.append((i, j))
        return saturated
