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
        Project conflicting concepts into disjoint subspaces.

        Each concept gets its own d-dimensional block within a k*d space.
        The context is replicated across all blocks. This guarantees
        that the dot product between exclusive concepts is zero
        while context similarities are preserved.

        Args:
            vectors: dict mapping label -> 1D array (length d)
            concepts: list of concept labels to orthogonalize
            context: label of the context element
            d: intrinsic dimension

        Returns:
            new_vectors: dict mapping label -> expanded array (length k*d)
            nd: new dimension
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
