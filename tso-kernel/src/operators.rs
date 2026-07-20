use ndarray::Array1;
use std::collections::HashMap;

pub struct TopographicOperator;

impl TopographicOperator {
    /// Strict Double Mapping (Lemma 1).
    /// Projects conflicting concepts into fully disjoint subspaces.
    /// Returns (new_vectors, new_dimension).
    pub fn double_mapping(
        vectors: &HashMap<String, Array1<f64>>,
        concepts: &[String],
        context: &str,
        d: usize,
    ) -> (HashMap<String, Array1<f64>>, usize) {
        let k = concepts.len();
        let nd = k * d;
        let mut new_vecs = HashMap::new();

        for (i, c) in concepts.iter().enumerate() {
            let mut zp = Array1::zeros(nd);
            if let Some(v) = vectors.get(c) {
                for j in 0..d {
                    zp[i * d + j] = v[j];
                }
            }
            new_vecs.insert(c.clone(), zp);
        }

        let mut ctx_zp = Array1::zeros(nd);
        if let Some(ctx_v) = vectors.get(context) {
            for i in 0..k {
                for j in 0..d {
                    ctx_zp[i * d + j] = ctx_v[j];
                }
            }
        }
        new_vecs.insert(context.to_string(), ctx_zp);

        (new_vecs, nd)
    }

    /// Soft Double Mapping with residual shared space (Lemma 1 v2).
    /// Instead of strict orthogonality, preserves shared features
    /// between exclusive concepts via an alpha coefficient.
    pub fn soft_double_mapping(
        vectors: &HashMap<String, Array1<f64>>,
        concepts: &[String],
        context: &str,
        d: usize,
        alpha: f64,
    ) -> (HashMap<String, Array1<f64>>, usize) {
        let k = concepts.len();
        let nd = k * d;
        let mut new_vecs = HashMap::new();

        for (i, c) in concepts.iter().enumerate() {
            let mut zp = Array1::zeros(nd);
            if let Some(v) = vectors.get(c) {
                for j in 0..d {
                    zp[i * d + j] = v[j];
                }
                for other_idx in 0..k {
                    if other_idx != i {
                        for j in 0..d {
                            zp[other_idx * d + j] = v[j] * alpha;
                        }
                    }
                }
            }
            new_vecs.insert(c.clone(), zp);
        }

        let mut ctx_zp = Array1::zeros(nd);
        if let Some(ctx_v) = vectors.get(context) {
            for i in 0..k {
                for j in 0..d {
                    ctx_zp[i * d + j] = ctx_v[j];
                }
            }
        }
        new_vecs.insert(context.to_string(), ctx_zp);

        (new_vecs, nd)
    }

    /// Inverse Motor: project SNN state to embedding space and select best word.
    ///
    /// Returns (best_index, cosine_similarity).
    pub fn inverse_motor(
        state: &Array1<f64>,
        w_proj: &Array1<f64>,
        embeddings: &[Array1<f64>],
        cluster_mask: &[bool],
    ) -> (usize, f64) {
        let state_dim = state.len();
        let embed_dim = w_proj.len() / state_dim;
        let proj_vec = apply_projection(state, w_proj, state_dim, embed_dim);

        let mut best_idx = 0;
        let mut best_cos = -2.0;

        for (i, emb) in embeddings.iter().enumerate() {
            if !cluster_mask[i] {
                continue;
            }
            let dot = emb.dot(&proj_vec);
            let na = emb.dot(emb).sqrt();
            let nb = proj_vec.dot(&proj_vec).sqrt();
            let cos = dot / (na * nb + 1e-8);
            if cos > best_cos {
                best_cos = cos;
                best_idx = i;
            }
        }

        (best_idx, best_cos)
    }
}

fn apply_projection(state: &Array1<f64>, w_proj: &Array1<f64>, state_dim: usize, embed_dim: usize) -> Array1<f64> {
    let mut result = Array1::zeros(embed_dim);
    for j in 0..embed_dim {
        let mut dot = 0.0;
        for i in 0..state_dim {
            dot += state[i] * w_proj[i * embed_dim + j];
        }
        result[j] = dot;
    }
    result
}
