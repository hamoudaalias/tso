use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// CSR sparse matrix for PPMI
// ---------------------------------------------------------------------------
pub struct PPMIMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<u32>,
    pub values: Vec<f64>,
}

impl PPMIMatrix {
    /// Build a PPMI matrix from co‑occurrence counts.
    pub fn from_counts(
        counts: &HashMap<(u32, u32), u64>,
        vocab_size: usize,
    ) -> Self {
        let total: f64 = counts.values().sum::<u64>() as f64;
        let mut row_sum = vec![0.0_f64; vocab_size];
        let mut col_sum = vec![0.0_f64; vocab_size];
        for (&(r, c), &cnt) in counts {
            let v = cnt as f64;
            row_sum[r as usize] += v;
            col_sum[c as usize] += v;
        }

        // Build CSR: collect non-zero PPMI entries per row
        let mut per_row: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vocab_size];
        for (&(r, c), &cnt) in counts {
            let v = cnt as f64;
            let p_ij = v / total;
            let p_i = row_sum[r as usize] / total;
            let p_j = col_sum[c as usize] / total;
            let ppmi = if p_i > 0.0 && p_j > 0.0 {
                (p_ij / (p_i * p_j)).ln().max(0.0)
            } else {
                0.0
            };
            if ppmi > 0.0 {
                per_row[r as usize].push((c, ppmi));
            }
        }

        let mut row_ptr = Vec::with_capacity(vocab_size + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for r in 0..vocab_size {
            for &(c, v) in &per_row[r] {
                col_ind.push(c);
                values.push(v);
            }
            row_ptr.push(col_ind.len());
        }

        Self { rows: vocab_size, cols: vocab_size, row_ptr, col_ind, values }
    }

    /// Sparse matrix‑vector multiply: y = self × x, then overwrite y.
    pub fn matvec(&self, x: &Array1<f64>, y: &mut Array1<f64>) {
        y.fill(0.0);
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut s = 0.0;
            for idx in start..end {
                s += self.values[idx] * x[self.col_ind[idx] as usize];
            }
            y[r] = s;
        }
    }

    /// Transpose sparse matrix‑vector multiply: y = self^T × x.
    pub fn matvec_t(&self, x: &Array1<f64>, y: &mut Array1<f64>) {
        y.fill(0.0);
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let xr = x[r];
            if xr != 0.0 {
                for idx in start..end {
                    y[self.col_ind[idx] as usize] += self.values[idx] * xr;
                }
            }
        }
    }

    /// Sparse matrix‑dense matrix multiply: Y = self × X (X is col‑major).
    /// Y: m × k, X: n × k.
    pub fn matmul_dense(&self, x: &Array2<f64>, y: &mut Array2<f64>) {
        let k = x.shape()[1];
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for col in 0..k {
                let mut s = 0.0;
                for idx in start..end {
                    s += self.values[idx] * x[[self.col_ind[idx] as usize, col]];
                }
                y[[r, col]] = s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QR decomposition via modified Gram‑Schmidt
// ---------------------------------------------------------------------------
pub fn mgs_qr(a: &mut Array2<f64>) -> Array2<f64> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let mut q = Array2::zeros((m, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let r_ij = q.column(i).dot(&a.column(j));
            for k in 0..m {
                v[k] -= q[[k, i]] * r_ij;
            }
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-12 {
            for k in 0..m {
                v[k] /= norm;
            }
        }
        q.column_mut(j).assign(&v);
    }
    q
}

// ---------------------------------------------------------------------------
// Eigendecomposition of a symmetric matrix (Jacobi, for small matrices)
// ---------------------------------------------------------------------------
pub fn symmetric_eigendecompose(
    a: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> (Array1<f64>, Array2<f64>) {
    let n = a.shape()[0];
    let mut v = Array2::eye(n);
    let mut d = Array1::from_vec(a.diag().to_vec());
    let off = |_: &Array1<f64>, a: &Array2<f64>| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                s += a[[i, j]] * a[[i, j]];
            }
        }
        s.sqrt()
    };

    let mut a_cur = a.to_owned();
    for _iter in 0..max_iter {
        if off(&d, &a_cur) < tol {
            break;
        }
        // Find largest off-diagonal entry
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a_cur[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        let tau = (a_cur[[q, q]] - a_cur[[p, p]]) / (2.0 * a_cur[[p, q]]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / ((-tau) + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let d_p = d[p];
        let d_q = d[q];
        d[p] = d_p - t * a_cur[[p, q]];
        d[q] = d_q + t * a_cur[[p, q]];

        let mut old = vec![0.0; n];
        for i in 0..n {
            old[i] = a_cur[[i, p]];
        }
        for i in 0..n {
            let a_ip = old[i];
            let a_iq = a_cur[[i, q]];
            a_cur[[i, p]] = c * a_ip - s * a_iq;
            a_cur[[p, i]] = a_cur[[i, p]];
            a_cur[[i, q]] = s * a_ip + c * a_iq;
            a_cur[[q, i]] = a_cur[[i, q]];
        }
        // Rotate eigenvectors
        for i in 0..n {
            let v_ip = v[[i, p]];
            let v_iq = v[[i, q]];
            v[[i, p]] = c * v_ip - s * v_iq;
            v[[i, q]] = s * v_ip + c * v_iq;
        }
        a_cur[[p, q]] = 0.0;
        a_cur[[q, p]] = 0.0;
    }

    // Sort by descending eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap());
    let sorted_d = Array1::from_iter(indices.iter().map(|&i| d[i]));
    let mut sorted_v = Array2::zeros((n, n));
    for (col_idx, &i) in indices.iter().enumerate() {
        sorted_v.column_mut(col_idx).assign(&v.column(i));
    }

    (sorted_d, sorted_v)
}

// ---------------------------------------------------------------------------
// Randomized SVD
// ---------------------------------------------------------------------------
/// Compute the rank‑k randomized SVD of the PPMI matrix.
///
/// Returns `(row_embeddings, singular_values)` where `row_embeddings` is
/// `vocab_size × k` and each row is the distributional vector of a word.
pub fn randomized_svd(
    matrix: &PPMIMatrix,
    k: usize,
    oversampling: usize,
    power_iter: usize,
    rng_seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let m = matrix.rows;
    let n = matrix.cols;
    let l = k + oversampling; // target dimension for the random projection

    // 1. Random matrix Ω: n × l (uniform [-1, 1])
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut omega = Array2::zeros((n, l));
    for i in 0..n {
        for j in 0..l {
            omega[[i, j]] = rng.gen::<f64>() * 2.0 - 1.0;
        }
    }

    // 2. Y = matrix × Ω  (m × l)
    let mut y = Array2::zeros((m, l));
    matrix.matmul_dense(&omega, &mut y);

    // 3. Power iteration for better accuracy
    for _ in 0..power_iter {
        // Z = matrix^T × Y
        let mut z = Array2::zeros((n, l));
        for col in 0..l {
            let y_col = y.column(col).to_owned();
            let mut z_col = Array1::zeros(n);
            matrix.matvec_t(&y_col, &mut z_col);
            z.column_mut(col).assign(&z_col);
        }
        // Y = matrix × Z
        matrix.matmul_dense(&z, &mut y);
    }

    // 4. QR(Y) → Q (m × l)
    let mut q = mgs_qr(&mut y);

    // 5. B = Q^T × matrix (l × n)
    let mut b = Array2::zeros((l, n));
    for r in 0..m {
        let start = matrix.row_ptr[r];
        let end = matrix.row_ptr[r + 1];
        for idx in start..end {
            let c = matrix.col_ind[idx] as usize;
            let val = matrix.values[idx];
            for i in 0..l {
                b[[i, c]] += q[[r, i]] * val;
            }
        }
    }

    // 6. SVD of small B: B = U_B × Σ × V^T
    //    Compute eigendecomposition of B × B^T (l × l)
    let mut gram = Array2::zeros((l, l));
    for i in 0..l {
        for j in 0..=i {
            let s = b.row(i).dot(&b.row(j));
            gram[[i, j]] = s;
            gram[[j, i]] = s;
        }
    }

    let (eigenvalues, u_b) = symmetric_eigendecompose(&gram, 1000, 1e-10);

    // 7. Singular values = sqrt(eigenvalues)
    let mut s = Array1::zeros(l);
    for i in 0..l {
        s[i] = eigenvalues[i].sqrt();
    }

    // 8. Row embeddings = Q × (U_B × diag(sqrt(S)))
    //    W = U_B[:, :k] × diag(sqrt(S[:k]))  → l × k
    let k_actual = k.min(l);
    let mut w = Array2::zeros((l, k_actual));
    for i in 0..l {
        for j in 0..k_actual {
            w[[i, j]] = u_b[[i, j]] * s[j].sqrt();
        }
    }

    //    E = Q × W  → m × k
    let mut embeddings = Array2::zeros((m, k_actual));
    for r in 0..m {
        for j in 0..k_actual {
            let mut sum = 0.0;
            for i in 0..l {
                sum += q[[r, i]] * w[[i, j]];
            }
            embeddings[[r, j]] = sum;
        }
    }

    let singular_values = s.slice(ndarray::s![..k_actual]).to_owned();

    (embeddings, singular_values)
}

// ---------------------------------------------------------------------------
// Feature extraction from embeddings
// ---------------------------------------------------------------------------
fn barycenter(
    words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
) -> Option<(Vec<f64>, f64)> {
    let dim = words
        .iter()
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())?;
    let mut sum = vec![0.0; dim];
    let mut total_w = 0.0;
    for w in words {
        if let Some(v) = word_embeddings.get(w) {
            for i in 0..dim {
                sum[i] += v[i];
            }
            total_w += 1.0;
        }
    }
    if total_w == 0.0 {
        return None;
    }
    for i in 0..dim {
        sum[i] /= total_w;
    }
    Some((sum, total_w))
}

fn barycenter_tfidf(
    words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
) -> Option<(Vec<f64>, f64)> {
    let dim = words
        .iter()
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())?;
    let mut sum = vec![0.0; dim];
    let mut total_w = 0.0;
    for w in words {
        if let Some(v) = word_embeddings.get(w) {
            let w_idf = idf.get(w).copied().unwrap_or(1.0);
            for i in 0..dim {
                sum[i] += v[i] * w_idf;
            }
            total_w += w_idf;
        }
    }
    if total_w == 0.0 {
        return None;
    }
    for i in 0..dim {
        sum[i] /= total_w;
    }
    Some((sum, total_w))
}

fn bary_features(vp: &[f64], vh: &[f64]) -> [f64; 3] {
    let dim = vp.len();
    let mut dot = 0.0;
    let mut np2 = 0.0;
    let mut nh2 = 0.0;
    for i in 0..dim {
        dot += vp[i] * vh[i];
        np2 += vp[i] * vp[i];
        nh2 += vh[i] * vh[i];
    }
    let np = np2.sqrt();
    let nh = nh2.sqrt();
    let cosine = if np > 0.0 && nh > 0.0 {
        dot / (np * nh)
    } else {
        0.0
    };
    let mut diff_sq = 0.0;
    for i in 0..dim {
        let d = vp[i] - vh[i];
        diff_sq += d * d;
    }
    let euclidean = diff_sq.sqrt();
    let norm_ratio = if np > 0.0 { nh / np } else { 1.0 };
    [cosine, euclidean, norm_ratio]
}

/// Compute 3 distributional features from barycenters of word embeddings.
pub fn distributional_features(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
) -> [f64; 3] {
    let (vp, _) = match barycenter(premise_words, word_embeddings) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    let (vh, _) = match barycenter(hypothesis_words, word_embeddings) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    bary_features(&vp, &vh)
}

/// 3 distributional features with TF-IDF weighted barycenters.
pub fn distributional_features_tfidf(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
) -> [f64; 3] {
    let (vp, _) = match barycenter_tfidf(premise_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    let (vh, _) = match barycenter_tfidf(hypothesis_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    bary_features(&vp, &vh)
}

// ---------------------------------------------------------------------------
// TSO geometric operators — transform hypothesis vectors conditioned on premise
// ---------------------------------------------------------------------------

/// Apply TSO operator to a hypothesis embedding given the best-matching premise word.
///
///   cos >  θ_t → support (identity)
///   cos <  θ_c → contradiction (inversion: -h)
///   otherwise  → tension (project h orthogonal to premise word)
fn tso_transform(h: &[f64], p: &[f64], cos: f64, theta_t: f64, theta_c: f64) -> Vec<f64> {
    if cos > theta_t {
        h.to_vec()
    } else if cos < theta_c {
        h.iter().map(|x| -x).collect()
    } else {
        let mut out = Vec::with_capacity(h.len());
        for i in 0..h.len() {
            out.push(h[i] - cos * p[i]);
        }
        out
    }
}

/// Transform all hypothesis word vectors conditioned on the premise.
///
/// For each hypothesis word, finds the premise word with maximal |cos| and
/// applies the TSO geometric operator.
pub fn transform_hypothesis(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    theta_t: f64,
    theta_c: f64,
) -> Vec<Vec<f64>> {
    let dim = premise_words
        .iter()
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())
        .unwrap_or(0);
    if dim == 0 {
        return hypothesis_words
            .iter()
            .filter_map(|w| word_embeddings.get(w).cloned())
            .collect();
    }

    let p_embs: Vec<&[f64]> = premise_words
        .iter()
        .filter_map(|w| word_embeddings.get(w).map(|v| v.as_slice()))
        .collect();

    if p_embs.is_empty() {
        return hypothesis_words
            .iter()
            .filter_map(|w| word_embeddings.get(w).cloned())
            .collect();
    }

    let mut transformed = Vec::with_capacity(hypothesis_words.len());
    for hw in hypothesis_words {
        let Some(emb_h) = word_embeddings.get(hw) else {
            transformed.push(vec![0.0; dim]);
            continue;
        };

        let mut best_cos = -1.0f64;
        let mut best_p = None;
        for p in &p_embs {
            let mut dot = 0.0;
            let mut np2 = 0.0;
            let mut nh2 = 0.0;
            for i in 0..dim {
                dot += emb_h[i] * p[i];
                np2 += p[i] * p[i];
                nh2 += emb_h[i] * emb_h[i];
            }
            let cos = if np2 > 0.0 && nh2 > 0.0 {
                dot / (np2.sqrt() * nh2.sqrt())
            } else {
                0.0
            };
            if cos.abs() > best_cos.abs() {
                best_cos = cos;
                best_p = Some(*p);
            }
        }

        let new_vec = match best_p {
            Some(p) => tso_transform(emb_h, p, best_cos, theta_t, theta_c),
            None => emb_h.clone(),
        };

        let norm: f64 = new_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            transformed.push(new_vec.iter().map(|x| x / norm).collect());
        } else {
            transformed.push(emb_h.clone());
        }
    }
    transformed
}

/// 3 distributional features using TSO-transformed hypothesis vectors.
///
/// Premise barycenter: standard TF-IDF weighted.
/// Hypothesis barycenter: TF-IDF weighted from TSO-conditioned vectors.
/// Returns [cosine, euclidean, norm_ratio].
pub fn distributional_features_tso(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
    theta_t: f64,
    theta_c: f64,
) -> [f64; 3] {
    let (vp, _) = match barycenter_tfidf(premise_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };

    let t_h = transform_hypothesis(premise_words, hypothesis_words, word_embeddings, theta_t, theta_c);
    if t_h.is_empty() {
        return [0.0, 0.0, 1.0];
    }

    let dim = vp.len();
    let mut vh = vec![0.0; dim];
    let mut total_w = 0.0;
    for (i, hw) in hypothesis_words.iter().enumerate() {
        if i >= t_h.len() {
            break;
        }
        let w_idf = idf.get(hw).copied().unwrap_or(1.0);
        for j in 0..dim {
            vh[j] += t_h[i][j] * w_idf;
        }
        total_w += w_idf;
    }

    if total_w == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    for j in 0..dim {
        vh[j] /= total_w;
    }

    bary_features(&vp, &vh)
}

// ---------------------------------------------------------------------------
// Max-pooling (non-linear pooling, replaces mean-based barycenter)
// ---------------------------------------------------------------------------

/// TF-IDF weighted max over word embeddings (element-wise maximum).
fn barycenter_max_tfidf(
    words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
) -> Option<(Vec<f64>, f64)> {
    let dim = words
        .iter()
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())?;
    let mut max_vec = vec![f64::NEG_INFINITY; dim];
    let mut total_w = 0.0;
    for w in words {
        if let Some(v) = word_embeddings.get(w) {
            let w_idf = idf.get(w).copied().unwrap_or(1.0);
            for i in 0..dim {
                let weighted = v[i] * w_idf;
                if weighted > max_vec[i] {
                    max_vec[i] = weighted;
                }
            }
            total_w += w_idf;
        }
    }
    if total_w == 0.0 {
        return None;
    }
    Some((max_vec, total_w))
}

/// 3 distributional features using max-pooled hypothesis representation.
///
/// Premise: TF-IDF weighted mean (standard).
/// Hypothesis: TF-IDF weighted max (element-wise maximum across words).
/// Returns [cosine, euclidean, norm_ratio] — the max introduces non-linearity.
pub fn distributional_features_max(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
) -> [f64; 3] {
    let (vp, _) = match barycenter_tfidf(premise_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    let (vh, _) = match barycenter_max_tfidf(hypothesis_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    bary_features(&vp, &vh)
}

/// 3 distributional features using max-pooled premise + max-pooled hypothesis.
/// Both sides use max-pooling (fully non-linear).
pub fn distributional_features_max2(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
) -> [f64; 3] {
    let (vp, _) = match barycenter_max_tfidf(premise_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    let (vh, _) = match barycenter_max_tfidf(hypothesis_words, word_embeddings, idf) {
        Some(v) => v,
        None => return [0.0, 0.0, 1.0],
    };
    bary_features(&vp, &vh)
}

/// 4 alignment features: quality of token-level lexical substitution.
///
/// Returns `[align_mean, align_min, align_cov, hyp_len_ratio]`.
pub fn alignment_features(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
) -> [f64; 4] {
    if premise_words.is_empty() || hypothesis_words.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let dim = match premise_words.iter().find_map(|w| word_embeddings.get(w)) {
        Some(v) => v.len(),
        None => return [0.0, 0.0, 0.0, 0.0],
    };
    let zero_vec = vec![0.0; dim];

    // Pre-fetch premise embeddings (OOV → zero vector)
    let p_embs: Vec<&[f64]> = premise_words
        .iter()
        .map(|w| word_embeddings.get(w).map(|v| v.as_slice()).unwrap_or(&zero_vec))
        .collect();

    let mut scores = Vec::with_capacity(hypothesis_words.len());
    for hw in hypothesis_words {
        let emb_h = word_embeddings.get(hw).map(|v| v.as_slice()).unwrap_or(&zero_vec);
        let max_cos = p_embs
            .iter()
            .map(|emb_p| {
                let mut dot = 0.0;
                let mut np2 = 0.0;
                let mut nh2 = 0.0;
                for i in 0..dim {
                    dot += emb_p[i] * emb_h[i];
                    np2 += emb_p[i] * emb_p[i];
                    nh2 += emb_h[i] * emb_h[i];
                }
                let np = np2.sqrt();
                let nh = nh2.sqrt();
                if np > 0.0 && nh > 0.0 {
                    dot / (np * nh)
                } else {
                    0.0
                }
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        scores.push(max_cos);
    }

    if scores.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let n = scores.len() as f64;
    let align_mean = scores.iter().sum::<f64>() / n;
    let align_min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let align_cov = scores.iter().filter(|&&s| s > 0.6).count() as f64 / n;
    let hyp_len_ratio = hypothesis_words.len() as f64 / premise_words.len() as f64;

    [align_mean, align_min, align_cov, hyp_len_ratio]
}

// ---------------------------------------------------------------------------
// Leaky Integrate-and-Fire (LIF) reservoir — temporal processing of sequences
// ---------------------------------------------------------------------------

/// A leaky integrator that processes word embeddings sequentially.
///
/// S_t = α * S_{t-1} + (1-α) * x_t
///
/// With α close to 1, the state decays slowly (long memory).
/// With α close to 0, the state is dominated by recent words (short memory).
pub struct LeakyIntegrator {
    state: Vec<f64>,
    alpha: f64,
}

impl LeakyIntegrator {
    pub fn new(dim: usize, alpha: f64) -> Self {
        Self {
            state: vec![0.0; dim],
            alpha,
        }
    }

    pub fn reset(&mut self) {
        for v in &mut self.state {
            *v = 0.0;
        }
    }

    pub fn process(&mut self, word: &[f64]) {
        for i in 0..self.state.len() {
            self.state[i] = self.alpha * self.state[i] + (1.0 - self.alpha) * word[i];
        }
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

fn process_lif_seq(
    lif: &mut LeakyIntegrator,
    words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
    negation_set: &HashSet<String>,
    dim: usize,
) {
    for w in words {
        if let Some(emb) = word_embeddings.get(w) {
            let w_idf = idf.get(w).copied().unwrap_or(1.0);
            let mut scaled = vec![0.0; dim];
            for i in 0..dim {
                scaled[i] = emb[i] * w_idf;
            }
            lif.process(&scaled);
        }
        if negation_set.contains(w.as_str()) {
            for v in &mut lif.state {
                *v = -*v;
            }
        }
    }
}

/// Compute 3 distributional features using LIF temporal traces.
///
/// Both premise and hypothesis are processed word-by-word through a leaky
/// integrator with IDF-weighted embeddings. When a negation word is
/// encountered (from `negation_set`), the LIF state is inverted (Algorithme 1
/// of the TSO paper — geometric negation operator).
///
/// Returns [cosine, euclidean, norm_ratio].
pub fn distributional_features_lif(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
    alpha: f64,
    negation_set: &HashSet<String>,
) -> [f64; 3] {
    let dim = premise_words
        .iter()
        .chain(hypothesis_words.iter())
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())
        .unwrap_or(0);
    if dim == 0 {
        return [0.0, 0.0, 1.0];
    }

    let mut lif = LeakyIntegrator::new(dim, alpha);

    // Process premise words through LIF (IDF-weighted, with negation inversion)
    process_lif_seq(&mut lif, premise_words, word_embeddings, idf, negation_set, dim);
    let p_state: Vec<f64> = lif.state().to_vec();

    // Reset and process hypothesis words
    lif.reset();
    process_lif_seq(&mut lif, hypothesis_words, word_embeddings, idf, negation_set, dim);
    let h_state: Vec<f64> = lif.state().to_vec();

    bary_features(&p_state, &h_state)
}

/// Compute 6 distributional features using Dual-LIF (slow + fast memory).
///
/// Two leaky integrators run in parallel per sentence:
/// - Slow (α=0.9): captures global context (who does what)
/// - Fast (α=0.5): captures local syntax (negation, word-order)
/// Negation inversion is applied to BOTH states (geometric operator).
///
/// Returns [cos_slow, euc_slow, ratio_slow, cos_fast, euc_fast, ratio_fast].
pub fn distributional_features_duallif(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
    alpha_slow: f64,
    alpha_fast: f64,
    negation_set: &HashSet<String>,
) -> [f64; 6] {
    let dim = premise_words
        .iter()
        .chain(hypothesis_words.iter())
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())
        .unwrap_or(0);
    if dim == 0 {
        return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    }

    let mut lif_slow = LeakyIntegrator::new(dim, alpha_slow);
    let mut lif_fast = LeakyIntegrator::new(dim, alpha_fast);

    // Process premise: update both LIF states, negation inverts both
    for w in premise_words {
        if let Some(emb) = word_embeddings.get(w) {
            let w_idf = idf.get(w).copied().unwrap_or(1.0);
            let mut scaled = vec![0.0; dim];
            for i in 0..dim {
                scaled[i] = emb[i] * w_idf;
            }
            lif_slow.process(&scaled);
            lif_fast.process(&scaled);
        }
        if negation_set.contains(w.as_str()) {
            for v in &mut lif_slow.state { *v = -*v; }
            for v in &mut lif_fast.state { *v = -*v; }
        }
    }
    let p_slow: Vec<f64> = lif_slow.state().to_vec();
    let p_fast: Vec<f64> = lif_fast.state().to_vec();

    // Reset and process hypothesis
    lif_slow.reset();
    lif_fast.reset();
    for w in hypothesis_words {
        if let Some(emb) = word_embeddings.get(w) {
            let w_idf = idf.get(w).copied().unwrap_or(1.0);
            let mut scaled = vec![0.0; dim];
            for i in 0..dim {
                scaled[i] = emb[i] * w_idf;
            }
            lif_slow.process(&scaled);
            lif_fast.process(&scaled);
        }
        if negation_set.contains(w.as_str()) {
            for v in &mut lif_slow.state { *v = -*v; }
            for v in &mut lif_fast.state { *v = -*v; }
        }
    }
    let h_slow: Vec<f64> = lif_slow.state().to_vec();
    let h_fast: Vec<f64> = lif_fast.state().to_vec();

    let [cs, es, rs] = bary_features(&p_slow, &h_slow);
    let [cf, ef, rf] = bary_features(&p_fast, &h_fast);
    [cs, es, rs, cf, ef, rf]
}

/// Compute 4 sequential friction features between hypothesis words and the
/// premise LIF state (with negation inversion). These capture contradiction/
/// support dynamics over time.
///
/// For each hypothesis word h_t, friction φ_t = 1 - cos(h_t, S_premise).
/// Negation words trigger state inversion per Algorithme 1 of the TSO paper.
///
/// Returns [mean_cos, min_cos, max_friction, final_friction].
pub fn phi_sequential(
    premise_words: &[String],
    hypothesis_words: &[String],
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf: &HashMap<String, f64>,
    alpha: f64,
    negation_set: &HashSet<String>,
) -> [f64; 4] {
    let dim = premise_words
        .iter()
        .chain(hypothesis_words.iter())
        .find_map(|w| word_embeddings.get(w))
        .map(|v| v.len())
        .unwrap_or(0);
    if dim == 0 || hypothesis_words.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    // Build premise LIF state (with negation)
    let mut lif = LeakyIntegrator::new(dim, alpha);
    process_lif_seq(&mut lif, premise_words, word_embeddings, idf, negation_set, dim);
    let p_state = lif.state().to_vec();
    let mut p_norm = 0.0;
    for i in 0..dim {
        p_norm += p_state[i] * p_state[i];
    }
    p_norm = p_norm.sqrt();

    // Compute cos for each hypothesis word against premise state
    // Negation words flip the effective premise state (Algorithme 1)
    let mut cos_sum = 0.0;
    let mut cos_min = 1.0;
    let mut last_cos = 0.0;
    let mut n_words = 0;
    let mut sign = 1.0;

    for hw in hypothesis_words {
        if negation_set.contains(hw.as_str()) {
            sign = -sign;
        }
        if let Some(emb) = word_embeddings.get(hw) {
            let mut dot = 0.0;
            let mut nh2 = 0.0;
            for i in 0..dim {
                dot += emb[i] * p_state[i] * sign;
                nh2 += emb[i] * emb[i];
            }
            let nh = nh2.sqrt();
            let cos = if p_norm > 1e-12 && nh > 1e-12 {
                (dot / (p_norm * nh)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            cos_sum += cos;
            if cos < cos_min {
                cos_min = cos;
            }
            last_cos = cos;
            n_words += 1;
        }
    }

    if n_words == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mean_cos = cos_sum / n_words as f64;
    let max_friction = (1.0 - cos_min) / 2.0;
    let final_friction = (1.0 - last_cos) / 2.0;

    [mean_cos, cos_min, max_friction, final_friction]
}
