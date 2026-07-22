use ndarray::{Array1, Array2};

/// Linear operator trait for matrix-free computations.
pub trait LinearOperator {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, x: &Array1<f64>) -> Array1<f64>;
}

/// Wrapper to make a dense Array2 usable as LinearOperator.
#[allow(dead_code)]
pub struct DenseOp(pub Array2<f64>);

impl LinearOperator for DenseOp {
    fn nrows(&self) -> usize { self.0.nrows() }
    fn ncols(&self) -> usize { self.0.ncols() }
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> { self.0.dot(x) }
}

/// Modified Gram–Schmidt QR decomposition (m ≥ n).
pub fn qr_decomposition(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));
    for j in 0..n {
        let mut v: Array1<f64> = a.column(j).to_owned();
        for i in 0..j {
            r[(i, j)] = q.column(i).dot(&a.column(j));
            let scaled: Array1<f64> = q.column(i).to_owned() * r[(i, j)];
            v = v - scaled;
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-12 {
            r[(j, j)] = norm;
            q.column_mut(j).assign(&(v / norm));
        } else {
            q[(j, j)] = 1.0;
        }
    }
    (q, r)
}

/// Eigendecomposition of a symmetric matrix via Jacobi rotations.
/// Returns (eigenvalues, eigenvectors) with eigenvalues sorted descending.
pub fn eigh_jacobi(mat: &Array2<f64>, tol: f64, max_sweeps: usize) -> (Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    let mut a = mat.clone();
    let mut eigvecs = Array2::eye(n);

    for _ in 0..max_sweeps {
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[(i, j)].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            break;
        }
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let theta = 0.5 * (aqq - app).atan2(2.0 * apq);
        let c = theta.cos();
        let s = theta.sin();
        for i in 0..n {
            let old_ip = a[(i, p)];
            let old_iq = a[(i, q)];
            a[(i, p)] = c * old_ip + s * old_iq;
            a[(i, q)] = -s * old_ip + c * old_iq;
        }
        for j in 0..n {
            let old_pj = a[(p, j)];
            let old_qj = a[(q, j)];
            a[(p, j)] = c * old_pj + s * old_qj;
            a[(q, j)] = -s * old_pj + c * old_qj;
        }
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
        for i in 0..n {
            let old_ip = eigvecs[(i, p)];
            let old_iq = eigvecs[(i, q)];
            eigvecs[(i, p)] = c * old_ip + s * old_iq;
            eigvecs[(i, q)] = -s * old_ip + c * old_iq;
        }
    }

    let mut eigvals = Array1::zeros(n);
    for i in 0..n {
        eigvals[i] = a[(i, i)];
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());
    let sorted_vals: Array1<f64> = indices.iter().map(|&i| eigvals[i]).collect();
    let sorted_vecs = {
        let mut m = Array2::zeros((n, n));
        for (col, &idx) in indices.iter().enumerate() {
            m.column_mut(col).assign(&eigvecs.column(idx));
        }
        m
    };
    (sorted_vals, sorted_vecs)
}

/// SVD of an m×n matrix where m ≤ n (small m).
/// Returns (U (m×m), Σ (m), V^T (m×n)).
pub fn svd_tall(vt: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let g = vt.dot(&vt.t());
    let (eigvals, u) = eigh_jacobi(&g, 1e-12, 100);
    let s: Array1<f64> = eigvals.mapv(|v| v.sqrt().max(0.0));
    let s_inv: Array2<f64> = Array2::from_diag(&s.mapv(|v| if v > 1e-12 { 1.0 / v } else { 0.0 }));
    let vt_final = s_inv.dot(&u.t()).dot(vt);
    (u, s, vt_final)
}

/// Randomized truncated SVD.
/// Returns (U (m×k), Σ (k), V^T (k×n)).
pub fn randomized_svd(
    a: &Array2<f64>,
    k: usize,
    n_oversamples: usize,
    n_power_iter: usize,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let l = (k + n_oversamples).min(m.min(n));

    let omega = Array2::from_shape_fn((n, l), |_| rand::random::<f64>() * 2.0 - 1.0);

    let mut y = a.dot(&omega);
    for _ in 0..n_power_iter {
        let (q, _) = qr_decomposition(&y);
        let a_t_q = a.t().dot(&q);
        let (q2, _) = qr_decomposition(&a_t_q);
        y = a.dot(&q2);
    }
    let (q, _) = qr_decomposition(&y);
    let b = q.t().dot(a);
    let (u_b, s, vt) = svd_tall(&b);
    let u = q.dot(&u_b);
    let kk = k.min(s.len());
    let u_k = u.slice(ndarray::s![.., ..kk]).to_owned();
    let s_k = s.slice(ndarray::s![..kk]).to_owned();
    let vt_k = vt.slice(ndarray::s![..kk, ..]).to_owned();
    (u_k, s_k, vt_k)
}

/// Randomized truncated SVD with a LinearOperator (matrix-free).
/// Returns (U (m×k), Σ (k), V^T (k×n)).
pub fn randomized_svd_op(
    op: &impl LinearOperator,
    k: usize,
    n_oversamples: usize,
    n_power_iter: usize,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = op.nrows();
    let n = op.ncols();
    let l = (k + n_oversamples).min(m.min(n));

    let omega = Array2::from_shape_fn((n, l), |_| rand::random::<f64>() * 2.0 - 1.0);

    // Y = A · Ω  (apply operator to each column of omega)
    let mut y = Array2::zeros((m, l));
    for j in 0..l {
        let col = omega.column(j).to_owned();
        let result = op.apply(&col);
        y.column_mut(j).assign(&result);
    }

    for _ in 0..n_power_iter {
        let (q, _) = qr_decomposition(&y);
        // A^T · Q
        let mut at_q = Array2::zeros((n, l));
        for j in 0..l {
            let col = q.column(j).to_owned();
            let result = op.apply(&col);
            at_q.column_mut(j).assign(&result);
        }
        let (q2, _) = qr_decomposition(&at_q);
        // A · Q2
        for j in 0..l {
            let col = q2.column(j).to_owned();
            let result = op.apply(&col);
            y.column_mut(j).assign(&result);
        }
    }

    let (q, _) = qr_decomposition(&y);
    // B = Q^T · A
    let mut b = Array2::zeros((l, n));
    // B[i, :] = Q[:, i]^T · A
    for i in 0..l {
        let q_col = q.column(i).to_owned();
        // Apply A to q_col, then B[i, :] = result
        let result = op.apply(&q_col);
        b.row_mut(i).assign(&result);
    }

    let (u_b, s, vt) = svd_tall(&b);
    let u = q.dot(&u_b);
    let kk = k.min(s.len());
    let u_k = u.slice(ndarray::s![.., ..kk]).to_owned();
    let s_k = s.slice(ndarray::s![..kk]).to_owned();
    let vt_k = vt.slice(ndarray::s![..kk, ..]).to_owned();
    (u_k, s_k, vt_k)
}

/// Compute word embeddings from the PPMI matrix via truncated SVD.
/// Returns an (n × d) matrix where each row is a word embedding unit-normalized.
pub fn compute_embeddings(ppmi: &Array2<f64>, dim: usize) -> Array2<f64> {
    let n = ppmi.nrows();
    let k = dim.min(n);
    let (u, s, _vt) = randomized_svd(ppmi, k, 5, 2);
    let mut emb = u * &s;
    for mut row in emb.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 1e-12 {
            row /= norm;
        }
    }
    if dim > k {
        let mut padded = Array2::zeros((n, dim));
        padded.slice_mut(ndarray::s![.., ..k]).assign(&emb);
        padded
    } else {
        emb
    }
}
