use ndarray::{Array1, ArrayD};

pub struct FpiConfig {
    pub num_iter: usize,
}

impl Default for FpiConfig {
    fn default() -> Self { FpiConfig { num_iter: 10 } }
}

fn log_stable(x: f64) -> f64 { x.max(1e-16).ln() }

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps = x.mapv(|v| (v - max).exp());
    let sum = exps.sum();
    if sum > 0.0 { exps / sum } else { Array1::ones(x.len()) / x.len() as f64 }
}

/// Calcule le log-likelihood pondere par l'observation pour une modalite.
/// A[m] shape (n_obs, n_states), obs shape (n_obs,)
/// retourne (n_states,) = log P(o|s) marginalise sur l'observation
fn weighted_log_likelihood(a: &ArrayD<f64>, obs: &Array1<f64>) -> Array1<f64> {
    let n_obs = a.shape()[0];
    let n_states = a.shape()[1];
    let mut ll = Array1::<f64>::zeros(n_states);
    for s in 0..n_states {
        let mut sum_o = 0.0;
        for o in 0..n_obs {
            sum_o += obs[o] * log_stable(a[[o, s]]);
        }
        ll[s] = sum_o;
    }
    ll
}

/// run_vanilla_fpi : tous les facteurs partagent la meme modalite
pub fn run_vanilla_fpi(
    A: &[ArrayD<f64>],
    obs: &[Array1<f64>],
    prior: &[Array1<f64>],
    num_iter: usize,
) -> Vec<Array1<f64>> {
    let nf = prior.len();
    // Log-likelihood pondere
    let log_likelihood = weighted_log_likelihood(&A[0], &obs[0]);
    let log_prior: Vec<Array1<f64>> = prior.iter().map(|p| p.mapv(|v| log_stable(v))).collect();
    let mut log_q: Vec<Array1<f64>> = prior.iter().map(|p| Array1::<f64>::zeros(p.len())).collect();
    for _iter in 0..num_iter {
        let q: Vec<Array1<f64>> = log_q.iter().map(|lq| softmax(lq)).collect();
        for f in 0..nf {
            let mll = marginal_log_likelihood(&q, &A[0], f);
            log_q[f] = &log_prior[f] + &(mll + &log_likelihood);
        }
    }
    log_q.iter().map(|lq| softmax(lq)).collect()
}

/// marginal_log_likelihood pour un facteur
pub fn marginal_log_likelihood(
    qs: &[Array1<f64>],
    A: &ArrayD<f64>,
    factor_idx: usize,
) -> Array1<f64> {
    let n_states_a = A.shape()[1];
    let q = &qs[factor_idx];
    // log P(o|q) = sum_s q[s] * log A[o,s]
    // mais ici on a deja log_likelihood -> on calcule sum_s q[s] * ll[s]
    let mut result = Array1::<f64>::zeros(A.shape()[0]);
    for o in 0..A.shape()[0] {
        let mut s_val = 0.0;
        for st in 0..n_states_a {
            s_val += q[st] * log_stable(A[[o, st]]);
        }
        result[o] = s_val;
    }
    result
}

/// run_factorized_fpi : sparse dependencies
pub fn run_factorized_fpi(
    A: &[ArrayD<f64>],
    obs: &[Array1<f64>],
    prior: &[Array1<f64>],
    A_dependencies: &[Vec<usize>],
    num_iter: usize,
) -> Vec<Array1<f64>> {
    let nf = prior.len();
    if nf == 1 { return run_vanilla_fpi(A, obs, prior, num_iter); }
    let log_likelihoods: Vec<Array1<f64>> = A.iter().zip(obs.iter())
        .map(|(a, o)| weighted_log_likelihood(a, o)).collect();
    let log_prior: Vec<Array1<f64>> = prior.iter().map(|p| p.mapv(|v| log_stable(v))).collect();
    let mut log_q: Vec<Array1<f64>> = prior.iter().map(|p| Array1::<f64>::zeros(p.len())).collect();
    for _iter in 0..num_iter {
        let q: Vec<Array1<f64>> = log_q.iter().map(|lq| softmax(lq)).collect();
        for f in 0..nf {
            let mut sum_ll = Array1::<f64>::zeros(prior[f].len());
            for (m, deps) in A_dependencies.iter().enumerate() {
                if deps.contains(&f) {
                    sum_ll = sum_ll + &log_likelihoods[m];
                }
            }
            log_q[f] = &log_prior[f] + &sum_ll;
        }
    }
    log_q.iter().map(|lq| softmax(lq)).collect()
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use super::*;

    #[test]
    fn test_vanilla_fpi_identity() {
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let qs = run_vanilla_fpi(&A, &obs, &prior, 5);
        assert!((qs[0][0] - 1.0).abs() < 1e-6);
        assert!((qs[0][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vanilla_fpi_convergence() {
        let A = vec![arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let qs = run_vanilla_fpi(&A, &obs, &prior, 10);
        assert!((qs[0][0] - 0.9).abs() < 0.15);
        assert!((qs[0][1] - 0.1).abs() < 0.15);
    }

    #[test]
    fn test_factorized_equals_vanilla() {
        let A = vec![arr2(&[[0.8, 0.2], [0.2, 0.8]]).into_dyn()];
        let obs = vec![arr1(&[0.0, 1.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let deps = vec![vec![0]];
        let qs0 = run_vanilla_fpi(&A, &obs, &prior, 10);
        let qs1 = run_factorized_fpi(&A, &obs, &prior, &deps, 10);
        assert!((qs1[0][0] - qs0[0][0]).abs() < 1e-6);
    }

    #[test]
    fn test_qs_sum_to_one() {
        let A = vec![arr2(&[[0.7, 0.3], [0.3, 0.7]]).into_dyn()];
        let obs = vec![arr1(&[0.6, 0.4])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let qs = run_vanilla_fpi(&A, &obs, &prior, 10);
        for q in &qs { assert!((q.sum() - 1.0).abs() < 1e-6); }
    }

    #[test]
    fn test_two_factors_independent() {
        let A = vec![
            arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(),
            arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(),
        ];
        let obs = vec![arr1(&[1.0, 0.0]), arr1(&[0.0, 1.0])];
        let prior = vec![arr1(&[0.5, 0.5]), arr1(&[0.5, 0.5])];
        let deps = vec![vec![0], vec![1]];
        let qs = run_factorized_fpi(&A, &obs, &prior, &deps, 10);
        assert_eq!(qs.len(), 2);
        assert!((qs[0][0] - 1.0).abs() < 1e-6);
        assert!((qs[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_prior_skew() {
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.9, 0.1])];
        let qs = run_vanilla_fpi(&A, &obs, &prior, 5);
        assert!((qs[0][0] - 1.0).abs() < 1e-6);
    }
}
