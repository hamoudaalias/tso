// inference.rs : bridge FPI pour TSO
// update_posterior_states = point d'entree standard

use ndarray::{Array1, ArrayD};
use crate::fpi::{self, run_vanilla_fpi};

/// Resultat de l'inference
#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub qs: Vec<Array1<f64>>,
    pub concept_id: usize,
    pub vfe: f64,
}

/// Calcule la VFE : F = E_q[log q] - E_q[log P(o|s)] - E_q[log P(s)]
pub fn calc_vfe(qs: &[Array1<f64>], _A: &[ArrayD<f64>], _obs: &[Array1<f64>], prior: &[Array1<f64>]) -> f64 {
    let mut F = 0.0;
    for (q, p) in qs.iter().zip(prior.iter()) {
        for i in 0..q.len() {
            let qi = q[i].max(1e-16);
            let pi = p[i].max(1e-16);
            F += qi * qi.ln() - qi * pi.ln();
        }
    }
    F
}

/// Infer states using FPI, retourne qs + concept_id = argmax du premier facteur
pub fn infer_states(
    A: &[ArrayD<f64>],
    obs: &[Array1<f64>],
    prior: Option<&[Array1<f64>]>,
    num_iter: usize,
) -> InferenceResult {
    // Prior par defaut : uniforme
    let default_prior: Vec<Array1<f64>>;
    let prior_ref = if let Some(p) = prior {
        p
    } else {
        default_prior = obs.iter().map(|o| {
            Array1::from_elem(o.len(), 1.0 / o.len() as f64)
        }).collect();
        &default_prior
    };

    let qs = run_vanilla_fpi(A, obs, prior_ref, num_iter);
    let concept_id = if !qs.is_empty() {
        // argmax du premier facteur
        let q0 = &qs[0];
        q0.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        0
    };
    let vfe = calc_vfe(&qs, A, obs, prior_ref);

    InferenceResult { qs, concept_id, vfe }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use super::*;

    #[test]
    fn test_infer_states_basic() {
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let result = infer_states(&A, &obs, Some(&prior), 5);
        assert_eq!(result.concept_id, 0);  // s0 = concept 0
        assert!((result.qs[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_infer_states_uniform_prior() {
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[0.0, 1.0])];
        let result = infer_states(&A, &obs, None, 5);
        assert_eq!(result.concept_id, 1);  // s1 = concept 1
    }

    #[test]
    fn test_infer_states_certain_to_uncertain() {
        let A = vec![arr2(&[[0.6, 0.4], [0.4, 0.6]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let result = infer_states(&A, &obs, Some(&prior), 10);
        // A est presque uniforme -> qs reste proche de prior
        assert!((result.qs[0][0] - 0.6).abs() < 0.1);
    }

    #[test]
    fn test_calc_vfe_zero_when_certain() {
        // qs = prior = [1,0] -> F = 0
        let qs = vec![arr1(&[1.0, 0.0])];
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[1.0, 0.0])];
        let F = calc_vfe(&qs, &A, &obs, &prior);
        assert!((F - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_calc_vfe_positive_when_divergent() {
        let qs = vec![arr1(&[0.9, 0.1])];
        let A = vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let prior = vec![arr1(&[0.5, 0.5])];
        let F = calc_vfe(&qs, &A, &obs, &prior);
        assert!(F > 0.0);  // divergence avec prior = energie positive
    }
}
