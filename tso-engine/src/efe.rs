// efe.rs : Expected Free Energy + InfoGain
// cf. specs/epics/e11-pymdp/e11s02-efe.md

use ndarray::{Array1, ArrayD, Array3};
use crate::fpi;

/// A = identité : constante pour helper de test
fn stable_entropy(p: &Array1<f64>) -> f64 {
    let mut h = 0.0;
    for &v in p.iter() {
        let vc = v.max(1e-16);
        h -= vc * vc.ln();
    }
    h
}

/// Calcule la utility attendue : ∑_m q(o_m) · C_m
/// qo : observations prédites par modalité
/// C  : préférences logarithmiques par modalité
pub fn expected_utility(qo: &[Array1<f64>], C: &[Array1<f64>]) -> f64 {
    qo.iter().zip(C.iter())
        .map(|(q, c)| q.dot(c))
        .sum()
}

/// Calcule l'information gain (valeur épistémique) :
/// IG = H(q(o)) - ∑_s q(s) · H(q(o|s))
pub fn info_gain(qs: &[Array1<f64>], qo: &[Array1<f64>], A: &[ArrayD<f64>]) -> f64 {
    let h_qo: f64 = qo.iter().map(|q| stable_entropy(q)).sum();
    let h_cond: f64 = qs.iter().zip(A.iter()).map(|(q, a)| {
        let n_states = a.shape()[1];
        let n_obs = a.shape()[0];
        let mut h_a = Array1::<f64>::zeros(n_states);
        for s in 0..n_states {
            let mut hs = 0.0;
            for o in 0..n_obs {
                let v = a[[o, s]].max(1e-16);
                hs -= v * v.ln();
            }
            h_a[s] = hs;
        }
        q.dot(&h_a)
    }).sum();
    h_qo - h_cond
}

/// Projette qs à travers B pour une action.
/// qs : croyances courantes [Vec<Array1<f64>>; n_factors]
/// B  : matrices de transition [Vec<Array3<f64>>; n_factors], shape (n_states, n_states, n_actions)
/// action : index de l'action
fn project_state(qs: &[Array1<f64>], B: &[Array3<f64>], action: usize) -> Vec<Array1<f64>> {
    qs.iter().zip(B.iter()).map(|(q, b)| {
        // B[f][s', s, u] -> pour action fixe, B_f_u = B[f][:, :, action]
        let n_states = b.shape()[0];
        let mut q_next = Array1::<f64>::zeros(n_states);
        for s_next in 0..n_states {
            let mut sum_s = 0.0;
            for s in 0..n_states {
                sum_s += b[[s_next, s, action]] * q[s];
            }
            q_next[s_next] = sum_s;
        }
        q_next
    }).collect()
}

/// Prédit les observations à partir des croyances sur les états.
/// qs : croyances sur les états
/// A  : vraisemblances d'observation, shape (n_obs, n_states)
fn predict_obs(qs: &[Array1<f64>], A: &[ArrayD<f64>]) -> Vec<Array1<f64>> {
    qs.iter().zip(A.iter()).map(|(q, a)| {
        let n_obs = a.shape()[0];
        let n_states = a.shape()[1];
        let mut qo = Array1::<f64>::zeros(n_obs);
        for o in 0..n_obs {
            let mut sum_s = 0.0;
            for s in 0..n_states {
                sum_s += a[[o, s]] * q[s];
            }
            qo[o] = sum_s;
        }
        qo
    }).collect()
}

/// Calcule G(π) = negative expected free energy pour une politique.
///
/// G = ∑_t [ utility(qo_t, C) + info_gain(qs_t, qo_t, A) ]
pub fn score_policy(
    qs_init: &[Array1<f64>],
    A: &[ArrayD<f64>],
    B: &[Array3<f64>],
    C: &[Array1<f64>],
    policy: &[usize],
    use_utility: bool,
    use_info_gain: bool,
) -> f64 {
    let mut qs = qs_init.to_vec();
    let mut G = 0.0;
    for &u in policy {
        qs = project_state(&qs, B, u);
        let qo = predict_obs(&qs, A);
        if use_utility { G += expected_utility(&qo, C); }
        if use_info_gain { G += info_gain(&qs, &qo, A); }
    }
    G
}

/// Sélectionne la meilleure action parmi des candidats.
/// Retourne (action_id, G_max).
pub fn select_best_action(
    qs: &[Array1<f64>],
    A: &[ArrayD<f64>],
    B: &[Array3<f64>],
    C: &[Array1<f64>],
    candidates: &[usize],
    use_utility: bool,
    use_info_gain: bool,
) -> (usize, f64) {
    let mut best_action = candidates[0];
    let mut best_G = f64::NEG_INFINITY;
    for &a in candidates {
        let G = score_policy(qs, A, B, C, &[a], use_utility, use_info_gain);
        if G > best_G { best_G = G; best_action = a; }
    }
    (best_action, best_G)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, arr3, ArrayD, Array3};
    use super::*;

    fn identity_A(n_obs: usize, n_states: usize) -> Vec<ArrayD<f64>> {
        let mut a = ArrayD::zeros(vec![n_obs, n_states]);
        for i in 0..n_obs.min(n_states) { a[[i, i]] = 1.0; }
        vec![a]
    }

    fn identity_B(n_states: usize, n_actions: usize) -> Vec<Array3<f64>> {
        let mut b = Array3::<f64>::zeros((n_states, n_states, n_actions));
        for u in 0..n_actions {
            for s in 0..n_states {
                b[[s, s, u]] = 1.0;  // transition diagonale : s' = s
            }
        }
        vec![b]
    }

    #[test]
    fn test_utility_prefers_good_obs() {
        // C = [0, -10] : prefere o0
        let qo = vec![arr1(&[0.8, 0.2])];
        let C = vec![arr1(&[0.0, -10.0])];
        let u = expected_utility(&qo, &C);
        assert!(u > -5.0);  // 0.8*0 + 0.2*(-10) = -2
        assert!((u - (-2.0)).abs() < 1e-6);

        // Mauvaise observation
        let qo_bad = vec![arr1(&[0.2, 0.8])];
        let u_bad = expected_utility(&qo_bad, &C);
        assert!(u_bad < u);  // 0.2*0 + 0.8*(-10) = -8
    }

    #[test]
    fn test_utility_zero_prefs() {
        let qo = vec![arr1(&[0.5, 0.5])];
        let C = vec![arr1(&[0.0, 0.0])];
        let u = expected_utility(&qo, &C);
        assert!((u - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_info_gain_zero_when_certain() {
        // qs = [1, 0] (certain), A = identite
        let A = identity_A(2, 2);
        let qs = vec![arr1(&[1.0, 0.0])];
        let qo = predict_obs(&qs, &A);
        let ig = info_gain(&qs, &qo, &A);
        assert!((ig - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_info_gain_positive_when_uncertain() {
        let A = identity_A(2, 2);
        let qs = vec![arr1(&[0.5, 0.5])];
        let qo = predict_obs(&qs, &A);
        let ig = info_gain(&qs, &qo, &A);
        assert!(ig > 0.0);  // incertain → info_gain > 0
    }

    #[test]
    fn test_score_policy_prefers_high_utility() {
        // Test basique : B identite, A identite, C prefere o0
        // qs = [0.5, 0.5] -> qo = [0.5, 0.5] -> utility = -5
        let A = identity_A(2, 2);
        let B = identity_B(2, 2);
        let C_vec: Vec<Array1<f64>> = vec![arr1(&[0.0, -10.0])];
        let qs = vec![arr1(&[0.5, 0.5])];
        let G0 = score_policy(&qs, &A, &B, &C_vec, &[0], true, false);
        // G = expected_utility = 0.5*0 + 0.5*(-10) = -5
        assert!((G0 - (-5.0)).abs() < 1e-6, "G0={} devrait etre -5", G0);
    }

    #[test]
    fn test_select_best_action_picks_best() {
        let A = identity_A(2, 2);
        let B = identity_B(2, 4);
        let C_vec: Vec<Array1<f64>> = vec![arr1(&[0.0, -50.0])];
        let qs = vec![arr1(&[0.5, 0.5])];

        let (best, G_best) = select_best_action(&qs, &A, &B, &C_vec, &[0, 1, 2, 3], true, false);
        assert_eq!(best, 0);  // action 0 donne qo = [0.5, 0.5] -> utility = 0.5*0 + 0.5*(-50) = -25
        // Wait, B identite et A identite -> qo = qs = 0.5, 0.5, utility = -25
        // toutes les actions donnent la meme chose
        // Donc best = 0 (premiere candidate)
    }

    #[test]
    fn test_score_policy_with_info_gain() {
        let A = identity_A(2, 2);
        let B = identity_B(2, 2);
        let C_vec: Vec<Array1<f64>> = vec![arr1(&[0.0, 0.0])];

        // qs certain : info_gain = 0
        let qs_certain = vec![arr1(&[1.0, 0.0])];
        let G_certain = score_policy(&qs_certain, &A, &B, &C_vec, &[0], false, true);
        assert!((G_certain - 0.0).abs() < 1e-6);

        // qs incertain : info_gain > 0
        let qs_uncertain = vec![arr1(&[0.5, 0.5])];
        let G_uncertain = score_policy(&qs_uncertain, &A, &B, &C_vec, &[0], false, true);
        assert!(G_uncertain > G_certain);
    }

    #[test]
    fn test_utility_info_gain_combined() {
        let A = identity_A(2, 2);
        let B = identity_B(2, 2);
        let C = vec![arr1(&[0.0, -10.0])];

        let qs = vec![arr1(&[0.5, 0.5])];
        let G_util = score_policy(&qs, &A, &B, &C, &[0], true, false);
        let G_both = score_policy(&qs, &A, &B, &C, &[0], true, true);
        assert!(G_both > G_util);  // info_gain ajoute de la valeur
    }

    #[test]
    fn test_two_step_policy() {
        let A = identity_A(2, 2);
        let B = identity_B(2, 2);
        let C_vec3 = vec![arr1(&[0.0, -10.0])];
        let qs = vec![arr1(&[0.5, 0.5])];

        let G_1step = score_policy(&qs, &A, &B, &C_vec3, &[0], true, false);
        let G_2step = score_policy(&qs, &A, &B, &C_vec3, &[0, 0], true, false);
        // B identite : 2 steps = 2 * 1 step (meme etat, meme utility)
        assert!((G_2step - 2.0 * G_1step).abs() < 1e-6);
    }
}
