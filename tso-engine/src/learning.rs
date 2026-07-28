// learning.rs : Dirichlet parameter updates
// cf. specs/epics/e11-pymdp/e11s03-learning.md

use ndarray::{Array1, Array3, ArrayD, IxDyn};

/// Produit tensoriel generalise sur N vecteurs 1D.
/// outer(v0, v1, ..., vN) -> ArrayD de dimensions [len(v0), len(v1), ..., len(vN)]
pub fn multidimensional_outer(arrays: &[&Array1<f64>]) -> ArrayD<f64> {
    if arrays.is_empty() {
        return ArrayD::zeros(IxDyn(&[]));
    }
    if arrays.len() == 1 {
        return arrays[0].clone().into_dyn();
    }
    let dims: Vec<usize> = arrays.iter().map(|a| a.len()).collect();
    let shape = IxDyn(&dims);
    let mut result = ArrayD::<f64>::zeros(shape);

    fn fill(arrays: &[&Array1<f64>], result: &mut ArrayD<f64>, current: &mut Vec<usize>, dim: usize) {
        if dim == arrays.len() {
            let mut val = 1.0;
            for (i, &idx) in current.iter().enumerate() {
                val *= arrays[i][idx];
            }
            result[IxDyn(current)] = val;
            return;
        }
        for i in 0..arrays[dim].len() {
            current.push(i);
            fill(arrays, result, current, dim + 1);
            current.pop();
        }
    }

    let mut current = Vec::new();
    fill(arrays, &mut result, &mut current, 0);
    result
}

/// Valeur attendue du Dirichlet : E[theta] = p / sum(p)
/// Gere les dimensions quelconques (ArrayD).
pub fn dirichlet_expected_value(p: &ArrayD<f64>) -> ArrayD<f64> {
    let sum = p.sum();
    if sum == 0.0 { return ArrayD::zeros(p.raw_dim()); }
    p.mapv(|v| v / sum)
}

/// Met a jour le Dirichlet pour une modalite de A.
///
/// pA_m += lr * outer(obs_m, qs[deps[0]], ..., qs[deps[k]])
/// A_m = dirichlet_expected_value(pA_m)
pub fn update_obs_likelihood_dirichlet_m(
    pA_m: ArrayD<f64>,
    obs_m: &Array1<f64>,
    qs: &[Array1<f64>],
    deps: &[usize],
    lr: f64,
) -> (ArrayD<f64>, ArrayD<f64>) {
    // Construire le produit tensoriel : outer(obs_m, qs[deps[0]], ..., qs[deps[k]])
    let mut outer_inputs: Vec<&Array1<f64>> = vec![obs_m];
    for &d in deps {
        if d < qs.len() {
            outer_inputs.push(&qs[d]);
        }
    }
    let outer = multidimensional_outer(&outer_inputs);

    // pA_m += lr * outer
    let pA_new = if pA_m.raw_dim() == outer.raw_dim() {
        pA_m + outer.mapv(|v| v * lr)
    } else {
        // Si les dimensions different (mismatch), on essaie d'adapter
        // Fallback : juste outer
        outer.mapv(|v| v * lr)
    };

    let A_m = dirichlet_expected_value(&pA_new);
    (pA_new, A_m)
}

/// Met a jour les parametres Dirichlet pour toutes les modalites de A.
pub fn update_obs_likelihood_dirichlet(
    pA: &mut [Option<ArrayD<f64>>],
    A: &mut [ArrayD<f64>],
    obs: &[Array1<f64>],
    qs: &[Array1<f64>],
    deps: &[Vec<usize>],
    lr: f64,
) {
    for m in 0..pA.len() {
        if let Some(ref pA_m) = pA[m] {
            let (pA_new, A_new) = update_obs_likelihood_dirichlet_m(
                pA_m.clone(), &obs[m], qs, &deps[m], lr,
            );
            pA[m] = Some(pA_new);
            A[m] = A_new;
        }
    }
}

/// Met a jour le Dirichlet pour un facteur de B.
///
/// pB_f += lr * outer(qs_t[f], qs_tm1[f], one_hot(action[f]))
/// B_f = dirichlet_expected_value(pB_f)
pub fn update_state_transition_dirichlet_f(
    pB_f: Array3<f64>,
    qs_t: &Array1<f64>,
    qs_tm1: &Array1<f64>,
    action: usize,
    lr: f64,
) -> (Array3<f64>, Array3<f64>) {
    // one_hot de l'action
    let n_actions = pB_f.shape()[2];
    let mut action_onehot = Array1::<f64>::zeros(n_actions);
    if action < n_actions {
        action_onehot[action] = 1.0;
    }

    // outer(qs_t, qs_tm1, action_onehot) -> (n_states, n_states, n_actions)
    let outer_d = multidimensional_outer(&[qs_t, qs_tm1, &action_onehot]);
    // Reshape en 3D pour etre compatible avec pB_f
    let (ns0, ns1, na) = (pB_f.shape()[0], pB_f.shape()[1], pB_f.shape()[2]);
    let outer_3d = outer_d.into_shape((ns0, ns1, na)).unwrap_or_else(|_| pB_f.clone());

    // pB_f += lr * outer
    let pB_new = pB_f + outer_3d.mapv(|v| v * lr);

    // B_f = dirichlet_expected_value(pB_f) — reshape 3D -> 3D
    let sum = pB_new.sum();
    let B_new = if sum > 0.0 {
        pB_new.mapv(|v| v / sum)
    } else {
        pB_new.clone()
    };

    (pB_new, B_new)
}

/// Met a jour les parametres Dirichlet pour tous les facteurs de B.
pub fn update_state_transition_dirichlet(
    pB: &mut [Array3<f64>],
    B: &mut [Array3<f64>],
    qs_t: &[Array1<f64>],
    qs_tm1: &[Array1<f64>],
    actions: &[usize],
    lr: f64,
) {
    for f in 0..pB.len() {
        if f < qs_t.len() && f < qs_tm1.len() && f < actions.len() {
            let (pB_new, B_new) = update_state_transition_dirichlet_f(
                pB[f].clone(), &qs_t[f], &qs_tm1[f], actions[f], lr,
            );
            pB[f] = pB_new;
            B[f] = B_new;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, arr3, Array3};
    use super::*;

    #[test]
    fn test_outer_2d() {
        let v0 = arr1(&[2.0, 3.0]);
        let v1 = arr1(&[5.0, 7.0, 11.0]);
        let outer = multidimensional_outer(&[&v0, &v1]);
        assert_eq!(outer.shape(), &[2, 3]);
        assert!((outer[[0, 0]] - 10.0).abs() < 1e-10);  // 2*5
        assert!((outer[[0, 1]] - 14.0).abs() < 1e-10);  // 2*7
        assert!((outer[[0, 2]] - 22.0).abs() < 1e-10);  // 2*11
        assert!((outer[[1, 0]] - 15.0).abs() < 1e-10);  // 3*5
        assert!((outer[[1, 1]] - 21.0).abs() < 1e-10);  // 3*7
        assert!((outer[[1, 2]] - 33.0).abs() < 1e-10);  // 3*11
    }

    #[test]
    fn test_outer_3d() {
        let v0 = arr1(&[1.0, 2.0]);
        let v1 = arr1(&[3.0, 4.0]);
        let v2 = arr1(&[5.0, 6.0]);
        let outer = multidimensional_outer(&[&v0, &v1, &v2]);
        assert_eq!(outer.shape(), &[2, 2, 2]);
        assert!((outer[[0, 0, 0]] - 15.0).abs() < 1e-10);  // 1*3*5
        assert!((outer[[1, 1, 1]] - 48.0).abs() < 1e-10);  // 2*4*6
    }

    #[test]
    fn test_outer_1d() {
        let v0 = arr1(&[4.0, 5.0, 6.0]);
        let outer = multidimensional_outer(&[&v0]);
        assert_eq!(outer.shape(), &[3]);
        assert!((outer[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_expected_value_simple() {
        let p = arr2(&[[2.0, 1.0], [1.0, 1.0]]).into_dyn();
        let e = dirichlet_expected_value(&p);
        // sum = 5.0
        assert!((e[[0, 0]] - 0.4).abs() < 1e-10);  // 2/5
        assert!((e[[0, 1]] - 0.2).abs() < 1e-10);  // 1/5
        assert!((e[[1, 0]] - 0.2).abs() < 1e-10);  // 1/5
        assert!((e[[1, 1]] - 0.2).abs() < 1e-10);  // 1/5
        assert!((e.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirichlet_expected_value_zeros() {
        let p = ArrayD::zeros(vec![3, 4]);
        let e = dirichlet_expected_value(&p);
        assert_eq!(e.shape(), &[3, 4]);
        assert!((e.sum() - 0.0).abs() < 1e-10);  // sum=0 -> tout zero
    }

    #[test]
    fn test_obs_dirichlet_update() {
        // pA = [[1, 1], [1, 1]] (uniforme)
        // obs = [1, 0] (o0)
        // qs = [[1, 0]] (s0 certain)
        // outer(o, s) = [[1,0],[0,0]]
        // pA_new = [[2,1],[1,1]]
        // A = [[2/4, 1/3], [1/4, 1/3]] = [[0.5, 0.333], [0.25, 0.333]]
        let pA_m = arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn();
        let obs_m = arr1(&[1.0, 0.0]);
        let qs = vec![arr1(&[1.0, 0.0])];
        let (pA_new, A_new) = update_obs_likelihood_dirichlet_m(pA_m, &obs_m, &qs, &[0], 1.0);

        // pA = [[2,1],[1,1]]
        assert!((pA_new[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((pA_new[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((pA_new[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((pA_new[[1, 1]] - 1.0).abs() < 1e-10);

        // sum = 2+1+1+1 = 5, donc A = [[2/5, 1/5], [1/5, 1/5]]
        assert!((A_new[[0, 0]] - 0.4).abs() < 1e-6);
        assert!((A_new[[0, 1]] - 0.2).abs() < 1e-6);
        assert!((A_new[[1, 0]] - 0.2).abs() < 1e-6);
        assert!((A_new[[1, 1]] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_obs_dirichlet_lr_zero() {
        let pA_m = arr2(&[[5.0, 3.0], [2.0, 7.0]]).into_dyn();
        let obs_m = arr1(&[1.0, 0.0]);
        let qs = vec![arr1(&[1.0, 0.0])];
        let (pA_new, _) = update_obs_likelihood_dirichlet_m(pA_m.clone(), &obs_m, &qs, &[0], 0.0);
        // lr=0 -> pA inchange
        for i in 0..2 {
            for j in 0..2 {
                assert!((pA_new[[i, j]] - pA_m[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_trans_dirichlet_update() {
        // pB = zeros(2, 2, 2) initialise a 1.0 partout
        // qs_t = [1, 0] (s0 certain)
        // qs_tm1 = [0, 1] (etait en s1)
        // action = 0
        // outer(qs_t, qs_tm1, onehot(0)) = [[[1,0],[0,0]],[[0,0],[0,0]]]
        // pB_new[0,0,0] = 2, pB_new[0,1,0] = 1, etc.
        let pB_f = Array3::<f64>::from_elem((2, 2, 2), 1.0);
        let qs_t = arr1(&[1.0, 0.0]);
        let qs_tm1 = arr1(&[0.0, 1.0]);
        let (pB_new, _) = update_state_transition_dirichlet_f(pB_f, &qs_t, &qs_tm1, 0, 1.0);

        // outer[0,0,0] = qs_t[0] * qs_tm1[0] * onehot[0] = 1*0*1 = 0
        // outer[0,1,0] = qs_t[0] * qs_tm1[1] * onehot[0] = 1*1*1 = 1
        // outer[1,0,0] = qs_t[1] * qs_tm1[0] * onehot[0] = 0*0*1 = 0
        // outer[1,1,0] = qs_t[1] * qs_tm1[1] * onehot[0] = 0*1*1 = 0
        // pB_new = 1 + 1*outer
        assert!((pB_new[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((pB_new[[0, 1, 0]] - 2.0).abs() < 1e-10);
        assert!((pB_new[[1, 1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_obs_batch_update() {
        let mut pA = vec![Some(arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn())];
        let mut A = vec![arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let qs = vec![arr1(&[1.0, 0.0])];
        let deps = vec![vec![0]];

        update_obs_likelihood_dirichlet(&mut pA, &mut A, &obs, &qs, &deps, 1.0);

        // pA[0] = [[2,1],[1,1]]
        assert!((pA[0].as_ref().unwrap()[[0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_obs_batch_skip_none() {
        let mut pA = vec![None];  // pas de Dirichlet pour cette modalite
        let mut A = vec![arr2(&[[0.5, 0.5], [0.5, 0.5]]).into_dyn()];
        let obs = vec![arr1(&[1.0, 0.0])];
        let qs = vec![arr1(&[1.0, 0.0])];
        let deps = vec![vec![0]];

        update_obs_likelihood_dirichlet(&mut pA, &mut A, &obs, &qs, &deps, 1.0);

        // Rien ne change (pA=None)
        assert!(pA[0].is_none());
    }
}
