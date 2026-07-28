// model.rs : Modele generatif A/B/C/D — Design A (struct plate)
// cf. specs/tech-architecture/model-interface-design.md

use ndarray::{Array1, Array3, ArrayD};
use serde::{Serialize, Deserialize};

/// Modele generatif pymdp : A (observation likelihood), B (transition),
/// C (prior preferences), D (prior over states).
///
/// Design A — struct plate : tous les champs sont publics.
/// V1 : pas de methodes, pas d'encapsulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct GenerativeModel {
    /// A[m] = P(o | s) pour chaque modalite m.
    /// Chaque ArrayD a shape (n_obs, n_s0, n_s1, ..., n_sk) selon les facteurs du modele.
    pub A: Vec<ArrayD<f64>>,

    /// B[f] = P(s'_f | s_f, u_f) pour chaque facteur f.
    /// Chaque Array3 a shape (n_states, n_states, n_actions).
    pub B: Vec<Array3<f64>>,

    /// C[m] = log P(o_m) : preferences logarithmiques sur les observations.
    /// Chaque Array1 a shape (n_obs_m,).
    pub C: Vec<Array1<f64>>,

    /// D[f] = P(s_f) : prior initial sur les etats caches.
    /// Chaque Array1 a shape (n_states_f,).
    pub D: Vec<Array1<f64>>,

    /// A_dependencies[m] = indices des facteurs d'etat dont la modalite m depend.
    pub A_dependencies: Vec<Vec<usize>>,

    /// B_dependencies[f] = indices des facteurs d'etat dont le facteur f depend.
    pub B_dependencies: Vec<Vec<usize>>,

    /// Nombre de facteurs d'etat caches.
    pub n_factors: usize,
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, arr3, ArrayD};
    use super::GenerativeModel;

    fn simple_model() -> GenerativeModel {
        GenerativeModel {
            A: vec![arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn()],
            B: vec![ndarray::Array3::from_shape_vec((2, 2, 1), vec![1.0, 0.0, 0.0, 1.0]).unwrap()],
            C: vec![arr1(&[0.0, -10.0])],
            D: vec![arr1(&[0.5, 0.5])],
            A_dependencies: vec![vec![0]],
            B_dependencies: vec![vec![0]],
            n_factors: 1,
        }
    }

    #[test]
    fn test_new_model_has_correct_dimensions() {
        let m = simple_model();
        assert_eq!(m.A.len(), 1);
        assert_eq!(m.B.len(), 1);
        assert_eq!(m.C.len(), 1);
        assert_eq!(m.D.len(), 1);
        assert_eq!(m.n_factors, 1);
    }

    #[test]
    fn test_A_shape_is_obs_x_states() {
        let m = simple_model();
        assert_eq!(m.A[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_B_shape_is_states_x_states_x_actions() {
        let m = simple_model();
        assert_eq!(m.B[0].shape(), &[2, 2, 1]);
    }

    #[test]
    fn test_C_is_obs_preferences() {
        let m = simple_model();
        assert_eq!(m.C[0].shape(), &[2]);
        assert!(m.C[0][0] > m.C[0][1]);  // prefere o0
    }

    #[test]
    fn test_D_is_state_prior() {
        let m = simple_model();
        assert_eq!(m.D[0].shape(), &[2]);
        assert!((m.D[0].sum() - 1.0).abs() < 1e-10);  // somme = 1
    }

    #[test]
    fn test_dependencies_match_factors() {
        let m = simple_model();
        assert_eq!(m.A_dependencies[0], vec![0]);
        assert_eq!(m.B_dependencies[0], vec![0]);
    }

    #[test]
    fn test_2_factors_2_modalities() {
        let m = GenerativeModel {
            A: vec![
                arr2(&[[0.8, 0.2], [0.2, 0.8]]).into_dyn(),
                arr2(&[[0.7, 0.3], [0.3, 0.7]]).into_dyn(),
            ],
            B: vec![
                ndarray::Array3::from_shape_vec((2, 2, 1), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
                ndarray::Array3::from_shape_vec((2, 2, 1), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            ],
            C: vec![arr1(&[0.0, -5.0]), arr1(&[-2.0, 0.0])],
            D: vec![arr1(&[0.5, 0.5]), arr1(&[0.6, 0.4])],
            A_dependencies: vec![vec![0], vec![1]],
            B_dependencies: vec![vec![0], vec![1]],
            n_factors: 2,
        };
        assert_eq!(m.A.len(), 2);
        assert_eq!(m.n_factors, 2);
    }

    #[test]
    fn test_identities_model() {
        let m = GenerativeModel {
            A: vec![arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn()],
            B: vec![arr3(&[[[1.0, 0.0], [0.0, 1.0]]])],
            C: vec![arr1(&[0.0, 0.0])],
            D: vec![arr1(&[1.0, 0.0])],
            A_dependencies: vec![vec![0]],
            B_dependencies: vec![vec![0]],
            n_factors: 1,
        };
        // A = identite : on verifie la diagonale sans .dot() sur ArrayD
        assert!((m.A[0][[0, 0]] - 1.0).abs() < 1e-10);
        assert!((m.A[0][[1, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_roundtrip() {
        use bincode;
        let m = simple_model();
        let encoded = bincode::serialize(&m).unwrap();
        let decoded: GenerativeModel = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.n_factors, m.n_factors);
        assert_eq!(decoded.A.len(), m.A.len());
        assert_eq!(decoded.B.len(), m.B.len());
        assert_eq!(decoded.C.len(), m.C.len());
        assert_eq!(decoded.D.len(), m.D.len());
    }
}
