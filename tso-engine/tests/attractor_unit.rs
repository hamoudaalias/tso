use ndarray::Array1;
use tso_engine::attractor::AttractorField;

fn e(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

#[test]
fn test_new_creates_prototypes() {
    let af = AttractorField::new(4, 3, 2, 0.01);
    assert_eq!(af.n_classes(), 3);
    assert_eq!(af.prototypes.len(), 3);
    for p in &af.prototypes {
        assert_eq!(p.len(), 2);
        for v in p {
            assert_eq!(v.len(), 4);
        }
    }
    assert!(e(af.lr, 0.01));
}

#[test]
fn test_predict_returns_valid_class() {
    let af = AttractorField::new(4, 3, 2, 0.01);
    for _ in 0..10 {
        let state = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let cid = af.predict(&state);
        assert!(cid < 3);
    }
}

#[test]
fn test_train_reduces_distance_to_same_class() {
    let mut af = AttractorField::new(2, 2, 1, 0.5);
    let state = Array1::from_vec(vec![0.9, 0.9]);
    let d_before = {
        let p = &af.prototypes[0][0];
        (state.clone() - p).dot(&(state.clone() - p)).sqrt()
    };
    for _ in 0..50 {
        af.train_step(&state, 0);
    }
    let d_after = {
        let p = &af.prototypes[0][0];
        (state.clone() - p).dot(&(state.clone() - p)).sqrt()
    };
    assert!(d_after < d_before, "d_after={} should be < d_before={}", d_after, d_before);
}

#[test]
fn test_add_class_returns_new_id() {
    let mut af = AttractorField::new(4, 2, 1, 0.01);
    let example = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let cid = af.add_class(&example);
    assert_eq!(cid, 2);
    assert_eq!(af.n_classes(), 3);
}

#[test]
fn test_add_prototype_to_existing_class() {
    let mut af = AttractorField::new(4, 1, 1, 0.01);
    let example = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    af.add_prototype(&example, 0);
    assert_eq!(af.prototypes[0].len(), 2);
}

#[test]
fn test_add_prototype_to_new_class_extends() {
    let mut af = AttractorField::new(4, 1, 1, 0.01);
    let example = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    af.add_prototype(&example, 5);
    assert_eq!(af.n_classes(), 6);
}

#[test]
fn test_prune_redundant_removes_close_prototypes() {
    let mut af = AttractorField::new(2, 1, 3, 0.01);
    let close = Array1::from_vec(vec![0.5, 0.5]);
    af.prototypes[0] = vec![
        close.clone(),
        Array1::from_vec(vec![0.51, 0.51]),
        Array1::from_vec(vec![0.9, 0.9]),
    ];
    let removed = af.prune_redundant(0.1);
    assert!(removed > 0);
    assert_eq!(af.prototypes[0].len(), 2);
}

#[test]
fn test_prune_redundant_keeps_at_least_one() {
    let mut af = AttractorField::new(2, 1, 3, 0.01);
    let v = Array1::from_vec(vec![0.5, 0.5]);
    af.prototypes[0] = vec![v.clone(), Array1::from_vec(vec![0.5001, 0.5001])];
    af.prune_redundant(10.0);
    assert_eq!(af.prototypes[0].len(), 1);
}

#[test]
fn test_predict_with_distance() {
    let af = AttractorField::new(2, 2, 1, 0.01);
    let state = Array1::from_vec(vec![0.5, 0.5]);
    let (cid, dist) = af.predict_with_distance(&state);
    assert!(cid < 2);
    assert!(dist >= 0.0);
}

#[test]
fn test_get_prototype_exists() {
    let af = AttractorField::new(4, 2, 1, 0.01);
    let p = af.get_prototype(0);
    assert!(p.is_some());
    assert_eq!(p.unwrap().len(), 4);
}

#[test]
fn test_get_prototype_nonexistent() {
    let af: AttractorField = AttractorField::new(4, 2, 1, 0.01);
    assert!(af.get_prototype(99).is_none());
}

#[test]
fn test_accuracy_on_trained_data() {
    let mut af = AttractorField::new(2, 2, 1, 0.5);
    let d0 = Array1::from_vec(vec![0.9, 0.9]);
    let d1 = Array1::from_vec(vec![0.1, 0.1]);
    for _ in 0..100 {
        af.train_step(&d0, 0);
        af.train_step(&d1, 1);
    }
    let data = vec![(d0, 0usize), (d1, 1usize)];
    let acc = af.accuracy(&data);
    assert!(acc >= 0.5, "acc={}", acc);
}

#[test]
fn test_accuracy_empty_returns_zero() {
    let af = AttractorField::new(4, 2, 1, 0.01);
    assert!(e(af.accuracy(&[]), 0.0));
}

#[test]
fn test_predict_returns_closest_class() {
    let mut af = AttractorField::new(2, 2, 1, 0.5);
    af.prototypes[0] = vec![Array1::from_vec(vec![0.9, 0.9])];
    af.prototypes[1] = vec![Array1::from_vec(vec![0.0, 0.0])];
    let near_class0 = Array1::from_vec(vec![0.8, 0.8]);
    let near_class1 = Array1::from_vec(vec![0.1, 0.0]);
    assert_eq!(af.predict(&near_class0), 0);
    assert_eq!(af.predict(&near_class1), 1);
}
