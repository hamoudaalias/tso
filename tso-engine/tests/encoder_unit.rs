//! Test du trait Encoder avec AttractorEncoder.
//!
//! Vérifie que :
//! - AttractorEncoder produit des catégories avec seuil de nouveauté
//! - L'interface EncodeResult est respectée

use ndarray::Array1;
use tso_engine::encoder::{Encoder, AttractorEncoder};

#[test]
fn test_attractor_encoder_basic() {
    let mut enc = AttractorEncoder::new(4);

    let p = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
    let r = enc.encode_raw(&p);

    assert!(r.is_new, "first perception = new category");

    let r2 = enc.encode_raw(&p);
    assert!(!r2.is_new, "same perception → not new (should match existing)");

    assert!(enc.n_categories() >= 1, "at least 1 category");
}

#[test]
fn test_encoder_polymorphism() {
    let mut att_enc = AttractorEncoder::new(4);
    let p = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);

    let encoders: [&mut dyn Encoder; 1] = [&mut att_enc];
    for enc in encoders {
        let r = enc.encode_raw(&p);
        assert!(r.is_new, "first call → is_new");
        eprintln!("Encoder test: cat={}, novelty={:.4}", r.category_id, r.novelty);
    }
}
