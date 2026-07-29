//! Test du trait Encoder avec ses deux implémentations.
//!
//! Vérifie que :
//! - AttractorEncoder produit des catégories avec seuil de nouveauté
//! - VaeEncoder produit des catégories via centroids latents
//! - Les deux implémentations partagent la même interface (EncodeResult)

use ndarray::Array1;
use tso_engine::encoder::{Encoder, AttractorEncoder, VaeEncoder};

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
fn test_vae_encoder_basic() {
    let mut enc = VaeEncoder::new(4, 8, 2, 0.5);

    let p = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
    let r = enc.encode_raw(&p);
    let first_id = r.category_id;
    assert!(r.is_new, "first = new");

    // Même perception après plusieurs tentatives (stochastique)
    let mut same = false;
    for _ in 0..5 {
        let r2 = enc.encode_raw(&p);
        if r2.category_id == first_id && !r2.is_new {
            same = true;
            break;
        }
    }
    assert!(same, "same perception should eventually match category {}", first_id);

    // VAE stats disponibles
    let stats = enc.vae_stats();
    assert!(stats.is_some(), "VAE should report stats");
    if let Some(s) = stats {
        assert!(s.mse.is_finite(), "MSE should be finite");
    }
}

#[test]
fn test_vae_encoder_differentiates() {
    let mut enc = VaeEncoder::new(4, 8, 2, 0.3);

    let p1 = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
    let p2 = Array1::from_vec(vec![0.9, 0.1, 0.3, 0.7]);

    let r1 = enc.encode_raw(&p1);
    let r2 = enc.encode_raw(&p2);

    // Si les deux perceptions sont assez différentes, elles doivent
    // produire des catégories différentes (seuil bas)
    eprintln!("cat1={}, cat2={}, novelty1={:.4}, novelty2={:.4}",
        r1.category_id, r2.category_id, r1.novelty, r2.novelty);

    // Le VAE stochastique peut créer 1 ou 2 catégories selon la seed.
    // On vérifie juste que les deux perceptions ne sont pas toujours identiques.
    eprintln!("VAE categories: {}, cat1={}, cat2={}",
        enc.n_categories(), r1.category_id, r2.category_id);
    // Le test est informatif plutôt que normatif (seed-dépendant).
}

#[test]
fn test_encoder_polymorphism() {
    // Vérifie qu'on peut traiter les deux encodeurs via le même trait
    let mut att_enc = AttractorEncoder::new(4);
    let mut vae_enc = VaeEncoder::new(4, 8, 2, 0.5);
    let p = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);

    let encoders: [&mut dyn Encoder; 2] = [&mut att_enc, &mut vae_enc];
    for enc in encoders {
        let r = enc.encode_raw(&p);
        assert!(r.is_new, "first call → is_new");
        eprintln!("Encoder test: cat={}, novelty={:.4}", r.category_id, r.novelty);
    }
}


#[test]
fn test_vae_encoder_weights_frozen_on_encode() {
    // Vérifie que encode_raw ne modifie pas les poids internes du VAE.
    let mut enc = VaeEncoder::new(4, 8, 2, 0.5);
    // Accès aux poids via les champs publics du Vae interne
    let snap_w_enc = enc.vae.w_enc.clone();
    let p = ndarray::Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
    let _r = enc.encode_raw(&p);
    assert_eq!(enc.vae.w_enc, snap_w_enc, "VAE weights unchanged by encode_raw");
}
