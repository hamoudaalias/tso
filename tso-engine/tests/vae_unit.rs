//! Tests du VAE (Variational Auto-Encoder) sur ndarray.
//!
//! Vérifie :
//! - encode → reparameterize → decode cycle
//! - ELBO loss (MSE + KL) est calculable et finie
//! - train_step réduit la loss sur des données synthétiques

use ndarray::Array1;
use tso_engine::vae::Vae;

#[test]
fn test_vae_forward_cycle() {
    let mut vae = Vae::new(4, 8, 2);
    let x = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);

    let x_recon = vae.forward(&x);

    assert_eq!(x_recon.len(), 4, "reconstruction should have same dim");
    assert!(x_recon[0].is_finite(), "reconstructed value should be finite");
}

#[test]
fn test_vae_elbo_loss_is_finite() {
    let mut vae = Vae::new(4, 8, 2);
    let x = Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);

    vae.encode(&x);
    let mu = vae.mu.clone();
    let logvar = vae.logvar.clone();
    let z = vae.reparameterize().to_vec();
    let x_recon = vae.decode(&z);

    let (elbo, mse, kl) = vae.elbo_loss(&x, &x_recon, &mu, &logvar);

    assert!(elbo.is_finite());
    assert!(mse >= 0.0);
    assert!(kl >= -1e-6);
    eprintln!("elbo={:.6}, mse={:.6}, kl={:.6}", elbo, mse, kl);
}

#[test]
fn test_vae_training_reduces_loss() {
    let mut vae = Vae::new(4, 8, 2);

    // Données synthétiques : 3 points fixes
    let data: Vec<Array1<f64>> = vec![
        Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]),
        Array1::from_vec(vec![0.1, 0.3, 0.9, 0.2]),
        Array1::from_vec(vec![0.7, 0.2, 0.4, 0.6]),
    ];

    let loss_before: f64 = data.iter()
        .map(|x| { vae.forward(x); vae.train_step(x, 0.0) })
        .sum();

    // 50 pas d'entraînement
    for _step in 0..50 {
        for x in &data {
            vae.train_step(x, 0.01);
        }
    }

    let loss_after: f64 = data.iter()
        .map(|x| { vae.forward(x); vae.train_step(x, 0.0) })
        .sum();

    eprintln!("loss before: {:.6}, after: {:.6}", loss_before, loss_after);

    // La loss doit baisser (entraînement simple)
    assert!(
        loss_after <= loss_before + 1e-6,
        "loss increased: {:.6} → {:.6}", loss_before, loss_after
    );
}


#[test]
fn test_vae_weights_frozen_without_train_step() {
    // Vérifie que forward seul ne modifie pas les poids du VAE.
    let mut vae = Vae::new(4, 8, 2);
    let snap = (
        vae.w_enc.clone(), vae.b_enc.clone(),
        vae.w_mu.clone(), vae.b_mu.clone(),
        vae.w_logvar.clone(), vae.b_logvar.clone(),
    );
    let x = ndarray::Array1::from_vec(vec![0.2, 0.5, 0.8, 0.1]);
    let _x_recon = vae.forward(&x);
    assert_eq!(vae.w_enc, snap.0, "w_enc unchanged by forward");
    assert_eq!(vae.b_enc, snap.1, "b_enc unchanged by forward");
    assert_eq!(vae.w_mu, snap.2, "w_mu unchanged by forward");
    assert_eq!(vae.b_mu, snap.3, "b_mu unchanged by forward");
    assert_eq!(vae.w_logvar, snap.4, "w_logvar unchanged by forward");
    assert_eq!(vae.b_logvar, snap.5, "b_logvar unchanged by forward");
}
