/// Entraîneur VAE hors ligne sur images 147D (synthétiques pour validation)
///
/// Génère un dataset synthétique 147D (pas de PyO3), entraîne le VAE,
/// sauvegarde les poids. Utilisable ensuite dans TSO via VaeEncoder.

use std::time::Instant;
use ndarray::Array1;
use tso_engine::vae::Vae;

fn collect_dataset(n_frames: usize) -> Vec<Array1<f64>> {
    use rand::Rng;
    let mut data = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut img = vec![0.0; 147];
        // Patterns structurés : cercle + bruit variant
        let cx = 3.5; let cy = 3.5;
        for y in 0..7 { for x in 0..7 {
            let d = (((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2))).sqrt();
            let base = if d < 2.5 { 1.0 - d / 3.0 } else { 0.0 };
            let r = ((x as f64 * 0.7 + y as f64 * 1.3 + i as f64 * 0.1).sin() * 0.5 + 0.5) * 0.2;
            img[y * 7 + x] = (base + r).min(1.0);
            // RGB: propager sur 3 canaux
            let idx = (y * 7 + x) * 3;
            img[idx] = (base + r * 0.3).min(1.0);
            img[idx+1] = (base * 0.8 + r * 0.5).min(1.0);
            img[idx+2] = (base * 0.5 + r * 0.7).min(1.0);
        }}
        data.push(Array1::from_vec(img));
        if (i+1) % 200 == 0 { eprintln!("  generated {} images...", i+1); }
    }
    data
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  VAE TRAINER — Pré-entraînement sur images Minigrid                 ║");
    eprintln!("║  Architecture: 147→32→8→32→147                                        ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    const INPUT_DIM: usize = 147;
    const HIDDEN_DIM: usize = 32;
    const LATENT_DIM: usize = 8;

    // Phase 1: collecte
    eprintln!("Generating 200 images...");
    let data = collect_dataset(200);
    eprintln!(" {} frames", data.len());

    // Phase 2: entraînement
    let mut vae = Vae::new(INPUT_DIM, HIDDEN_DIM, LATENT_DIM);
    let t0 = Instant::now();
    const EPOCHS: usize = 100;

    for epoch in 0..EPOCHS {
        let mut elbo_sum = 0.0;
        for x in &data { elbo_sum += vae.train_step(x, 0.01); }
        if (epoch + 1) % 20 == 0 {
            let (mse_sum, kl_sum) = data.iter().map(|x| {
                vae.encode(x);
                let mu = vae.mu.clone(); let lv = vae.logvar.clone();
                let z: Vec<f64> = mu.iter().map(|m| *m).collect();
                let xr = vae.decode(&z);
                let (_, m, k) = vae.elbo_loss(x, &xr, &mu, &lv);
                (m, k)
            }).fold((0.0, 0.0), |(ms, kl), (m, k)| (ms+m, kl+k));
            let n = data.len() as f64;
            eprintln!("  epoch={:3} elbo={:.4} mse={:.6} kl={:.4} [{:.1?}]",
                epoch, elbo_sum/n, mse_sum/n, kl_sum/n, t0.elapsed());
        }
    }

    // Phase 3: sauvegarde
    let bytes = bincode::serialize(&vae).unwrap();
    let path = "vae_weights.bin";
    std::fs::write(path, &bytes).unwrap();
    eprintln!("  Weights saved to {}", path);
    eprintln!("  Taille: {} bytes", bytes.len());

    // Phase 4: test de stabilité
    eprintln!();
    let n_protos = (data.len() as f64 * 0.1) as usize;
    let mut correct = 0usize;
    for i in 0..n_protos.min(100) {
        let x = &data[i];
        vae.encode(x);
        let mu = vae.mu.clone();
        let z: Vec<f64> = mu.iter().map(|m| *m).collect();
        let xr = vae.decode(&z);
        let mse: f64 = x.iter().zip(xr.iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>() / 147.0;
        if mse < 0.05 { correct += 1; }
    }
    eprintln!("  Stabilité encode→decode (MSE<0.05): {}/{} = {:.1}%",
        correct, n_protos.min(100), correct as f64 / n_protos.min(100) as f64 * 100.0);
}
