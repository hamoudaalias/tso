/// VAE trainer for GridWorld 5×5 — 25D observations → latent 4D (iso-dim whiskers)
///
/// Architecture: 25→16→4→16→25 (latent 4D = même dimension que les moustaches TSO).

use std::time::Instant;
use ndarray::Array1;
use tso_engine::vae::Vae;

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  VAE TRAINER — GridWorld 5×5                                        ║");
    eprintln!("║  Architecture: 25→16→4→16→25                                         ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    const INPUT_DIM: usize = 25;
    const HIDDEN_DIM: usize = 16;
    const LATENT_DIM: usize = 4;

    // Chargement des frames GridWorld
    eprintln!("Chargement des frames GridWorld...");
    let raw = std::fs::read("../scripts/gridworld_data_10k.bin")
        .expect("scripts/gridworld_data_10k.bin not found");
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    assert_eq!(d, INPUT_DIM, "dim mismatch: expected {INPUT_DIM}, got {d}");
    let data: Vec<Array1<f64>> = raw[8..]
        .chunks(d * 8).take(n)
        .map(|chunk| {
            let vals: Vec<f64> = chunk.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
            Array1::from_vec(vals)
        }).collect();
    eprintln!("  {} frames, dimension {}\n", data.len(), INPUT_DIM);

    // Entraînement
    let mut vae = Vae::new(INPUT_DIM, HIDDEN_DIM, LATENT_DIM);
    let t0 = Instant::now();
    const EPOCHS: usize = 200;

    for epoch in 0..EPOCHS {
        let mut elbo_sum = 0.0;
        for x in &data { elbo_sum += vae.train_step(x, 0.01); }
        if (epoch + 1) % 50 == 0 {
            let (mse_sum, kl_sum) = data.iter().map(|x| {
                vae.encode(x);
                let mu = vae.mu.clone(); let lv = vae.logvar.clone();
                let z: Vec<f64> = mu.iter().map(|m| *m).collect();
                let xr = vae.decode(&z);
                let (_, m, k) = vae.elbo_loss(x, &xr, &mu, &lv);
                (m, k)
            }).fold((0.0, 0.0), |(ms, kl), (m, k)| (ms+m, kl+k));
            let n_ = data.len() as f64;
            eprintln!("  epoch={:3} elbo={:.4} mse={:.7} kl={:.4} [{:.1?}]",
                epoch, elbo_sum/n_, mse_sum/n_, kl_sum/n_, t0.elapsed());
        }
    }

    // Sauvegarde
    let bytes = bincode::serialize(&vae).unwrap();
    let path = "vae_gridworld.bin";
    std::fs::write(path, &bytes).unwrap();
    eprintln!("\n  Weights saved to {}", path);
    eprintln!("  Taille: {} bytes", bytes.len());

    // Test stabilité
    let n_test = data.len().min(200);
    let mut correct = 0usize;
    for i in 0..n_test {
        let x = &data[i];
        vae.encode(x);
        let mu = vae.mu.clone();
        let z: Vec<f64> = mu.iter().map(|m| *m).collect();
        let xr = vae.decode(&z);
        let mse: f64 = x.iter().zip(xr.iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>() / INPUT_DIM as f64;
        if mse < 0.05 { correct += 1; }
    }
    eprintln!("  Stabilité (MSE<0.05): {}/{} = {:.1}%", correct, n_test, correct as f64 / n_test as f64 * 100.0);
}
