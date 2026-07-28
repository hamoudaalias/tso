/// ════════════════════════════════════════════════════════════════════════════
///  Livrable : VAE encoder pour images 8×8 (entrée 64D, latent 8D)
///
///  Boucle TSO : chaque step reçoit une image 8×8 (64 pixels) à la place
///  des moustaches. VaeEncoder encode → centroid → category_id.
///
///  Métriques de validation :
///    - MSE reconstruction sur 100 images vues une fois
///    - Stabilité catégorielle (même image → même catégorie ?)
///    - Taux de création de catégories
///    - Perte ELBO (MSE + KL)
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use tso_engine::encoder::{Encoder, VaeEncoder};

const INPUT_DIM: usize = 64;   // 8×8 pixels
const HIDDEN_DIM: usize = 32;
const LATENT_DIM: usize = 8;
const NOVELTY_THRESHOLD: f64 = 0.5;

/// Génère une image 8×8 synthétique (cercle, ligne, bruit structuré).
fn make_image(seed: usize) -> Array1<f64> {
    let mut img = vec![0.0f64; 64];
    match seed % 4 {
        0 => {
            // Cercle au centre
            let cx = 3.5; let cy = 3.5; let r = 2.5;
            for y in 0..8 { for x in 0..8 {
                let d = (((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2))).sqrt();
                img[y * 8 + x] = if d < r { 1.0 - d / r } else { 0.0 };
            }}
        }
        1 => {
            // Barre diagonale
            for y in 0..8 { for x in 0..8 {
                let d = (x as isize - y as isize).abs();
                img[y * 8 + x] = if d < 2 { 0.8 } else { 0.0 };
            }}
        }
        2 => {
            // Carré dans le coin
            for y in 0..4 { for x in 0..4 {
                img[y * 8 + x] = 0.9;
            }}
        }
        3 => {
            // Bruit structuré
            for y in 0..8 { for x in 0..8 {
                let v = ((x * 3 + y * 7) % 10) as f64 / 10.0;
                img[y * 8 + x] = v;
            }}
        }
        _ => {}
    }
    // Ajoute une variante basée sur seed pour créer une famille
    let variant = (seed / 4) as f64 * 0.1;
    for i in 0..64 {
        img[i] = (img[i] + variant * 0.2).min(1.0);
    }
    Array1::from_vec(img)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  LIVRABLE — VAE encoder pour images 8×8 (64D → 8D latent)           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Architecture: 64→32→8→32→64, threshold={:.1}", NOVELTY_THRESHOLD);
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let mut enc = VaeEncoder::new(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NOVELTY_THRESHOLD);
    let mut rng = rand::thread_rng();

    // Mode déterministe + freeze = inférence stable après pré-entraînement
    enc.deterministic = true;
    enc.freeze = true;

    // ── Phase 1 : Entraînement en ligne sur 200 images aléatoires ──────
    eprintln!("--- Phase 1 : Entraînement (200 images, 1 step chacune) ---");
    let mut total_mse = 0.0;
    let mut total_kl = 0.0;

    for step in 0..200 {
        let img = make_image(step);
        let result = enc.encode_raw(&img);
        if let Some(stats) = enc.vae_stats() {
            total_mse += stats.mse;
            total_kl += stats.kl;
        }

        if (step + 1) % 50 == 0 {
            let avg_mse = total_mse / (step + 1) as f64;
            let avg_kl = total_kl / (step + 1) as f64;
            eprintln!("  step={:3} catégories={:3} avg_mse={:.6} avg_kl={:.4} new={}",
                step + 1, enc.n_categories(), avg_mse, avg_kl, if result.is_new { 1 } else { 0 });
        }
    }

    eprintln!();

    // ── Phase 2 : Test de reconstruction ───────────────────────────────
    eprintln!("--- Phase 2 : Reconstruction (100 images) ---");
    let mut recon_errors = Vec::new();
    for i in 0..100 {
        let img = make_image(i);
        let result = enc.encode_raw(&img);
        // Reconstruire via VaeEncoder
        let stats = enc.vae_stats().unwrap_or_else(|| {
            tso_engine::encoder::VaeStats { mu: vec![], logvar: vec![], elbo: 0.0, kl: 0.0, mse: 999.0 }
        });
        recon_errors.push(stats.mse);
    }

    let avg_recon: f64 = recon_errors.iter().sum::<f64>() / recon_errors.len() as f64;
    let min_recon = recon_errors.iter().cloned().fold(f64::MAX, f64::min);
    let max_recon = recon_errors.iter().cloned().fold(f64::MIN, f64::max);
    eprintln!("  MSE  reconstruction  —  moyenne={:.6}  min={:.6}  max={:.6}", avg_recon, min_recon, max_recon);
    eprintln!();

    // ── Phase 3 : Test de stabilité catégorielle ───────────────────────
    eprintln!("--- Phase 3 : Stabilité catégorielle (même image 50×) ---");
    let stable_img = make_image(0);
    let mut cat_hist = std::collections::HashMap::new();
    for _ in 0..50 {
        let result = enc.encode_raw(&stable_img);
        *cat_hist.entry(result.category_id).or_insert(0) += 1;
    }

    let n_distinct = cat_hist.len();
    let max_count = cat_hist.values().cloned().max().unwrap_or(0);
    let stability_pct = max_count as f64 / 50.0 * 100.0;
    eprintln!("  {} catégories distinctes pour 50× la même image", n_distinct);
    eprintln!("  Catégorie majoritaire : {}%", stability_pct);

    // ── Phase 4 : Images types ─────────────────────────────────────────
    eprintln!();
    eprintln!("--- Phase 4 : 4 types d'images, 5 variantes chacune ---");
    for base in 0..4 {
        let first_id = enc.encode_raw(&make_image(base * 4)).category_id;
        let mut same = 0;
        for v in 1..5 {
            let id = enc.encode_raw(&make_image(base * 4 + v)).category_id;
            if id == first_id { same += 1; }
        }
        eprintln!("  Type {} : premier cat={}, {} sur 4 variants dans la même catégorie",
            base, first_id, same);
    }

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  RÉSUMÉ                                                              ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Images 8×8 (64D) → latent 8D → centroids                          ║");
    eprintln!("║  Catégories créées: {}", enc.n_categories());
    eprintln!("║  MSE reconstruction: {:.6}", avg_recon);
    eprintln!("║  Stabilité (50× même image): {} catégories, {}% majoritaire",
        n_distinct, stability_pct as usize);
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
