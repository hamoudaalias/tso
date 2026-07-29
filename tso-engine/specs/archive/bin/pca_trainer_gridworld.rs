/// PCA trainer for GridWorld 5×5 — 25D → 4D latent via SVD
///
/// Calcule les 4 premières composantes principales du dataset GridWorld,
/// exporte (mean_25, components_4×25) en format binaire pour PcaEncoder.

use std::time::Instant;

/// SVD réduit : autovalues de la matrice de covariance via Jacobi.
fn pca_fit(data: &[Vec<f64>], n_components: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = data.len();
    let d = data[0].len();

    // Mean
    let mut mean = vec![0.0; d];
    for row in data { for (j, v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..d { mean[j] /= n as f64; }

    // Covariance matrix (d×d)
    let mut cov = vec![0.0; d * d];
    for row in data {
        for i in 0..d {
            let di = row[i] - mean[i];
            for j in 0..d {
                cov[i * d + j] += di * (row[j] - mean[j]);
            }
        }
    }
    for i in 0..d*d { cov[i] /= (n - 1) as f64; }

    // Jacobi eigenvalue decomposition (max 100 iterations)
    let mut v = vec![0.0; d * d];
    for i in 0..d { v[i * d + i] = 1.0; }
    let mut a = cov.clone();
    let mut changed = true;
    let mut iter = 0;

    while changed && iter < 100 {
        changed = false;
        iter += 1;
        for p in 0..d {
            for q in (p+1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-12 { continue; }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let theta = 0.5 * (aqq - app).atan2(2.0 * apq);
                let c = theta.cos();
                let s = theta.sin();
                // Rotate A
                for i in 0..d {
                    let api = a[p * d + i];
                    let aqi = a[q * d + i];
                    a[p * d + i] = c * api - s * aqi;
                    a[q * d + i] = s * api + c * aqi;
                }
                for i in 0..d {
                    let aip = a[i * d + p];
                    let aiq = a[i * d + q];
                    a[i * d + p] = c * aip - s * aiq;
                    a[i * d + q] = s * aip + c * aiq;
                }
                // Rotate V
                for i in 0..d {
                    let vip = v[i * d + p];
                    let viq = v[i * d + q];
                    v[i * d + p] = c * vip - s * viq;
                    v[i * d + q] = s * vip + c * viq;
                }
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;
                changed = true;
            }
        }
    }

    // Eigenvalues on diagonal of a
    let mut eigen: Vec<(usize, f64)> = (0..d).map(|i| (i, a[i * d + i])).collect();
    eigen.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top n_components eigenvectors
    let mut components = Vec::with_capacity(n_components);
    for k in 0..n_components {
        let idx = eigen[k].0;
        let mut comp = (0..d).map(|i| v[i * d + idx]).collect::<Vec<f64>>();
        // Normalize
        let norm: f64 = comp.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut comp { *x /= norm; }
        }
        components.push(comp);
        eprintln!("  PC{}: eigenvalue={:.4}, variance_ratio={:.2}%",
            k + 1, eigen[k].1, eigen[k].1 / eigen.iter().map(|(_, v)| v).sum::<f64>() * 100.0);
    }

    (mean, components)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PCA TRAINER — GridWorld 5×5  25D → 4D                            ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    const INPUT_DIM: usize = 25;
    const LATENT_DIM: usize = 4;

    // Charger les frames
    let raw = std::fs::read("../scripts/gridworld_data_10k.bin")
        .expect("scripts/gridworld_data_10k.bin not found");
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    assert_eq!(d, INPUT_DIM);
    let data: Vec<Vec<f64>> = raw[8..]
        .chunks(d * 8).take(n)
        .map(|chunk| {
            chunk.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()
        }).collect();
    eprintln!("  {} frames, dimension {}\n", data.len(), INPUT_DIM);

    let t0 = Instant::now();
    let (mean, components) = pca_fit(&data, LATENT_DIM);
    eprintln!("\n  PCA computed in {:.1?}", t0.elapsed());

    // Sauvegarder: mean (25 f64) + components (4×25 = 100 f64)
    let mut buf = Vec::with_capacity((INPUT_DIM + LATENT_DIM * INPUT_DIM) * 8);
    for v in &mean { buf.extend_from_slice(&v.to_le_bytes()); }
    for comp in &components {
        for v in comp { buf.extend_from_slice(&v.to_le_bytes()); }
    }
    let path = "pca_gridworld.bin";
    std::fs::write(path, &buf).unwrap();
    eprintln!("  Saved → {} ({} bytes)", path, buf.len());

    // Test reconstruction error
    let mut mse_sum = 0.0;
    for row in data.iter().take(1000) {
        // Project
        let mut latent = vec![0.0; LATENT_DIM];
        for k in 0..LATENT_DIM {
            for j in 0..INPUT_DIM {
                latent[k] += components[k][j] * (row[j] - mean[j]);
            }
        }
        // Reconstruct
        let mut recon = mean.clone();
        for j in 0..INPUT_DIM {
            for k in 0..LATENT_DIM {
                recon[j] += latent[k] * components[k][j];
            }
        }
        let mse: f64 = row.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / INPUT_DIM as f64;
        mse_sum += mse;
    }
    eprintln!("  Reconstruction MSE (1000 samples): {:.7}", mse_sum / 1000.0);
}
