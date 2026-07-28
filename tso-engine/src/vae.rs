/// ════════════════════════════════════════════════════════════════════════════
///  vae — Variationnal Auto-Encoder sur ndarray
///
///  Encodeur différentiable alternatif à l'AttractorField (seuil discret).
///  Transforme une perception continue en distribution latente gaussienne,
///  permettant la rétropropagation du gradient à travers l'encodeur.
///
///  Architecture : perception → h (tanh) → µ / logσ² → z ~ N(µ, σ²) → h' (tanh) → reconstruction
///  Perte : ELBO = reconstruction MSE + KL(N(µ,σ²) || N(0,1))
///
///  Réutilise le format de poids Vec<Vec<f64>> du Cerebellum.
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use serde::{Serialize, Deserialize};

fn tanh(x: f64) -> f64 { x.tanh() }
#[allow(dead_code)]
fn tanh_deriv(y: f64) -> f64 { 1.0 - y * y }

/// Auto-encodeur variationnel.
#[derive(Serialize, Deserialize)]
pub struct Vae {
    pub dim: usize,         // dimension d'entrée (perception)
    pub hidden_dim: usize,  // dimension cachée
    pub latent_dim: usize,  // dimension latente z

    // Encodeur : entrée → h
    pub w_enc: Vec<Vec<f64>>,  // [hidden_dim × dim]
    pub b_enc: Vec<f64>,       // [hidden_dim]

    // Moyenne et log-variance latente
    pub w_mu: Vec<Vec<f64>>,   // [latent_dim × hidden_dim]
    pub b_mu: Vec<f64>,        // [latent_dim]
    pub w_logvar: Vec<Vec<f64>>, // [latent_dim × hidden_dim]
    pub b_logvar: Vec<f64>,    // [latent_dim]

    // Décodeur : z → reconstruction
    pub w_dec_h: Vec<Vec<f64>>, // [hidden_dim × latent_dim]
    pub b_dec_h: Vec<f64>,      // [hidden_dim]
    pub w_dec: Vec<Vec<f64>>,   // [dim × hidden_dim]
    pub b_dec: Vec<f64>,        // [dim]

    // Cache forward
    pub h_enc: Vec<f64>,  // hidden activations (encodeur)
    pub z: Vec<f64>,      // échantillon latent
    pub mu: Vec<f64>,     // moyenne
    pub logvar: Vec<f64>, // log variance
}

impl Vae {
    /// Crée un VAE avec des poids aléatoires (distribution uniforme [-r, r]).
    pub fn new(dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let r = 0.1;
        fn rand_vec(rows: usize, cols: usize, r: f64) -> Vec<Vec<f64>> {
            (0..rows).map(|_| {
                (0..cols).map(|_| rand::random::<f64>() * 2.0 * r - r).collect()
            }).collect()
        }
        fn rand_bias(n: usize, r: f64) -> Vec<f64> {
            (0..n).map(|_| rand::random::<f64>() * 2.0 * r - r).collect()
        }

        Vae {
            dim, hidden_dim, latent_dim,
            w_enc: rand_vec(hidden_dim, dim, r),
            b_enc: rand_bias(hidden_dim, r),
            w_mu: rand_vec(latent_dim, hidden_dim, r),
            b_mu: rand_bias(latent_dim, r),
            w_logvar: rand_vec(latent_dim, hidden_dim, r),
            b_logvar: rand_bias(latent_dim, r),
            w_dec_h: rand_vec(hidden_dim, latent_dim, r),
            b_dec_h: rand_bias(hidden_dim, r),
            w_dec: rand_vec(dim, hidden_dim, r),
            b_dec: rand_bias(dim, r),
            h_enc: vec![0.0; hidden_dim],
            z: vec![0.0; latent_dim],
            mu: vec![0.0; latent_dim],
            logvar: vec![0.0; latent_dim],
        }
    }

    /// Encode une perception en distribution (µ, logσ²).
    pub fn encode(&mut self, x: &Array1<f64>) -> (&[f64], &[f64]) {
        // hidden layer
        for j in 0..self.hidden_dim {
            let mut s = self.b_enc[j];
            for i in 0..self.dim {
                s += self.w_enc[j][i] * x[i];
            }
            self.h_enc[j] = tanh(s);
        }
        // mu
        for k in 0..self.latent_dim {
            let mut s = self.b_mu[k];
            for j in 0..self.hidden_dim {
                s += self.w_mu[k][j] * self.h_enc[j];
            }
            self.mu[k] = s;
        }
        // logvar
        for k in 0..self.latent_dim {
            let mut s = self.b_logvar[k];
            for j in 0..self.hidden_dim {
                s += self.w_logvar[k][j] * self.h_enc[j];
            }
            self.logvar[k] = s;
        }
        (&self.mu, &self.logvar)
    }

    /// Reparametrization trick : z = µ + σ · ε, ε ~ N(0,1)
    pub fn reparameterize(&mut self) -> &[f64] {
        for k in 0..self.latent_dim {
            let eps: f64 = rand::random::<f64>() * 2.0 - 1.0; // approximation uniforme
            let sigma = (self.logvar[k] * 0.5).exp(); // σ = exp(0.5 · logvar)
            self.z[k] = self.mu[k] + sigma * eps;
        }
        &self.z
    }

    /// Décode un latent z en reconstruction x̂.
    pub fn decode(&self, z: &[f64]) -> Array1<f64> {
        // hidden layer
        let mut h_dec = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut s = self.b_dec_h[j];
            for k in 0..self.latent_dim {
                s += self.w_dec_h[j][k] * z[k];
            }
            h_dec[j] = tanh(s);
        }
        // reconstruction (sortie linéaire)
        let mut out = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut s = self.b_dec[i];
            for j in 0..self.hidden_dim {
                s += self.w_dec[i][j] * h_dec[j];
            }
            out[i] = s;
        }
        Array1::from_vec(out)
    }

    /// Perte ELBO : MSE reconstruction + KL(N(µ,σ²) || N(0,1))
    /// Retourne (elbo, mse, kl).
    /// Prend mu/logvar en paramètres pour éviter les soucis de borrow.
    pub fn elbo_loss(&self, x: &Array1<f64>, x_recon: &Array1<f64>, mu: &[f64], logvar: &[f64]) -> (f64, f64, f64) {
        // MSE reconstruction (par dimension)
        let mse: f64 = (0..self.dim)
            .map(|i| (x[i] - x_recon[i]).powi(2))
            .sum::<f64>() / self.dim as f64;

        // KL divergence : -0.5 · Σ(1 + logvar - mu² - exp(logvar))
        let kl: f64 = (0..self.latent_dim)
            .map(|k| 1.0 + logvar[k] - mu[k].powi(2) - logvar[k].exp())
            .sum::<f64>() * -0.5;

        // ELBO = -MSE + KL / N (on minimise, donc on minimise MSE + beta * KL)
        (mse + kl * 0.001, mse, kl)
    }

    /// Forward complet : encode → reparameterize → decode.
    pub fn forward(&mut self, x: &Array1<f64>) -> Array1<f64> {
        self.encode(x);
        let z = self.reparameterize().to_vec();
        self.decode(&z)
    }

    /// Calcule un gradient simple (approximation SGD) pour une paire (x, x̂).
    /// Version simplifiée : descente de gradient sur MSE uniquement.
    /// Version complète nécessiterait rétropropagation totale (travail futur).
    pub fn train_step(&mut self, x: &Array1<f64>, lr: f64) -> f64 {
        // Copie toutes les sorties avant toute ré-emprunt
        self.encode(x);
        let mu = self.mu.clone();
        let logvar = self.logvar.clone();
        // Termine les traitements qui empruntent self de manière unique
        self.z = (0..self.latent_dim)
            .map(|k| {
                let eps: f64 = rand::random::<f64>() * 2.0 - 1.0;
                let sigma = (logvar[k] * 0.5).exp();
                mu[k] + sigma * eps
            })
            .collect();
        let z = self.z.clone();
        let (x_recon, h_dec) = self.decode_hidden(&z);
        let (elbo, _mse, _kl) = self.elbo_loss(x, &x_recon, &mu, &logvar);

        // Gradient approximé sur w_dec et b_dec (dernière couche)
        for i in 0..self.dim {
            let grad = 2.0 * (x_recon[i] - x[i]); // ∇ MSE
            for j in 0..self.hidden_dim {
                self.w_dec[i][j] -= lr * grad * h_dec[j];
            }
            self.b_dec[i] -= lr * grad;
        }

        elbo
    }

    /// Version interne de decode qui retourne aussi les activations cachées.
    fn decode_hidden(&self, z: &[f64]) -> (Array1<f64>, Vec<f64>) {
        let mut h_dec = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut s = self.b_dec_h[j];
            for k in 0..self.latent_dim {
                s += self.w_dec_h[j][k] * z[k];
            }
            h_dec[j] = tanh(s);
        }
        let mut out = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut s = self.b_dec[i];
            for j in 0..self.hidden_dim {
                s += self.w_dec[i][j] * h_dec[j];
            }
            out[i] = s;
        }
        (Array1::from_vec(out), h_dec)
    }
}
