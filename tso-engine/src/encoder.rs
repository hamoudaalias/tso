/// ════════════════════════════════════════════════════════════════════════════
///  encoder — Trait unifié pour l'encodage perception → concept/latent
///
///  Design C (profond) : une seule méthode requise `encode_raw()`, le résultat
///  unifié `EncodeResult` permet à TsoEngine::step() de toujours faire le même
///  code quelle que soit l'implémentation (AttractorField ou VAE).
///
///  Deux implémentations fournies :
///    - AttractorEncoder : wrapper autour de l'AttractorField existant
///    - VaeEncoder       : encodeur variationnel continu
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use serde::{Serialize, Deserialize};

// ─── Résultat unifié ───────────────────────────────────────────────────────

/// Sortie commune à tous les encodeurs. Permet à step() de faire le même code
/// que l'encodeur soit AttractorField (discret) ou VAE (continu).
#[derive(Clone, Debug)]
pub struct EncodeResult {
    /// Identifiant de la catégorie / concept mappé.
    pub category_id: usize,
    /// Distance / erreur de reconstruction (surprise).
    pub novelty: f64,
    /// Nouvelle catégorie créée ?
    pub is_new: bool,
}

// ─── Métadonnées VAE (optionnelles, seulement pour VaeEncoder) ─────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VaeStats {
    pub mu: Vec<f64>,
    pub logvar: Vec<f64>,
    pub elbo: f64,
    pub kl: f64,
    pub mse: f64,
}

// ─── Trait Encoder ─────────────────────────────────────────────────────────

/// Trait unifié pour l'encodage perception → catégorie.
///
/// Une seule méthode obligatoire : `encode_raw()`.
/// Toutes les autres ont des implémentations par défaut.
pub trait Encoder: Send {
    /// Encode une perception et retourne un résultat unifié.
    fn encode_raw(&mut self, perception: &Array1<f64>) -> EncodeResult;

    /// Nombre de catégories connues (défaut 0).
    fn n_categories(&self) -> usize { 0 }

    /// Prototype vector for a category (None for VAE / continuous encoders).
    /// Used by graph edges, sleep replay.
    fn get_prototype(&self, _category_id: usize) -> Option<&Array1<f64>> { None }

    /// Number of prototype vectors across all categories (for parsimony).
    fn prototype_count(&self) -> usize { 0 }

    /// Adaptation post-encodage (seuil, etc.). Appelée par step() après encode_raw().
    fn adapt(&mut self, _category_id: usize, _novelty: f64) {}

    /// Métadonnées VAE (défaut None — seuls les VAE les fournissent).
    fn vae_stats(&self) -> Option<VaeStats> { None }

    /// Rétropropagation du gradient TD (δ) dans l'encodeur.
    /// Implémentation no-op par défaut (pour les encodeurs sans gradient).
    fn backprop_td(&mut self, _delta: f64) {}
}

// ─── Implémentation : AttractorEncoder (wrapper autour d'AttractorField) ───

/// Wrapper qui adapte l'AttractorField existant au trait Encoder.
/// Garde les seuils adaptatifs, le pruning, et tout le comportement actuel.
pub struct AttractorEncoder {
    pub field: crate::attractor::AttractorField,
    pub novelty_threshold: f64,
    pub concept_novelty_thresholds: Vec<f64>,
    pub concept_local_error: Vec<f64>,
}

impl AttractorEncoder {
    pub fn new(dim: usize) -> Self {
        AttractorEncoder {
            field: crate::attractor::AttractorField::new(dim, 8, 3, 0.01),
            novelty_threshold: 0.15,
            concept_novelty_thresholds: Vec::new(),
            concept_local_error: Vec::new(),
        }
    }

    /// Mécanisme d'adaptation des seuils (déplacé depuis tso_engine.rs).
    fn adapt_novelty_threshold(&mut self, concept_id: usize, dist: f64, is_new: bool) {
        // Étendre les vecteurs si nécessaire
        while self.concept_novelty_thresholds.len() <= concept_id {
            self.concept_novelty_thresholds.push(self.novelty_threshold);
        }
        while self.concept_local_error.len() <= concept_id {
            self.concept_local_error.push(0.0);
        }

        if !is_new {
            let local_dist = dist;
            self.concept_local_error[concept_id] =
                0.9 * self.concept_local_error[concept_id] + 0.1 * local_dist;

            let target_ratio = 0.6;
            let adapt_rate = 0.05;
            let t = self.concept_novelty_thresholds[concept_id].max(1e-8);
            let ratio = self.concept_local_error[concept_id] / t;
            let error = ratio - target_ratio;
            self.concept_novelty_thresholds[concept_id] *= 1.0 - adapt_rate * error;
            self.concept_novelty_thresholds[concept_id] =
                self.concept_novelty_thresholds[concept_id].clamp(0.05, 0.5);
        }
    }
}

impl Encoder for AttractorEncoder {
    fn encode_raw(&mut self, perception: &Array1<f64>) -> EncodeResult {
        let (concept_id, dist) = self.field.predict_with_distance(perception);
        let threshold = self.concept_novelty_thresholds
            .get(concept_id)
            .copied()
            .unwrap_or(self.novelty_threshold);
        let is_new = dist > threshold;
        let category_id = if is_new {
            self.field.add_class(perception)
        } else {
            concept_id
        };

        // Train step du prototype gagnant
        self.adapt_novelty_threshold(category_id, dist, is_new);
        self.field.train_step(perception, category_id);

        EncodeResult {
            category_id,
            novelty: dist,
            is_new,
        }
    }

    fn n_categories(&self) -> usize {
        self.field.n_classes()
    }

    fn get_prototype(&self, category_id: usize) -> Option<&Array1<f64>> {
        self.field.get_prototype(category_id)
    }

    fn prototype_count(&self) -> usize {
        self.field.prototypes.len()
    }

    fn adapt(&mut self, category_id: usize, novelty: f64) {
        // L'adaptation est déjà faite dans encode_raw via adapt_novelty_threshold
        let _ = (category_id, novelty);
    }
}

// ─── Implémentation : VaeEncoder ───────────────────────────────────────────

/// Encodeur variationnel continu. Mappe le latent à un `category_id` via
/// un buffer de centroids (approximation discrète en surface, continue en
/// profondeur). Idéal pour passer de moustaches à vision (pixels → latent).
pub struct VaeEncoder {
    pub vae: crate::vae::Vae,
    /// Centroids : chaque catégorie existante a un latent moyen.
    /// Quand un latent s'écarte trop du centroid le plus proche, nouvelle catégorie.
    pub centroids: Vec<Vec<f64>>,
    pub novelty_threshold: f64,
    /// Mode déterministe : utilise µ au lieu de z = µ + σ·ε.
    /// Active APRÈS pré-entraînement hors ligne pour une stabilité parfaite.
    pub deterministic: bool,
    pub lr: f64,
    pub temperature: f64,
    pub softmax_weights: Vec<f64>,
    pub last_z: Vec<f64>,
    pub last_best_centroid_idx: usize,
    /// Gèle la mise à jour des centroids (true = inférence seule).
    pub freeze: bool,
    /// Mode continu : pas de centroids, encode_raw retourne z comme latent brut.
    pub continuous: bool,
    /// Dernières stats VAE (pour interrogation externe).
    last_stats: Option<VaeStats>,
}

impl VaeEncoder {
    /// Backprop TD error to VAE encoder weights (end-to-end gradient).
    /// Called after reinforce_td with the well-being as scalar signal.
    /// Gradient: dW = -α·δ·(z - centroid[idx]) · z^T
    pub fn backprop_td(&mut self, delta: f64) {
        if self.last_z.is_empty() { return; }
        let idx = self.last_best_centroid_idx;
        if idx >= self.centroids.len() { return; }
        let lr = self.lr * 0.1;
        let t = self.temperature.max(0.1);
        for k in 0..self.last_z.len().min(self.vae.w_mu.len()) {
            let g = -delta * (self.last_z[k] - self.centroids[idx][k]) / t;
            for j in 0..self.vae.w_mu[k].len().min(self.last_z.len()) {
                self.vae.w_mu[k][j] -= lr * g * self.last_z[j];
            }
        }
    }

    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, novelty_threshold: f64) -> Self {
        VaeEncoder {
            vae: crate::vae::Vae::new(input_dim, hidden_dim, latent_dim),
            centroids: Vec::new(),
            novelty_threshold,
            deterministic: false,
            lr: 0.001,
            temperature: 1.0,
            softmax_weights: Vec::new(),
            last_z: Vec::new(),
            last_best_centroid_idx: 0,
            freeze: false,
            continuous: false,
            last_stats: None,
        }
    }

    /// Distance euclidienne entre deux latents.
    fn latent_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}

impl VaeEncoder {
    pub fn anneal_temperature(&mut self, rate: f64) {
        self.temperature = (self.temperature * rate).max(0.1);
    }
}

impl VaeEncoder {
    /// Encode en mode continu : retourne z brut (pas de centroids).
    pub fn encode_continuous(&mut self, perception: &Array1<f64>) -> Vec<f64> {
        self.vae.encode(perception);
        let mu = self.vae.mu.clone();
        let z: Vec<f64> = if self.deterministic { mu } else { self.vae.reparameterize().to_vec() };
        z
    }
}

impl Encoder for VaeEncoder {
    fn encode_raw(&mut self, perception: &Array1<f64>) -> EncodeResult {
        if self.continuous {
            let z = self.encode_continuous(perception);
            return EncodeResult { category_id: 0, novelty: 0.0, is_new: false };
        }
        self.vae.encode(perception);
        let mu = self.vae.mu.clone();
        let logvar = self.vae.logvar.clone();

        // Mode déterministe : z = µ (stable après pré-entraînement)
        // Mode stochastique : z = µ + σ·ε (exploration, entraînement)
        let z: Vec<f64> = if self.deterministic {
            mu.clone()
        } else {
            self.vae.reparameterize().to_vec()
        };

        let x_recon = self.vae.decode(&z);
        let (_elbo, mse, kl) = self.vae.elbo_loss(perception, &x_recon, &mu, &logvar);
        let novelty = mse.sqrt();

        // Premier appel : créer le premier centroid
        if self.centroids.is_empty() {
            self.centroids.push(z);
            self.last_stats = Some(VaeStats { mu, logvar, elbo: mse + kl * 0.001, kl, mse });
            return EncodeResult { category_id: 0, novelty, is_new: true };
        }

        // Softmax straight-through: argmax forward, softmax gradient in centroid update
        let n_cent = self.centroids.len();
        let mut dists = vec![0.0; n_cent];
        for (i, c) in self.centroids.iter().enumerate() {
            dists[i] = Self::latent_dist(&z, c);
        }
        // Gumbel-Softmax weights: softmax over negative distances
        let tau = self.temperature.max(0.01);
        let max_d = dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exps = vec![0.0; n_cent];
        let mut sum_exp = 0.0;
        for (i, d) in dists.iter().enumerate() {
            exps[i] = (-(d - max_d) / tau).exp();
            sum_exp += exps[i];
        }
        let mut softmax_w = vec![0.0; n_cent];
        for (i, e) in exps.iter().enumerate() {
            softmax_w[i] = e / sum_exp;
        }
        self.softmax_weights = softmax_w.clone();

        // Hard assignment (argmin) for category_id
        let best_idx = dists.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        let best_dist = dists[best_idx];

        // Anneal temperature after each call
        self.last_z = z.clone();
        self.last_best_centroid_idx = best_idx;
        self.anneal_temperature(0.995);

        if best_dist > self.novelty_threshold {
            let new_id = self.centroids.len();
            self.centroids.push(z);
            self.last_stats = Some(VaeStats { mu, logvar, elbo: mse + kl * 0.001, kl, mse });
            EncodeResult { category_id: new_id, novelty, is_new: true }
        } else {
            // Soft update via Gumbel weights (straight-through: softmax weights, argmax forward)
            if !self.freeze {
                let rate = 0.1;
                for k in 0..self.vae.latent_dim {
                    let mut update = 0.0;
                    for (i, w) in softmax_w.iter().enumerate() {
                        update += w * (z[k] - self.centroids[i][k]);
                    }
                    self.centroids[best_idx][k] += rate * update;
                }
            }
            self.last_stats = Some(VaeStats { mu, logvar, elbo: mse + kl * 0.001, kl, mse });
            EncodeResult { category_id: best_idx, novelty, is_new: false }
        }
    }

    fn n_categories(&self) -> usize {
        self.centroids.len()
    }

    fn vae_stats(&self) -> Option<VaeStats> {
        self.last_stats.clone()
    }
}
