/// ════════════════════════════════════════════════════════════════════════════
///  encoder — Trait unifié pour l'encodage perception → concept
///
///  Une seule méthode requise `encode_raw()`, le résultat unifié `EncodeResult`
///  permet à TsoEngine::step() de faire le même code quelle que soit
///  l'implémentation.
///
///  Implémentation fournie :
///    - AttractorEncoder : wrapper autour de l'AttractorField existant
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;

// ─── Résultat unifié ───────────────────────────────────────────────────────

/// Sortie commune à tous les encodeurs.
#[derive(Clone, Debug)]
pub struct EncodeResult {
    /// Identifiant de la catégorie / concept mappé.
    pub category_id: usize,
    /// Distance / erreur de reconstruction (surprise).
    pub novelty: f64,
    /// Nouvelle catégorie créée ?
    pub is_new: bool,
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

    /// Prototype vector for a category.
    /// Used by graph edges, sleep replay.
    fn get_prototype(&self, _category_id: usize) -> Option<&Array1<f64>> { None }

    /// Number of prototype vectors across all categories (for parsimony).
    fn prototype_count(&self) -> usize { 0 }

    /// Adaptation post-encodage (seuil, etc.). Appelée par step() après encode_raw().
    fn adapt(&mut self, _category_id: usize, _novelty: f64) {}

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


