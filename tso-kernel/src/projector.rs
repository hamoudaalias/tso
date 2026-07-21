use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;

/// Learned word-to-cluster projection matrix W: V × C.
///
/// Each word maps to a C-dimensional vector of cluster activation strengths.
/// Vectors are updated by R-STDP during pre-training:
///   dΦ < 0 (prediction improves) → strengthen active clusters (+lr_pos × rates)
///   dΦ > 0 (surprise) → weaken active clusters (-lr_neg × rates)
///
/// After training, W encodes contextual semantic preferences:
/// "avocat" near "tribunal" → different cluster activation than near "manger".
pub struct WordProjector {
    n_clusters: usize,
    lr_pos: f64,
    lr_neg: f64,
    /// The projection matrix: word → cluster activation vector
    projections: HashMap<String, Vec<f64>>,
    /// Cached previous Φ for computing dΦ
    prev_phi: Option<f64>,
}

impl WordProjector {
    pub fn new(n_clusters: usize, lr_pos: f64, lr_neg: f64) -> Self {
        Self {
            n_clusters,
            lr_pos,
            lr_neg,
            projections: HashMap::new(),
            prev_phi: None,
        }
    }

    /// Warm start: initialize projections from SVD centroids (cosine similarity).
    pub fn from_svd(
        n_clusters: usize,
        embeddings: &HashMap<String, Vec<f64>>,
        centroids: &[Vec<f64>],
        lr_pos: f64,
        lr_neg: f64,
    ) -> Self {
        let mut proj = HashMap::new();
        for (word, emb) in embeddings {
            let mut act = Vec::with_capacity(n_clusters);
            for c in centroids {
                let cos = cosine_similarity(emb, c);
                act.push(cos.max(0.0)); // rectified: no negative activations
            }
            proj.insert(word.clone(), act);
        }
        Self {
            n_clusters,
            lr_pos,
            lr_neg,
            projections: proj,
            prev_phi: None,
        }
    }

    /// Look up a word's projection vector.
    /// If the word is unknown, initialize with a positive random vector (cold start).
    pub fn lookup(&mut self, word: &str) -> Vec<f64> {
        if let Some(p) = self.projections.get(word) {
            return p.clone();
        }
        // Cold start: strictly positive random projection ∈ [0.1, 1.0]
        // This guarantees LIF neurons fire on first encounter.
        let mut rng = rand::thread_rng();
        let v: Vec<f64> = (0..self.n_clusters)
            .map(|_| 0.1 + rng.gen::<f64>() * 0.9)
            .collect();
        self.projections.insert(word.to_string(), v.clone());
        v
    }

    /// R-STDP update: adjust the word's projection based on Φ change.
    ///
    /// `rates`: the layer output rates (which clusters fired for this word).
    /// `current_phi`: the total inter-layer Φ after processing this word.
    ///
    /// Learning rule:
    ///   dΦ = current_phi - prev_phi
    ///   if dΦ < 0: projection += lr_pos × rates  (reward active clusters)
    ///   if dΦ > 0: projection -= lr_neg × rates  (punish active clusters)
    ///   Then L2-normalize projection.
    pub fn update(&mut self, word: &str, rates: &Array1<f64>, current_phi: f64) {
        let Some(prev) = self.prev_phi else {
            self.prev_phi = Some(current_phi);
            return;
        };
        let dphi = current_phi - prev;
        self.prev_phi = Some(current_phi);

        // Get or create the projection for this word
        let entry = self.projections.entry(word.to_string()).or_insert_with(|| {
            let mut rng = rand::thread_rng();
            (0..self.n_clusters).map(|_| 0.1 + rng.gen::<f64>() * 0.9).collect()
        });

        // R-STDP update
        if dphi < 0.0 {
            // Improvement: strengthen connections to active clusters
            for (e, r) in entry.iter_mut().zip(rates.iter()) {
                *e += self.lr_pos * r;
            }
        } else if dphi > 0.0 {
            // Surprise: weaken connections to active clusters
            for (e, r) in entry.iter_mut().zip(rates.iter()) {
                *e -= self.lr_neg * r;
            }
        }

        // Clamp to non-negative
        for e in entry.iter_mut() {
            if *e < 0.0 {
                *e = 0.0;
            }
        }
    }

    /// Reset the Φ cache (call when starting a new sequence).
    pub fn reset_phi(&mut self) {
        self.prev_phi = None;
    }

    /// Number of learned word projections.
    pub fn len(&self) -> usize {
        self.projections.len()
    }

    /// Returns true if the projector has no learned entries.
    pub fn is_empty(&self) -> bool {
        self.projections.is_empty()
    }

    /// Number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

/// Cosine similarity between two slices.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb < 1e-12 { 0.0 } else { dot / (na * nb) }
}
