use ndarray::Array1;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct AttractorField {
    pub prototypes: Vec<Vec<Array1<f64>>>,
    pub lr: f64,
}

impl AttractorField {
    pub fn new(dim: usize, n_classes: usize, k: usize, lr: f64) -> Self {
        let mut prototypes = Vec::new();
        for _ in 0..n_classes {
            let mut class_ps = Vec::new();
            for _ in 0..k {
                let v: Array1<f64> = (0..dim).map(|_| rand::random::<f64>() * 0.5).collect();
                class_ps.push(v);
            }
            prototypes.push(class_ps);
        }
        AttractorField { prototypes, lr }
    }

    fn euclidean_dist(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        (a - b).dot(&(a - b)).sqrt()
    }

    pub fn predict(&self, state: &Array1<f64>) -> usize {
        let mut best_class = 0;
        let mut best_dist = f64::MAX;
        for (c, protos) in self.prototypes.iter().enumerate() {
            for p in protos {
                let d = Self::euclidean_dist(state, p);
                if d < best_dist {
                    best_dist = d;
                    best_class = c;
                }
            }
        }
        best_class
    }

    pub fn train_step(&mut self, state: &Array1<f64>, true_label: usize) {
        let mut dists: Vec<(f64, usize, usize)> = Vec::new();
        for (c, protos) in self.prototypes.iter().enumerate() {
            for (i, p) in protos.iter().enumerate() {
                let d = Self::euclidean_dist(state, p);
                dists.push((d, c, i));
            }
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (best_c, best_k) = (dists[0].1, dists[0].2);
        if best_c == true_label {
            let dir = state - &self.prototypes[best_c][best_k];
            self.prototypes[best_c][best_k] = &self.prototypes[best_c][best_k] + self.lr * dir;
        } else {
            let dir_repel = state - &self.prototypes[best_c][best_k];
            self.prototypes[best_c][best_k] = &self.prototypes[best_c][best_k] - self.lr * dir_repel;

            let mut best_true_dist = f64::MAX;
            let mut best_true_k = 0;
            for (k, p) in self.prototypes[true_label].iter().enumerate() {
                let d = Self::euclidean_dist(state, p);
                if d < best_true_dist {
                    best_true_dist = d;
                    best_true_k = k;
                }
            }
            let dir = state - &self.prototypes[true_label][best_true_k];
            self.prototypes[true_label][best_true_k] = &self.prototypes[true_label][best_true_k] + self.lr * dir;
        }
    }

    pub fn add_class(&mut self, example: &Array1<f64>) -> usize {
        let v = example.clone();
        let c = self.prototypes.len();
        self.prototypes.push(vec![v]);
        c
    }

    pub fn add_prototype(&mut self, example: &Array1<f64>, class: usize) {
        let v = example.clone();
        while self.prototypes.len() <= class {
            self.prototypes.push(Vec::new());
        }
        self.prototypes[class].push(v);
    }

    /// Remove redundant prototypes within each class.
    /// Two prototypes closer than `threshold` are merged: the second is removed.
    /// Each class keeps at least one prototype.
    /// Returns the number of prototypes removed.
    pub fn prune_redundant(&mut self, threshold: f64) -> usize {
        let mut total_removed = 0;
        for class_protos in self.prototypes.iter_mut() {
            if class_protos.len() <= 1 { continue; }
            let mut kept: Vec<Array1<f64>> = Vec::new();
            for proto in class_protos.drain(..) {
                let redundant = kept.iter().any(|kp| Self::euclidean_dist(kp, &proto) < threshold);
                if !redundant || kept.is_empty() {
                    kept.push(proto);
                } else {
                    total_removed += 1;
                }
            }
            *class_protos = kept;
        }
        total_removed
    }

    pub fn n_classes(&self) -> usize {
        self.prototypes.len()
    }

    pub fn predict_with_distance(&self, state: &Array1<f64>) -> (usize, f64) {
        let mut best_class = 0;
        let mut best_dist = f64::MAX;
        for (c, protos) in self.prototypes.iter().enumerate() {
            for p in protos {
                let d = Self::euclidean_dist(state, p);
                if d < best_dist {
                    best_dist = d;
                    best_class = c;
                }
            }
        }
        (best_class, best_dist)
    }

    pub fn get_prototype(&self, class_id: usize) -> Option<&Array1<f64>> {
        self.prototypes.get(class_id).and_then(|protos| protos.first())
    }

    pub fn accuracy(&self, data: &[(Array1<f64>, usize)]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let correct = data.iter().filter(|(s, l)| self.predict(s) == *l).count();
        correct as f64 / data.len() as f64
    }
}
