use ndarray::Array1;

#[derive(Clone)]
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
                let mut v: Array1<f64> = (0..dim).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
                let n = v.dot(&v).sqrt().max(1e-12);
                v /= n;
                class_ps.push(v);
            }
            prototypes.push(class_ps);
        }
        AttractorField { prototypes, lr }
    }

    fn cosine_dist(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let dot = a.dot(b);
        let na = a.dot(a).sqrt().max(1e-12);
        let nb = b.dot(b).sqrt().max(1e-12);
        1.0 - (dot / (na * nb))
    }

    pub fn predict(&self, state: &Array1<f64>) -> usize {
        let mut best_class = 0;
        let mut best_dist = f64::MAX;
        for (c, protos) in self.prototypes.iter().enumerate() {
            for p in protos {
                let d = Self::cosine_dist(state, p);
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
                let d = Self::cosine_dist(state, p);
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

            for &(_, c, k) in &dists[..2] {
                if c == true_label {
                    let dir = state - &self.prototypes[c][k];
                    self.prototypes[c][k] = &self.prototypes[c][k] + self.lr * dir;
                    break;
                }
            }
        }
    }

    /// One-shot: add a prototype from a single example.
    /// Returns the new class index.
    pub fn add_class(&mut self, example: &Array1<f64>) -> usize {
        let mut v = example.clone();
        let n = v.dot(&v).sqrt().max(1e-12);
        v /= n;
        let c = self.prototypes.len();
        self.prototypes.push(vec![v]);
        c
    }

    /// Add a prototype to an existing class (for multi-prototype refinement).
    pub fn add_prototype(&mut self, example: &Array1<f64>, class: usize) {
        let mut v = example.clone();
        let n = v.dot(&v).sqrt().max(1e-12);
        v /= n;
        while self.prototypes.len() <= class {
            self.prototypes.push(Vec::new());
        }
        self.prototypes[class].push(v);
    }

    pub fn n_classes(&self) -> usize {
        self.prototypes.len()
    }

    pub fn predict_with_distance(&self, state: &Array1<f64>) -> (usize, f64) {
        let mut best_class = 0;
        let mut best_dist = f64::MAX;
        for (c, protos) in self.prototypes.iter().enumerate() {
            for p in protos {
                let d = Self::cosine_dist(state, p);
                if d < best_dist {
                    best_dist = d;
                    best_class = c;
                }
            }
        }
        (best_class, best_dist)
    }

    pub fn accuracy(&self, data: &[(Array1<f64>, usize)]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let correct = data.iter().filter(|(s, l)| self.predict(s) == *l).count();
        correct as f64 / data.len() as f64
    }
}
