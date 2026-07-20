use ndarray::Array1;
use std::collections::HashMap;

#[derive(Clone)]
pub struct SharpAttractorField {
    pub attractors: HashMap<i32, Array1<f64>>,
    pub contrast: Option<Array1<f64>>,
    pub contrasted: HashMap<i32, Array1<f64>>,
    pub power: i32,
}

impl SharpAttractorField {
    pub fn new(power: i32) -> Self {
        Self {
            attractors: HashMap::new(),
            contrast: None,
            contrasted: HashMap::new(),
            power,
        }
    }

    pub fn fit(&mut self, x: &[Array1<f64>], y: &[i32]) {
        // Compute centroids per class
        let mut sums: HashMap<i32, Array1<f64>> = HashMap::new();
        let mut counts: HashMap<i32, usize> = HashMap::new();
        let dim = if x.is_empty() { 3 } else { x[0].len() };

        for (xi, yi) in x.iter().zip(y.iter()) {
            let entry = sums.entry(*yi).or_insert_with(|| Array1::zeros(dim));
            *entry += xi;
            *counts.entry(*yi).or_insert(0) += 1;
        }

        for label in [0, 1, 2] {
            let centroid = if let (Some(s), Some(c)) = (sums.get(&label), counts.get(&label)) {
                s / *c as f64
            } else {
                Array1::from_vec(vec![0.5; dim])
            };
            self.attractors.insert(label, centroid);
        }

        // Contrast = background noise = mean of attractors
        let all_attr: Vec<&Array1<f64>> = [0, 1, 2]
            .iter()
            .filter_map(|l| self.attractors.get(l))
            .collect();
        if !all_attr.is_empty() {
            let n = all_attr.len();
            let mut bg = Array1::zeros(dim);
            for a in all_attr {
                bg += a;
            }
            bg /= n as f64;
            self.contrast = Some(bg);

            // Contrasted attractors
            let bg = self.contrast.as_ref().unwrap();
            for (label, attr) in &self.attractors {
                self.contrasted.insert(*label, attr - bg);
            }
        }
    }

    fn affinity(&self, x: &Array1<f64>) -> HashMap<i32, f64> {
        let mut scores = HashMap::new();
        for (label, attr) in &self.attractors {
            let d = (x - attr).mapv(|v| v.powi(2)).sum().sqrt();
            scores.insert(*label, d);
        }
        scores
    }

    pub fn predict(&self, x: &[Array1<f64>]) -> Vec<i32> {
        x.iter()
            .map(|xi| {
                let scores = self.affinity(xi);
                scores
                    .into_iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(label, _)| label)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn predict_with_scores(&self, x: &[Array1<f64>]) -> Vec<HashMap<String, f64>> {
        x.iter()
            .map(|xi| {
                let scores = self.affinity(xi);
                let mut named = HashMap::new();
                let names = ["Entail", "Neutral", "Contra"];
                for (label, score) in &scores {
                    named.insert(names[*label as usize].to_string(), -score);
                }
                named
            })
            .collect()
    }

    pub fn accuracy(&self, x: &[Array1<f64>], y: &[i32]) -> f64 {
        let preds = self.predict(x);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        correct as f64 / y.len() as f64 * 100.0
    }
}

impl Default for SharpAttractorField {
    fn default() -> Self {
        Self::new(4)
    }
}
