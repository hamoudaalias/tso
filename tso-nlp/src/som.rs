use std::collections::HashMap;

/// Self-Organizing Map for semantic clustering of words.
/// Maps high-dimensional distributional vectors to a 2D grid of clusters.
pub struct SOM {
    grid_rows: usize,
    grid_cols: usize,
    dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl SOM {
    pub fn new(grid_rows: usize, grid_cols: usize, dim: usize) -> Self {
        let n_neurons = grid_rows * grid_cols;
        let mut weights = Vec::with_capacity(n_neurons);
        for i in 0..n_neurons {
            let mut w = Vec::with_capacity(dim);
            let seed = (i * 2654435761) as f64 * 0.0001;
            for j in 0..dim {
                w.push(((seed + j as f64 * 0.618).fract() - 0.5) * 0.1);
            }
            weights.push(w);
        }
        Self { grid_rows, grid_cols, dim, weights }
    }

    pub fn n_clusters(&self) -> usize {
        self.grid_rows * self.grid_cols
    }

    /// Train the SOM on the given data for one epoch (Kohonen rule).
    /// Learning rate adapts linearly from `initial_lr` to 0.
    /// Neighborhood radius adapts linearly from `initial_radius` to 1.
    pub fn train(&mut self, data: &[Vec<f64>], initial_lr: f64, initial_radius: f64) {
        if data.is_empty() || self.dim == 0 { return; }

        let lr_start = initial_lr;
        let lr_end = 0.01;
        let r_start = initial_radius;
        let r_end = 1.0;

        for (step, input) in data.iter().enumerate() {
            let progress = step as f64 / data.len() as f64;
            let lr = lr_start + (lr_end - lr_start) * progress;
            let radius = r_start + (r_end - r_start) * progress;
            let radius_sq = radius * radius;

            let bmu_idx = self.bmu(input);
            let bmu_row = bmu_idx / self.grid_cols;
            let bmu_col = bmu_idx % self.grid_cols;

            for i in 0..self.weights.len() {
                let row = i / self.grid_cols;
                let col = i % self.grid_cols;
                let dr = (row as f64 - bmu_row as f64).abs();
                let dc = (col as f64 - bmu_col as f64).abs();
                let dist_sq = dr * dr + dc * dc;

                if dist_sq <= radius_sq + 0.5 {
                    let influence = (-dist_sq / (2.0 * radius_sq.max(0.01))).exp();
                    let rate = lr * influence;
                    for j in 0..self.dim {
                        self.weights[i][j] += rate * (input[j] - self.weights[i][j]);
                    }
                }
            }
        }
    }

    /// Find the Best Matching Unit for the given input vector.
    pub fn bmu(&self, input: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, w) in self.weights.iter().enumerate() {
            let dist = input.iter().zip(w.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Build distributional vectors for all words in the graph.
    /// Each word's vector = conditional probabilities to the top-D most connected words.
    /// Vectors are L2-normalized.
    pub fn build_word_vectors(
        graph: &HashMap<String, HashMap<String, f64>>,
        dim: usize,
    ) -> (Vec<String>, Vec<Vec<f64>>, Vec<String>) {
        // Find the top-D words by degree (most connected = most frequent)
        let mut degrees: Vec<(&String, usize)> = graph.iter()
            .map(|(w, neighbors)| (w, neighbors.len()))
            .collect();
        degrees.sort_by(|a, b| b.1.cmp(&a.1));
        let top_words: Vec<String> = degrees.iter()
            .take(dim)
            .map(|(w, _)| (*w).clone())
            .collect();

        let mut words = Vec::with_capacity(graph.len());
        let mut vectors = Vec::with_capacity(graph.len());

        for (word, neighbors) in graph.iter() {
            let mut vec = Vec::with_capacity(dim);
            for top_word in &top_words {
                let val = neighbors.get(top_word).copied().unwrap_or(0.0);
                vec.push(val);
            }
            // L2 normalize
            let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in &mut vec { *v /= norm; }
            }
            words.push(word.clone());
            vectors.push(vec);
        }

        (words, vectors, top_words)
    }

    /// Assign cluster IDs to all words based on their BMU.
    /// Returns a HashMap<String, usize> mapping word → cluster_id (0..n_clusters).
    pub fn assign_clusters(&self, words: &[String], vectors: &[Vec<f64>]) -> HashMap<String, usize> {
        words.iter().zip(vectors.iter())
            .map(|(word, vec)| (word.clone(), self.bmu(vec)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_som_creation() {
        let som = SOM::new(5, 5, 10);
        assert_eq!(som.n_clusters(), 25);
        assert_eq!(som.weights.len(), 25);
        assert_eq!(som.weights[0].len(), 10);
    }

    #[test]
    fn test_bmu_identical() {
        let mut som = SOM::new(3, 3, 4);
        let input = vec![0.5, 0.3, 0.1, 0.9];
        // Set one neuron to exactly the input
        som.weights[4] = input.clone();
        let idx = som.bmu(&input);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_train_does_not_crash() {
        let mut som = SOM::new(4, 4, 5);
        let data = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0.5, 0.4, 0.3, 0.2, 0.1],
            vec![0.9, 0.8, 0.7, 0.6, 0.5],
        ];
        som.train(&data, 0.5, 3.0);
        // After training, BMU for each input should be consistent
        for v in &data {
            let idx = som.bmu(v);
            assert!(idx < 16);
        }
    }

    #[test]
    fn test_build_word_vectors() {
        let mut graph: HashMap<String, HashMap<String, f64>> = HashMap::new();
        let mut neighbors = HashMap::new();
        neighbors.insert("animal".to_string(), 0.8);
        graph.insert("dog".to_string(), neighbors);
        graph.insert("cat".to_string(), HashMap::new());

        let (words, vectors, top_words) = SOM::build_word_vectors(&graph, 2);
        assert_eq!(words.len(), 2);
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), 2);
        assert!(!top_words.is_empty());
    }

    #[test]
    fn test_assign_clusters() {
        let mut som = SOM::new(2, 2, 3);
        som.weights[0] = vec![1.0, 0.0, 0.0];
        som.weights[1] = vec![0.0, 1.0, 0.0];
        som.weights[2] = vec![0.0, 0.0, 1.0];
        som.weights[3] = vec![0.5, 0.5, 0.5];

        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let vectors = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.0, 0.0, 0.9],
        ];

        let clusters = som.assign_clusters(&words, &vectors);
        assert_eq!(clusters.get("a").copied(), Some(0));
        assert_eq!(clusters.get("b").copied(), Some(1));
        assert_eq!(clusters.get("c").copied(), Some(2));
    }
}
