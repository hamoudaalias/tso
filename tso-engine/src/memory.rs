use ndarray::Array1;

#[derive(Clone)]
pub struct Entry {
    pub vector: Array1<f64>,
    pub data: usize,
}

#[derive(Clone)]
pub struct AssociativeMemory {
    pub entries: Vec<Entry>,
}

impl AssociativeMemory {
    pub fn new() -> Self {
        AssociativeMemory { entries: Vec::new() }
    }

    pub fn store(&mut self, vector: &Array1<f64>, data: usize) {
        self.entries.push(Entry {
            vector: vector.clone(),
            data,
        });
    }

    pub fn recall(&self, query: &Array1<f64>) -> Option<usize> {
        let mut best_sim = -1.0;
        let mut best_data = None;
        for e in &self.entries {
            let sim = cosine_sim(query, &e.vector);
            if sim > best_sim {
                best_sim = sim;
                best_data = Some(e.data);
            }
        }
        best_data
    }

    pub fn recall_with_sim(&self, query: &Array1<f64>) -> Option<(usize, f64)> {
        let mut best_sim = -1.0;
        let mut best_data = None;
        for e in &self.entries {
            let sim = cosine_sim(query, &e.vector);
            if sim > best_sim {
                best_sim = sim;
                best_data = Some(e.data);
            }
        }
        best_data.map(|d| (d, best_sim))
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

fn cosine_sim(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let na = a.dot(a).sqrt().max(1e-12);
    let nb = b.dot(b).sqrt().max(1e-12);
    dot / (na * nb)
}
