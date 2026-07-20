#[derive(Debug, Clone)]
pub struct Metrics {
    pub accuracy: f64,
    pub confusion: [[usize; 3]; 3],
    pub precision: [f64; 3],
    pub recall: [f64; 3],
    pub f1: [f64; 3],
    pub total: usize,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            confusion: [[0; 3]; 3],
            precision: [0.0; 3],
            recall: [0.0; 3],
            f1: [0.0; 3],
            total: 0,
        }
    }

    pub fn add(&mut self, predicted: usize, actual: usize) {
        if predicted < 3 && actual < 3 {
            self.confusion[actual][predicted] += 1;
            self.total += 1;
        }
    }

    pub fn compute(&mut self) {
        if self.total == 0 {
            return;
        }
        let mut correct = 0;
        for i in 0..3 {
            correct += self.confusion[i][i];
        }
        self.accuracy = correct as f64 / self.total as f64;

        for c in 0..3 {
            let tp = self.confusion[c][c] as f64;
            let fp: usize = (0..3).map(|r| self.confusion[r][c]).sum();
            let fn_: usize = (0..3).map(|c2| self.confusion[c][c2]).sum();
            self.precision[c] = if tp + fp as f64 > 0.0 { tp / (tp + fp as f64) } else { 0.0 };
            self.recall[c] = if tp + fn_ as f64 > 0.0 { tp / (tp + fn_ as f64) } else { 0.0 };
            self.f1[c] = if self.precision[c] + self.recall[c] > 0.0 {
                2.0 * self.precision[c] * self.recall[c] / (self.precision[c] + self.recall[c])
            } else {
                0.0
            };
        }
    }

    pub fn report(&self) -> String {
        let labels = ["entailment", "neutral", "contradiction"];
        let mut s = String::new();
        s.push_str(&format!("Accuracy: {:.2}%\n", self.accuracy * 100.0));
        s.push_str(&format!("Total: {}\n\n", self.total));
        s.push_str("Confusion matrix (actual ↓ predicted →):\n");
        s.push_str(&format!("{:>13} {:>6} {:>6} {:>6}\n", "", "ENT", "NEU", "CON"));
        for (i, label) in labels.iter().enumerate() {
            s.push_str(&format!(
                "{:>12} {:>6} {:>6} {:>6}\n",
                label,
                self.confusion[i][0],
                self.confusion[i][1],
                self.confusion[i][2],
            ));
        }
        s.push_str("\nPer class:\n");
        s.push_str(&format!("{:>13} {:>8} {:>8} {:>8}\n", "", "Prec", "Rec", "F1"));
        for (i, label) in labels.iter().enumerate() {
            s.push_str(&format!(
                "{:>12} {:>7.1}% {:>7.1}% {:>7.1}%\n",
                label,
                self.precision[i] * 100.0,
                self.recall[i] * 100.0,
                self.f1[i] * 100.0,
            ));
        }
        s
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_accuracy() {
        let mut m = Metrics::new();
        for _ in 0..10 {
            m.add(0, 0);
            m.add(1, 1);
            m.add(2, 2);
        }
        m.compute();
        assert!((m.accuracy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix() {
        let mut m = Metrics::new();
        m.add(0, 0);
        m.add(1, 1);
        m.add(0, 2); // actual contradiction, predicted entailment
        m.compute();
        assert_eq!(m.confusion[0][0], 1);
        assert_eq!(m.confusion[1][1], 1);
        assert_eq!(m.confusion[2][0], 1);
    }
}
