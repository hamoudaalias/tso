/// Multi-module grid cells — code de position injectif multi-échelle.
///
/// Remplace le code 1D scalaire par M modules à périodes incommensurables.
/// Chaque module encode la position (x, y) avec une paire sin/cos par axe,
/// imitant les cellules de grille de l'entorhinal (Hafting et al. 2005).
///
/// Avec des périodes [p₁, p₂, ..., pₘ] premières entre elles, le code produit
/// est injectif sur une grille de taille w×h dès que Π p_i > w×h.
///
/// Pour 5×5, les périodes [3, 5] donnent 3×5=15 > 25? Non, 15 < 25.
/// [3, 5, 7] donne 105 > 25 → injectif garanti.
/// En pratique [2, 3, 5] suffit (produit = 30 > 25).
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct MultiGridCell {
    /// Périodes spatiales des modules (ex: [2, 3, 5])
    pub periods: Vec<usize>,
    /// Nombre total de dimensions = periods.len() × 4
    /// (cos_x, sin_x, cos_y, sin_y par module)
    pub total_dim: usize,
    /// Cache des codes pour chaque position (x, y) — évite de re-calculer
    code_cache: Vec<Vec<Vec<f64>>>,
}

impl MultiGridCell {
    /// Crée un encodeur multi-module avec les périodes données.
    /// Les périodes doivent être > 1 et de préférence premières entre elles.
    pub fn new(width: usize, height: usize, periods: &[usize]) -> Self {
        assert!(!periods.is_empty(), "MultiGridCell: need at least one period");
        let total_dim = periods.len() * 4;

        let mut cells = vec![vec![vec![0.0; total_dim]; height]; width];
        let periods_vec = periods.to_vec();

        for x in 0..width {
            for y in 0..height {
                let mut code = Vec::with_capacity(total_dim);
                for &p in &periods_vec {
                    let px = 2.0 * std::f64::consts::PI * x as f64 / p as f64;
                    let py = 2.0 * std::f64::consts::PI * y as f64 / p as f64;
                    code.push(px.cos());
                    code.push(px.sin());
                    code.push(py.cos());
                    code.push(py.sin());
                }
                cells[x][y] = code;
            }
        }

        MultiGridCell {
            periods: periods_vec,
            total_dim,
            code_cache: cells,
        }
    }

    /// Retourne le code de grille pour une position donnée.
    pub fn encode(&self, x: usize, y: usize) -> &[f64] {
        if x < self.code_cache.len() && y < self.code_cache[0].len() {
            &self.code_cache[x][y]
        } else {
            // Fallback pour les positions hors-limite (ne devrait pas arriver)
            &self.code_cache[0][0]
        }
    }

    /// Étend un vecteur perception avec le code de grille.
    pub fn augment(&self, perception: &[f64], x: usize, y: usize) -> Vec<f64> {
        let code = self.encode(x, y);
        let mut augmented = Vec::with_capacity(perception.len() + self.total_dim);
        augmented.extend_from_slice(perception);
        augmented.extend_from_slice(code);
        augmented
    }

    /// Dimension ajoutée à la perception.
    pub fn extra_dim(&self) -> usize {
        self.total_dim
    }

    /// Teste l'injectivité du code sur une grille w×h.
    /// Retourne `true` si toutes les positions ont des codes uniques
    /// (distance cosinus > 0.01 entre chaque paire).
    /// Affiche aussi les paires les plus proches pour diagnostic.
    pub fn test_injectivity(&self, width: usize, height: usize) -> bool {
        let mut codes: Vec<(usize, usize, Vec<f64>)> = Vec::with_capacity(width * height);
        for x in 0..width {
            for y in 0..height {
                codes.push((x, y, self.encode(x, y).to_vec()));
            }
        }

        let mut min_dist = f64::MAX;
        let mut collisions: Vec<((usize, usize), (usize, usize), f64)> = Vec::new();
        let threshold = 0.01;

        for i in 0..codes.len() {
            for j in i+1..codes.len() {
                let (x1, y1, ref c1) = codes[i];
                let (x2, y2, ref c2) = codes[j];
                let dot: f64 = c1.iter().zip(c2.iter()).map(|(a, b)| a * b).sum();
                let n1: f64 = c1.iter().map(|a| a * a).sum::<f64>().sqrt();
                let n2: f64 = c2.iter().map(|a| a * a).sum::<f64>().sqrt();
                let cos_sim = if n1 > 0.0 && n2 > 0.0 { dot / (n1 * n2) } else { 0.0 };
                let dist = 1.0 - cos_sim;

                if dist < min_dist {
                    min_dist = dist;
                }
                if dist < threshold {
                    collisions.push(((x1, y1), (x2, y2), dist));
                }
            }
        }

        eprintln!("╔═══ MultiGridCell — Test d'injectivité ═══╗");
        eprintln!("║  Périodes    : {:?}", self.periods);
        eprintln!("║  Dimensions  : {}", self.total_dim);
        eprintln!("║  Grille      : {} × {} = {} positions", width, height, width * height);
        eprintln!("║  Distance min: {:.6}", min_dist);
        eprintln!("║  Collisions  : {} (seuil {})", collisions.len(), threshold);

        if collisions.is_empty() {
            eprintln!("║  ✅ Code INJECTIF — toutes les positions sont uniques");
        } else {
            eprintln!("║  ❌ Code NON injectif — {} collisions:", collisions.len());
            for ((x1, y1), (x2, y2), d) in collisions.iter().take(10) {
                eprintln!("║     ({},{}) ↔ ({},{})  dist={:.6}", x1, y1, x2, y2, d);
            }
        }
        eprintln!("╚═══════════════════════════════════════════╝");

        collisions.is_empty()
    }
}
