use ndarray::Array1;
use serde::{Serialize, Deserialize};

/// Système de cellules de grille — désambiguïse l'aliasing perceptuel
/// en assignant un identifiant unique à chaque position (x, y) dans le monde.
///
/// Problème : dans un labyrinthe >6×6, deux positions différentes peuvent
/// produire des lectures de moustaches identiques (aliasing). L'attracteur
/// crée alors un seul concept pour les deux positions, empêchant
/// l'apprentissage de valeurs différentes.
///
/// Solution : encoder la position absolue comme une dimension
/// supplémentaire dans la perception. Chaque cellule a un ID unique
/// normalisé en [0, 1] qui sert de « colonne de grille » — analogue
/// aux cellules de lieu (place cells) de l'hippocampe.
#[derive(Clone, Serialize, Deserialize)]
pub struct GridCells {
    /// width × height de l'environnement
    pub width: usize,
    pub height: usize,
    /// cell_id[x][y] = identifiant unique normalisé
    cells: Vec<Vec<f64>>,
    /// Combien de dimensions sont réservées aux cellules de grille
    pub n_dims: usize,
}

impl GridCells {
    pub fn new(width: usize, height: usize) -> Self {
        let n_dims = if width * height > 36 { 1 } else { 0 };
        let mut cells = vec![vec![0.0; height]; width];
        let total = (width * height) as f64;
        for x in 0..width {
            for y in 0..height {
                cells[x][y] = (x * height + y) as f64 / total;
            }
        }
        GridCells { width, height, cells, n_dims }
    }

    /// Active ou désactive les cellules selon la taille de la grille
    pub fn auto_configure(&mut self, w: usize, h: usize) {
        self.width = w;
        self.height = h;
        self.n_dims = if w * h > 36 { 1 } else { 0 };
        let total = (w * h) as f64;
        self.cells = vec![vec![0.0; h]; w];
        for x in 0..w {
            for y in 0..h {
                self.cells[x][y] = (x * h + y) as f64 / total;
            }
        }
    }

    /// Force l'activation des cellules de grille (pour tests comparatifs)
    pub fn force_on(&mut self, w: usize, h: usize) {
        self.width = w;
        self.height = h;
        self.n_dims = 1;
        self.cells = vec![vec![0.0; h]; w];
        let total = (w * h) as f64;
        for x in 0..w {
            for y in 0..h {
                self.cells[x][y] = (x * h + y) as f64 / total;
            }
        }
    }

    /// Force la désactivation des cellules
    pub fn force_off(&mut self) {
        self.n_dims = 0;
        self.cells.clear();
    }

    /// Retourne le cell_id normalisé pour une position donnée
    pub fn cell_id(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.cells[x][y]
        } else {
            0.0
        }
    }

    /// Nombre de dimensions supplémentaires à ajouter à la perception
    pub fn extra_dim(&self) -> usize {
        self.n_dims
    }

    /// Étend un vecteur de perception avec les cellules de grille
    pub fn augment(&self, perception: &Array1<f64>, x: usize, y: usize) -> Array1<f64> {
        if self.n_dims == 0 {
            return perception.clone();
        }
        let cell = self.cell_id(x, y);
        let mut augmented = Vec::with_capacity(perception.len() + 1);
        for &v in perception.iter() {
            augmented.push(v);
        }
        augmented.push(cell);
        Array1::from_vec(augmented)
    }

    /// Dimension totale de la perception augmentée
    pub fn dim_with_cells(&self, base_dim: usize) -> usize {
        base_dim + self.extra_dim()
    }
}
