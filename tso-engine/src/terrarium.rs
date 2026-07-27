use ndarray::Array1;
use serde::{Serialize, Deserialize};

/// Terrarium — environnement de survie avec récompenses rares.
/// L'agent doit trouver de la nourriture et de l'eau pour maintenir
/// son énergie. Les récompenses sont rares (uniquement sur les ressources).
///
/// Perception : [N, S, O, E, food_sensed, water_sensed, cell_id?]
/// Actions : 0=N, 1=S, 2=O, 3=E.
#[derive(Clone, Serialize, Deserialize)]
pub struct Terrarium {
    pub width: usize,
    pub height: usize,
    walls: Vec<Vec<bool>>,
    pub agent: (usize, usize),
    food: Vec<(usize, usize)>,
    water: Vec<(usize, usize)>,
    pub energy: f64,
    pub done: bool,
    pub steps: usize,
    pub max_steps: usize,
    /// Récompense accumulée (pour monitoring)
    pub total_reward: f64,
}

impl Terrarium {
    pub fn new(_seed: u64) -> Self {
        let w = 7;
        let h = 7;

        let mut walls = vec![vec![false; h]; w];
        for i in 0..w { walls[i][0] = true; walls[i][h-1] = true; }
        for j in 0..h { walls[0][j] = true; walls[w-1][j] = true; }

        // Passages internes pour former un labyrinthe
        for x in 2..w-1 { walls[x][3] = true; }
        for y in 1..h-1 { walls[3][y] = true; }
        walls[2][3] = false;
        walls[3][2] = false;
        walls[4][3] = false;

        // Placer nourriture (3 sources)
        let food = vec![(2, 1), (5, 4), (1, 5)];
        // Placer eau (3 sources)
        let water = vec![(5, 1), (2, 5), (4, 2)];

        let agent = (1, 1);

        Terrarium {
            width: w, height: h, walls, agent,
            food, water,
            energy: 1.0, done: false, steps: 0,
            max_steps: 200, total_reward: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.agent = (1, 1);
        self.energy = 1.0;
        self.done = false;
        self.steps = 0;
        self.total_reward = 0.0;
    }

    pub fn is_walkable(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return false;
        }
        !self.walls[x as usize][y as usize]
    }

    fn has_food(&self, x: usize, y: usize) -> bool {
        self.food.contains(&(x, y))
    }

    fn has_water(&self, x: usize, y: usize) -> bool {
        self.water.contains(&(x, y))
    }

    /// Perception : [N, S, O, E, food_sensed, water_sensed, cell_id?]
    pub fn perception(&self, cell_id: Option<f64>) -> Array1<f64> {
        let md = self.width.max(self.height) as f64;
        let x = self.agent.0 as isize;
        let y = self.agent.1 as isize;

        let mut p = vec![
            self.ray(x, y, 0, -1) as f64 / md,
            self.ray(x, y, 0, 1) as f64 / md,
            self.ray(x, y, -1, 0) as f64 / md,
            self.ray(x, y, 1, 0) as f64 / md,
        ];

        // Détection de nourriture à proximité (dans les 2 cases)
        let mut food_near = 0.0;
        for &(fx, fy) in &self.food {
            let dx = (x - fx as isize).abs() as f64;
            let dy = (y - fy as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 2.0 { food_near = (1.0 - d / 3.0).max(0.0); break; }
        }
        p.push(food_near);

        // Détection d'eau à proximité
        let mut water_near = 0.0;
        for &(wx, wy) in &self.water {
            let dx = (x - wx as isize).abs() as f64;
            let dy = (y - wy as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 2.0 { water_near = (1.0 - d / 3.0).max(0.0); break; }
        }
        p.push(water_near);

        if let Some(cid) = cell_id {
            p.push(cid);
        }

        Array1::from_vec(p)
    }

    fn ray(&self, x: isize, y: isize, dx: isize, dy: isize) -> usize {
        let mut d = 0;
        let mut cx = x + dx;
        let mut cy = y + dy;
        loop {
            if cx < 0 || cy < 0 || cx >= self.width as isize || cy >= self.height as isize { return d; }
            if self.walls[cx as usize][cy as usize] { return d; }
            d += 1;
            cx += dx;
            cy += dy;
        }
    }

    pub fn step(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;

        // Énergie diminue chaque pas
        self.energy -= 0.01;
        if self.energy <= 0.0 {
            self.done = true;
            self.total_reward -= 10.0;
            return -10.0;
        }

        let (dx, dy) = match action {
            0 => (0, -1), 1 => (0, 1), 2 => (-1, 0), 3 => (1, 0),
            _ => (0, 0),
        };
        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;

        if !self.is_walkable(nx, ny) {
            if self.steps >= self.max_steps { self.done = true; }
            return -0.5;
        }

        self.agent = (nx as usize, ny as usize);

        // Récompense si nourriture
        if self.has_food(self.agent.0, self.agent.1) {
            self.energy = (self.energy + 0.3).min(1.0);
            self.total_reward += 10.0;
            if self.steps >= self.max_steps { self.done = true; }
            return 10.0;
        }

        // Récompense si eau
        if self.has_water(self.agent.0, self.agent.1) {
            self.energy = (self.energy + 0.2).min(1.0);
            self.total_reward += 8.0;
            if self.steps >= self.max_steps { self.done = true; }
            return 8.0;
        }

        if self.steps >= self.max_steps {
            self.done = true;
            return -1.0;
        }

        -0.02
    }
}
