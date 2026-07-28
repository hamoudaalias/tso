use ndarray::Array1;
use std::collections::VecDeque;
use rand::Rng;

/// GridWorld 2D — environnement à "moustaches" (whiskers).
/// L'agent ne voit jamais sa position absolue : uniquement 4 distances aux murs.
/// Chaque pas donne -0.05, chaque mur donne -1.0, le but donne +10.0.
/// Le reward shaping utilise la distance BFS au but (respecte les murs).
///
/// ## Renderer
/// `render_ascii()` produit une chaîne console. Avec la feature `image`,
/// `render_png()` écrit un fichier PNG de la grille.
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub walls: Vec<Vec<bool>>,
    pub goal: (usize, usize),
    start: (usize, usize),
    pub agent: (usize, usize),
    pub done: bool,
    pub steps: usize,
    pub max_steps: usize,
    bfs_dist: Vec<Vec<Option<usize>>>,
    max_bfs: f64,
    prev_pos: (usize, usize),
    gamma: f64,
    pub visit_count: Vec<Vec<usize>>,
}

fn flood_fill(walls: &[Vec<bool>], start: (usize, usize)) -> Vec<Vec<bool>> {
    let w = walls.len();
    let h = walls[0].len();
    let mut visited = vec![vec![false; h]; w];
    let mut q = VecDeque::new();
    if !walls[start.0][start.1] {
        visited[start.0][start.1] = true;
        q.push_back(start);
    }
    while let Some((cx, cy)) = q.pop_front() {
        for &(dx, dy) in &[(0isize, -1isize), (0, 1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                let (nx, ny) = (nx as usize, ny as usize);
                if !walls[nx][ny] && !visited[nx][ny] {
                    visited[nx][ny] = true;
                    q.push_back((nx, ny));
                }
            }
        }
    }
    visited
}

fn bfs_distance(walls: &[Vec<bool>], goal: (usize, usize)) -> Vec<Vec<Option<usize>>> {
    let w = walls.len();
    let h = walls[0].len();
    let mut dist = vec![vec![None; h]; w];
    let mut q = VecDeque::new();
    dist[goal.0][goal.1] = Some(0);
    q.push_back(goal);
    while let Some((cx, cy)) = q.pop_front() {
        for &(dx, dy) in &[(0isize, -1isize), (0, 1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                let (nx, ny) = (nx as usize, ny as usize);
                if !walls[nx][ny] && dist[nx][ny].is_none() {
                    dist[nx][ny] = Some(dist[cx][cy].unwrap() + 1);
                    q.push_back((nx, ny));
                }
            }
        }
    }
    dist
}

impl GridWorld {
    /// Pièce vide 5×5. Start=(1,1), Goal=(3,3).
    pub fn empty_room() -> Self {
        let mut walls = vec![vec![false; 5]; 5];
        for i in 0..5 { walls[i][0] = true; walls[0][i] = true; walls[i][4] = true; walls[4][i] = true; }
        let goal = (3, 3);
        let bfs = bfs_distance(&walls, goal);
        let max_bfs = bfs.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld { width: 5, height: 5, walls, goal, start: (1, 1), agent: (1, 1), done: false, steps: 0, max_steps: 50, bfs_dist: bfs, max_bfs, prev_pos: (1, 1), gamma: 0.99, visit_count: vec![vec![0usize; 5]; 5] }
    }

    /// Corridor horizontal 10×1 : start à gauche, goal à droite, aucun mur interne.
    /// Test minimal pour vérifier que l'apprentissage fonctionne.
    pub fn straight() -> Self {
        let w = 10;
        let h = 1;
        let mut walls = vec![vec![false; h]; w];
        for i in 0..w { walls[i][0] = false; }
        walls[0][0] = true; walls[w-1][0] = true;
        let goal = (8, 0);
        let bfs = bfs_distance(&walls, goal);
        let max_bfs = bfs.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld { width: w, height: h, walls, goal, start: (1, 0), agent: (1, 0), done: false, steps: 0, max_steps: 50, bfs_dist: bfs, max_bfs, prev_pos: (1, 0), gamma: 0.99, visit_count: vec![vec![0usize; h]; w] }
    }

    /// L-shaped corridor 10×10 : start (1,1) → right to col 8 → down to (8,8).
    /// A single decision point at the corner. Random walks will reliably discover the goal,
    /// giving the cerebellum the positive reward signal it needs to learn value.
    pub fn corridor() -> Self {
        let mut w = vec![vec![true; 10]; 10];
        for x in 1..9 { w[x][1] = false; }
        for y in 1..9 { w[8][y] = false; }
        let goal = (8, 8);
        let bfs = bfs_distance(&w, goal);
        let max_bfs = bfs.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld { width: 10, height: 10, walls: w, goal, start: (1, 1), agent: (1, 1), done: false, steps: 0, max_steps: 200, bfs_dist: bfs, max_bfs, prev_pos: (1, 1), gamma: 0.99, visit_count: vec![vec![0usize; 10]; 10] }
    }

    /// Labyrinthe aléatoire de taille donnée.
    /// Génère un intérieur avec ~35% de murs, garantit la connexité,
    /// place le but à la plus grande distance BFS du départ.
    pub fn random(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut walls = vec![vec![false; height]; width];
        for i in 0..width { walls[i][0] = true; walls[i][height-1] = true; }
        for j in 0..height { walls[0][j] = true; walls[width-1][j] = true; }
        for x in 1..width-1 {
            for y in 1..height-1 {
                if rng.r#gen::<f64>() < 0.35 { walls[x][y] = true; }
            }
        }
        walls[1][1] = false;
        loop {
            let reachable = flood_fill(&walls, (1, 1));
            let mut changed = false;
            for x in 1..width-1 {
                for y in 1..height-1 {
                    if !walls[x][y] { continue; }
                    let mut adj_reach = false;
                    let mut adj_passage = false;
                    for &(dx, dy) in &[(0, -1), (0, 1), (-1, 0), (1, 0)] {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx > 0 && ny > 0 && nx < width as isize-1 && ny < height as isize-1 {
                            let (nx, ny) = (nx as usize, ny as usize);
                            if reachable[nx][ny] { adj_reach = true; }
                            else if !walls[nx][ny] { adj_passage = true; }
                        }
                    }
                    if adj_reach && adj_passage { walls[x][y] = false; changed = true; }
                }
            }
            if !changed { break; }
        }
        let start = (1, 1);
        let bfs_from_start = bfs_distance(&walls, start);
        let max_dist = bfs_from_start.iter().flat_map(|r| r.iter()).filter_map(|&d| d).max().unwrap_or(0);
        let candidates: Vec<_> = bfs_from_start.iter().enumerate()
            .flat_map(|(x, row)| row.iter().enumerate()
                .filter_map(move |(y, &d)| if d == Some(max_dist) { Some((x, y)) } else { None }))
            .collect();
        let goal = if candidates.is_empty() { (width-2, height-2) } else { candidates[rng.gen_range(0..candidates.len())] };
        let bfs_from_goal = bfs_distance(&walls, goal);
        let max_bfs = bfs_from_goal.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld {
            width, height, walls, goal, start,
            agent: start, done: false, steps: 0,
            max_steps: (width * height * 2).max(50),
            bfs_dist: bfs_from_goal, max_bfs, prev_pos: start, gamma: 0.99,
            visit_count: vec![vec![0usize; height]; width],
        }
    }


    pub fn reset(&mut self) {
        self.agent = self.start;
        self.prev_pos = self.start;
        self.done = false;
        self.steps = 0;
    }

    pub fn is_wall(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize { return true; }
        self.walls[x as usize][y as usize]
    }

    /// Perception : 4 whiskers N,S,E,O + odométrie locale.
    /// Le TsoEngine utilise les 4 premières dims pour les concepts (espace pur)
    /// et la 5ème pour le decision_state (contexte temporel).
    pub fn bfs_at_current_pos(&self) -> Option<usize> {
        let (x, y) = self.agent;
        if x < self.bfs_dist.len() && y < self.bfs_dist[0].len() {
            self.bfs_dist[x][y]
        } else {
            None
        }
    }

    /// Perception 4D pure : whiskers uniquement (pas de bfs_frac)
    pub fn perception_4d(&self) -> Array1<f64> {
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let md = self.width.max(self.height) as f64;
        Array1::from_vec(vec![
            Self::ray(x, y, 0, -1, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, 0, 1, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, 1, 0, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, -1, 0, &self.walls, self.width, self.height) as f64 / md,
        ])
    }

    pub fn perception(&self) -> Array1<f64> {
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let md = self.width.max(self.height) as f64;
        let bfs_frac = self.bfs_dist[x as usize][y as usize]
            .map(|d| d as f64 / self.max_bfs)
            .unwrap_or(1.0);
        Array1::from_vec(vec![
            Self::ray(x, y, 0, -1, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, 0, 1, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, 1, 0, &self.walls, self.width, self.height) as f64 / md,
            Self::ray(x, y, -1, 0, &self.walls, self.width, self.height) as f64 / md,
            bfs_frac,
        ])
    }

    pub fn step_count_norm(&self) -> f64 {
        self.steps as f64 / self.max_steps.max(1) as f64
    }

    fn ray(mut x: isize, mut y: isize, dx: isize, dy: isize, walls: &[Vec<bool>], w: usize, h: usize) -> usize {
        let mut d = 0;
        loop { x += dx; y += dy;
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize { return d; }
            if walls[x as usize][y as usize] { return d; } d += 1; }
    }

    /// Step sans potential-based shaping (récompense plate)
    pub fn step_flat(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let (nx, ny) = match action { 0 => (x, y - 1), 1 => (x, y + 1), 2 => (x - 1, y), 3 => (x + 1, y), _ => (x, y) };
        if self.is_wall(nx, ny) {
            if self.steps >= self.max_steps { self.done = true; }
            self.visit_count[self.agent.0][self.agent.1] += 1;
            return -0.5;
        }
        self.agent = (nx as usize, ny as usize);
        self.visit_count[self.agent.0][self.agent.1] += 1;
        if self.agent == self.goal { self.done = true; 20.0 }
        else if self.steps >= self.max_steps { self.done = true; -1.0 }
        else { 0.0 }
    }

    pub fn step(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;
        self.prev_pos = self.agent;
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let (nx, ny) = match action { 0 => (x, y - 1), 1 => (x, y + 1), 2 => (x - 1, y), 3 => (x + 1, y), _ => (x, y) };
        if self.is_wall(nx, ny) {
            if self.steps >= self.max_steps { self.done = true; }
            return -0.5;
        }
        self.agent = (nx as usize, ny as usize);

        let base = if self.agent == self.goal { self.done = true; 20.0 }
                   else if self.steps >= self.max_steps { self.done = true; -1.0 }
                   else { -0.01 };

        // Potential-based shaping with BFS frac (normalised, cœur du signal d'apprentissage)
        let d_old = self.bfs_dist[self.prev_pos.0][self.prev_pos.1]
            .map(|d| d as f64 / self.max_bfs).unwrap_or(1.0);
        let d_new = self.bfs_dist[self.agent.0][self.agent.1]
            .map(|d| d as f64 / self.max_bfs).unwrap_or(1.0);
        let shaping = self.gamma * (-2.5 * d_new) - (-2.5 * d_old);
        base + shaping
    }

    /// BFS advantage for each action: positive = toward goal, negative = away, -999 = wall
    pub fn bfs_gradient(&self) -> Vec<f64> {
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let cur = self.bfs_dist[self.agent.0][self.agent.1].unwrap_or(self.max_bfs as usize) as f64;
        let mut out = Vec::with_capacity(4);
        for &(dx, dy) in &[(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let nx = x + dx; let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= self.width as isize || ny >= self.height as isize
                || self.walls[nx as usize][ny as usize]
            {
                out.push(-999.0);
            } else {
                let nd = self.bfs_dist[nx as usize][ny as usize].unwrap_or(self.max_bfs as usize) as f64;
                out.push((cur - nd) / self.max_bfs);
            }
        }
        out
    }

    pub fn open_cells(&self) -> Vec<(usize, usize)> {
        let mut c = Vec::new();
        for x in 0..self.width { for y in 0..self.height { if !self.walls[x][y] { c.push((x, y)); } } }
        c
    }

    /// Exploration bonus inversely proportional to visit count.
    /// Visits to infrequently-visited cells get a positive bonus that decays
    /// as the cell becomes familiar. Creates an intrinsic gradient toward
    /// unexplored regions of the corridor.
    pub fn exploration_bonus(&self) -> f64 {
        let count = self.visit_count[self.agent.0][self.agent.1].max(1);
        0.8 / (count as f64).sqrt()
    }

    // ── Renderer ───────────────────────────────────────────────────────

    /// Rend la grille en ASCII.
    /// @=agent  G=goal  #=wall  ~=water  .=empty
    pub fn render_ascii(&self) -> String {
        let water = [(1,1), (3,3), (1,4)];
        let mut out = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if (x, y) == self.agent {
                    out.push('@');
                } else if (x, y) == self.goal {
                    out.push('G');
                } else if self.walls[x][y] {
                    out.push('#');
                } else if water.contains(&(x, y)) {
                    out.push('~');
                } else {
                    out.push('.');
                }
            }
            out.push('\n');
        }
        out
    }

    /// Écrit un PNG 200×200 de la grille si la feature `image` est activée.
    #[cfg(feature = "image")]
    pub fn render_png(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use image::{Rgb, RgbImage};
        let cell = 40u32;
        let mut img = RgbImage::new(self.width as u32 * cell, self.height as u32 * cell);
        let water = [(1usize,1usize), (3,3), (1,4)];
        for y in 0..self.height {
            for x in 0..self.width {
                let c = if (x, y) == self.agent { Rgb([0,255,0]) }
                    else if (x, y) == self.goal { Rgb([255,215,0]) }
                    else if self.walls[x][y] { Rgb([80,80,80]) }
                    else if water.contains(&(x, y)) { Rgb([0,100,255]) }
                    else { Rgb([240,240,240]) };
                for py in 0..cell {
                    for px in 0..cell {
                        img.put_pixel(x as u32 * cell + px, y as u32 * cell + py, c);
                    }
                }
            }
        }
        img.save(path)?;
        Ok(())
    }
}


