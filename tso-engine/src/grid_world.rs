use ndarray::Array1;
use std::collections::VecDeque;

/// GridWorld 2D — environnement à "moustaches" (whiskers).
/// L'agent ne voit jamais sa position absolue : uniquement 4 distances aux murs.
/// Chaque pas donne -0.05, chaque mur donne -1.0, le but donne +10.0.
/// Le reward shaping utilise la distance BFS au but (respecte les murs).
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    walls: Vec<Vec<bool>>,
    goal: (usize, usize),
    start: (usize, usize),
    pub agent: (usize, usize),
    pub done: bool,
    steps: usize,
    max_steps: usize,
    bfs_dist: Vec<Vec<Option<usize>>>,
    max_bfs: f64,
    prev_pos: (usize, usize),
    gamma: f64,
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
        GridWorld { width: 5, height: 5, walls, goal, start: (1, 1), agent: (1, 1), done: false, steps: 0, max_steps: 50, bfs_dist: bfs, max_bfs, prev_pos: (1, 1), gamma: 0.99 }
    }

    /// Labyrinthe 10×10 en zigzag
    pub fn zigzag() -> Self {
        let mut w = vec![vec![false; 10]; 10];
        for i in 0..10 { w[i][0] = true; w[i][9] = true; w[0][i] = true; w[9][i] = true; }
        for x in 1..7 { w[x][2] = true; }
        for x in 2..9 { w[x][4] = true; }
        for x in 1..7 { w[x][6] = true; }
        let goal = (8, 8);
        let bfs = bfs_distance(&w, goal);
        let max_bfs = bfs.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld { width: 10, height: 10, walls: w, goal, start: (1, 1), agent: (1, 1), done: false, steps: 0, max_steps: 200, bfs_dist: bfs, max_bfs, prev_pos: (1, 1), gamma: 0.99 }
    }

    /// L-Maze 7×7 — un couloir en forme de L.
    pub fn l_maze() -> Self {
        let mut w = vec![vec![false; 7]; 7];
        for i in 0..7 { w[i][0] = true; w[i][6] = true; w[0][i] = true; w[6][i] = true; }
        for x in 2..6 { w[x][2] = true; }
        for y in 3..5 { w[5][y] = true; }
        let goal = (5, 5);
        let bfs = bfs_distance(&w, goal);
        let max_bfs = bfs.iter().flat_map(|r| r.iter()).filter_map(|o| *o).max().unwrap_or(1) as f64;
        GridWorld { width: 7, height: 7, walls: w, goal, start: (1, 1), agent: (1, 1), done: false, steps: 0, max_steps: 50, bfs_dist: bfs, max_bfs, prev_pos: (1, 1), gamma: 0.99 }
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

    pub fn step(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;
        self.prev_pos = self.agent;
        let (x, y) = (self.agent.0 as isize, self.agent.1 as isize);
        let (nx, ny) = match action { 0 => (x, y - 1), 1 => (x, y + 1), 2 => (x - 1, y), 3 => (x + 1, y), _ => (x, y) };
        if self.is_wall(nx, ny) {
            if self.steps >= self.max_steps { self.done = true; }
            return -1.0;
        }
        self.agent = (nx as usize, ny as usize);

        let base = if self.agent == self.goal { self.done = true; 20.0 }
                   else if self.steps >= self.max_steps { self.done = true; -1.0 }
                   else { -0.05 };

        // Potential-based shaping with BFS distance (wall-respecting gradient)
        let d_old = self.bfs_dist[self.prev_pos.0][self.prev_pos.1];
        let d_new = self.bfs_dist[self.agent.0][self.agent.1];
        let shaping = match (d_old, d_new) {
            (Some(do_), Some(dn)) => self.gamma * (-0.05 * dn as f64) - (-0.05 * do_ as f64),
            _ => 0.0,
        };
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
}


