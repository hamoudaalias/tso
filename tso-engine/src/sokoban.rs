use ndarray::Array1;
use serde::{Serialize, Deserialize};

/// Sokoban — jeu de poussée de caisses à niveaux croissants.
/// Niveaux prédéfinis, garantis solvables, de 5×5 (1 box) à 8×8 (4 boxes).
///
/// Perception : [N, S, O, E, box_adjacent, box_dir, target_sensed, cell_id?]
/// Actions : 0=N, 1=S, 2=O, 3=E.
#[derive(Clone, Serialize, Deserialize)]
pub struct Sokoban {
    pub width: usize,
    pub height: usize,
    walls: Vec<Vec<bool>>,
    pub agent: (usize, usize),
    boxes: Vec<(usize, usize)>,
    targets: Vec<(usize, usize)>,
    pub done: bool,
    pub steps: usize,
    pub max_steps: usize,
    pub level: usize,
    pub boxes_on_target: usize,
    n_boxes: usize,
    n_targets: usize,
}

/// Niveaux prédéfinis (légende : #=mur .=cible $=caisse @=joueur *=caisse/cible)
const LEVELS: &[&[&str]] = &[
    // Niveau 1 : 5×5, 1 caisse
    &[
        "#####",
        "#@  #",
        "# $ #",
        "# . #",
        "#####",
    ],
    // Niveau 2 : 5×5, 1 caisse, mur interne
    &[
        "#####",
        "#@# #",
        "# $ #",
        "# . #",
        "#####",
    ],
    // Niveau 3 : 6×6, 2 caisses
    &[
        "######",
        "#@   #",
        "#  $ #",
        "# $  #",
        "#  . #",
        "######",
    ],
    // Niveau 4 : 7×7, 2 caisses (aliasing commence ici)
    &[
        "#######",
        "#@    #",
        "#  #  #",
        "# $$  #",
        "#  .  #",
        "#   . #",
        "#######",
    ],
    // Niveau 5 : 8×8, 3 caisses (cellules activées)
    &[
        "########",
        "#@     #",
        "#  #   #",
        "#  $   #",
        "#   $ .#",
        "#  .   #",
        "#    $ #",
        "########",
    ],
    // Niveau 6 : 8×8, 4 caisses (cellules activées)
    &[
        "########",
        "#@  #  #",
        "#  $   #",
        "#.  $  #",
        "#  $   #",
        "#   . $#",
        "#  .   #",
        "########",
    ],
];

impl Sokoban {
    pub fn targets_len(&self) -> usize { self.n_targets }
    pub fn boxes_len(&self) -> usize { self.n_boxes }

    pub fn generate(level: usize) -> Self {
        let idx = (level - 1).min(LEVELS.len() - 1);
        let lines = LEVELS[idx];
        let h = lines.len();
        let w = lines[0].len();

        let mut walls = vec![vec![false; h]; w];
        let mut boxes = Vec::new();
        let mut targets = Vec::new();
        let mut agent = (1usize, 1usize);

        for y in 0..h {
            for (x, ch) in lines[y].char_indices() {
                match ch {
                    '#' => walls[x][y] = true,
                    '@' => agent = (x, y),
                    '$' => boxes.push((x, y)),
                    '.' => targets.push((x, y)),
                    '*' => { boxes.push((x, y)); targets.push((x, y)); }
                    '+' => { agent = (x, y); targets.push((x, y)); }
                    _ => {}
                }
            }
        }

        let n_boxes = boxes.len();
        let n_targets = targets.len();
        let max_steps = (w * h * 15).max(80);

        Sokoban {
            width: w, height: h, walls, agent, boxes, targets,
            done: false, steps: 0, max_steps, level, boxes_on_target: 0,
            n_boxes, n_targets,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::generate(self.level);
    }

    pub fn is_walkable(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return false;
        }
        !self.walls[x as usize][y as usize]
    }

    fn box_at(&self, x: isize, y: isize) -> Option<usize> {
        for (i, &(bx, by)) in self.boxes.iter().enumerate() {
            if bx == x as usize && by == y as usize { return Some(i); }
        }
        None
    }

    fn is_target(&self, x: usize, y: usize) -> bool {
        self.targets.contains(&(x, y))
    }

    /// Perception : [N, S, O, E, box_adjacent, box_dir, target_sensed, cell_id?]
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

        let mut box_adj = 0.0;
        let mut box_dir = 4.0;
        for (di, (dx, dy)) in [(0, -1), (0, 1), (-1, 0), (1, 0)].iter().enumerate() {
            if self.box_at(x + dx, y + dy).is_some() {
                box_adj = 1.0;
                box_dir = di as f64;
                break;
            }
        }
        p.push(box_adj);
        p.push(box_dir / 4.0);

        // Distance à la cible la plus proche
        let mut min_dist = self.width.max(self.height) as f64;
        for &(tx, ty) in &self.targets {
            let dx = (x - tx as isize).abs() as f64;
            let dy = (y - ty as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt() / md;
            if d < min_dist { min_dist = d; }
        }
        p.push(1.0 - min_dist);

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
            if self.box_at(cx, cy).is_some() { return d; }
            d += 1;
            cx += dx;
            cy += dy;
        }
    }

    pub fn step(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;

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

        if let Some(bi) = self.box_at(nx, ny) {
            let bx = nx + dx;
            let by = ny + dy;
            if !self.is_walkable(bx, by) || self.box_at(bx, by).is_some() {
                if self.steps >= self.max_steps { self.done = true; }
                return -0.5;
            }
            let old_target = self.is_target(self.boxes[bi].0, self.boxes[bi].1);
            self.boxes[bi] = (bx as usize, by as usize);
            let new_target = self.is_target(bx as usize, by as usize);
            if !old_target && new_target { self.boxes_on_target += 1; }
            else if old_target && !new_target { self.boxes_on_target -= 1; }
        }

        self.agent = (nx as usize, ny as usize);

        if self.boxes_on_target == self.targets.len() {
            self.done = true;
            return 50.0;
        }
        if self.steps >= self.max_steps {
            self.done = true;
            return -5.0;
        }
        -0.05
    }
}
