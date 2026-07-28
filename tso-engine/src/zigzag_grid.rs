/// GridWorld Zigzag 10×10 avec reward stationnaire pour benchmark attention.
///
/// Aliasing maximal : deux positions distantes (ex: couloir horizontal puis vertical)
/// peuvent produire la même lecture 4D de moustaches. L'attention spatiale devrait
/// aider à désambiguïser en amplifiant les directions où la prédiction épisodique
/// diffère de la perception.
use ndarray::Array1;

const N_ACTIONS: usize = 4;
const MAX_STEPS: usize = 200;
const GOAL: (usize, usize) = (8, 8);
const START: (usize, usize) = (1, 1);
const W: usize = 10;
const H: usize = 10;

/// Layout Zigzag : mur au milieu (ligne d'obstacles colonne 4, trou au milieu).
fn is_zigzag_wall(x: usize, y: usize) -> bool {
    // Bordure
    if x == 0 || y == 0 || x == W - 1 || y == H - 1 { return true; }
    // Mur central horizontal entre y=3 et y=6, avec trou à x=5
    if y >= 3 && y <= 6 && (x == 4 || x == 6) {
        if y == 4 || y == 5 { return false; } // trous
        return true;
    }
    false
}

pub struct ZigzagGrid {
    pub agent: (usize, usize),
    pub step: usize,
    pub done: bool,
    pub bfs: Vec<Vec<usize>>, // distance BFS depuis chaque cellule vers GOAL
}

impl ZigzagGrid {
    pub fn new() -> Self {
        let mut grid = ZigzagGrid {
            agent: START,
            step: 0,
            done: false,
            bfs: vec![vec![0; W]; H],
        };
        grid.compute_bfs();
        grid
    }

    fn compute_bfs(&mut self) {
        let mut dist = vec![vec![usize::MAX; W]; H];
        let mut queue = std::collections::VecDeque::new();
        dist[GOAL.1][GOAL.0] = 0;
        queue.push_back(GOAL);
        while let Some((cx, cy)) = queue.pop_front() {
            for (dx, dy) in [(0,1),(0,-1),(1,0),(-1,0)] {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                    let (nxu, nyu) = (nx as usize, ny as usize);
                    if !is_zigzag_wall(nxu, nyu) && dist[nyu][nxu] == usize::MAX {
                        dist[nyu][nxu] = dist[cy][cx] + 1;
                        queue.push_back((nxu, nyu));
                    }
                }
            }
        }
        self.bfs = dist;
    }

    pub fn perceive(&self) -> Array1<f64> {
        let (x, y) = self.agent;
        let ix = x as isize;
        let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0;
            let mut cx = ix + dx;
            let mut cy = iy + dy;
            loop {
                if cx < 0 || cy < 0 || cx >= W as isize || cy >= H as isize { break; }
                let (cxu, cyu) = (cx as usize, cy as usize);
                if is_zigzag_wall(cxu, cyu) { break; }
                d += 1;
                cx += dx;
                cy += dy;
            }
            d as f64 / (W.max(H) as f64)
        };
        let bfs_norm = if self.bfs[y][x] == usize::MAX {
            1.0
        } else {
            self.bfs[y][x] as f64 / (W * H) as f64
        };
        Array1::from_vec(vec![ray(0, -1), ray(0, 1), ray(-1, 0), ray(1, 0), bfs_norm])
    }

    pub fn reset(&mut self) -> Array1<f64> {
        self.agent = START;
        self.step = 0;
        self.done = false;
        self.perceive()
    }

    pub fn step_env(&mut self, action: usize) -> (f64, Array1<f64>) {
        if self.done { return (0.0, self.perceive()); }
        self.step += 1;
        let (dx, dy) = match action { 0 => (0,-1), 1 => (0,1), 2 => (-1,0), 3 => (1,0), _ => (0,0) };
        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;
        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            return (-0.5, self.perceive());
        }
        let (nxu, nyu) = (nx as usize, ny as usize);
        if is_zigzag_wall(nxu, nyu) { return (-0.5, self.perceive()); }
        self.agent = (nxu, nyu);
        if self.agent == GOAL || self.step >= MAX_STEPS {
            self.done = true;
            let reward = if self.agent == GOAL { 20.0 } else { -1.0 };
            (reward, self.perceive())
        } else {
            // Shaping par BFS
            let bfs_here = if self.bfs[self.agent.1][self.agent.0] == usize::MAX {
                (W*H) as f64
            } else {
                self.bfs[self.agent.1][self.agent.0] as f64
            };
            let reward = -0.01 + 0.0 * (bfs_here); // step_flat
            (reward, self.perceive())
        }
    }
}
