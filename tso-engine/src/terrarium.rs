use ndarray::Array1;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
struct FoodSource {
    x: usize,
    y: usize,
    alive: bool,
    timer: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct WaterSource {
    x: usize,
    y: usize,
    alive: bool,
    timer: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Terrarium {
    pub width: usize,
    pub height: usize,
    walls: Vec<Vec<bool>>,
    pub agent: (usize, usize),
    food: Vec<FoodSource>,
    water: Vec<WaterSource>,
    pub energy: f64,
    pub done: bool,
    pub steps: usize,
    pub max_steps: usize,
    pub total_reward: f64,
    pub perishable: bool,
    pub respawn_delay: usize,
}

impl Terrarium {
    pub fn new(_seed: u64) -> Self {
        let w = 7;
        let h = 7;

        let mut walls = vec![vec![false; h]; w];
        for i in 0..w { walls[i][0] = true; walls[i][h-1] = true; }
        for j in 0..h { walls[0][j] = true; walls[w-1][j] = true; }

        for x in 2..w-1 { walls[x][3] = true; }
        for y in 1..h-1 { walls[3][y] = true; }
        walls[2][3] = false;
        walls[3][2] = false;
        walls[4][3] = false;

        let food_pos = [(2, 1), (5, 4), (1, 5)];
        let water_pos = [(5, 1), (2, 5), (4, 2)];

        let food: Vec<FoodSource> = food_pos.iter().map(|&(x, y)| FoodSource { x, y, alive: true, timer: 0 }).collect();
        let water: Vec<WaterSource> = water_pos.iter().map(|&(x, y)| WaterSource { x, y, alive: true, timer: 0 }).collect();

        Terrarium {
            width: w, height: h, walls,
            agent: (1, 1),
            food, water,
            energy: 1.0, done: false, steps: 0,
            max_steps: 200, total_reward: 0.0,
            perishable: true,
            respawn_delay: 15,
        }
    }

    pub fn reset(&mut self) {
        self.agent = (1, 1);
        self.energy = 1.0;
        self.done = false;
        self.steps = 0;
        self.total_reward = 0.0;
        for f in &mut self.food {
            f.alive = true;
            f.timer = 0;
        }
        for w in &mut self.water {
            w.alive = true;
            w.timer = 0;
        }
    }

    pub fn is_walkable(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return false;
        }
        !self.walls[x as usize][y as usize]
    }

    fn random_walkable(&self) -> (usize, usize) {
        use rand::Rng;
        loop {
            let x = rand::thread_rng().gen_range(1..self.width - 1);
            let y = rand::thread_rng().gen_range(1..self.height - 1);
            if !self.walls[x][y] && (x, y) != self.agent {
                let occupied_by_food = self.food.iter().any(|f| f.alive && f.x == x && f.y == y);
                let occupied_by_water = self.water.iter().any(|w| w.alive && w.x == x && w.y == y);
                if !occupied_by_food && !occupied_by_water {
                    return (x, y);
                }
            }
        }
    }

    fn has_food(&self, x: usize, y: usize) -> bool {
        self.food.iter().any(|f| f.alive && f.x == x && f.y == y)
    }

    fn has_water(&self, x: usize, y: usize) -> bool {
        self.water.iter().any(|w| w.alive && w.x == x && w.y == y)
    }

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

        let mut food_near = 0.0;
        for f in &self.food {
            if !f.alive { continue; }
            let dx = (x - f.x as isize).abs() as f64;
            let dy = (y - f.y as isize).abs() as f64;
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 2.0 { food_near = (1.0 - d / 3.0).max(0.0); break; }
        }
        p.push(food_near);

        let mut water_near = 0.0;
        for w in &self.water {
            if !w.alive { continue; }
            let dx = (x - w.x as isize).abs() as f64;
            let dy = (y - w.y as isize).abs() as f64;
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

    fn collect_resource_reward(&mut self, reward: f64) -> f64 {
        self.energy = (self.energy + if reward >= 10.0 { 0.3 } else { 0.2 }).min(1.0);
        self.total_reward += reward;
        if self.steps >= self.max_steps { self.done = true; }
        reward
    }

    fn consume_food(&mut self, idx: usize) -> f64 {
        self.food[idx].alive = false;
        self.food[idx].timer = self.respawn_delay;
        self.collect_resource_reward(10.0)
    }

    fn consume_water(&mut self, idx: usize) -> f64 {
        self.water[idx].alive = false;
        self.water[idx].timer = self.respawn_delay;
        self.collect_resource_reward(8.0)
    }

    fn tick_respawn(&mut self) {
        if !self.perishable { return; }

        let mut respawn_food: Vec<usize> = Vec::new();
        for (idx, f) in self.food.iter_mut().enumerate() {
            if !f.alive && f.timer > 0 {
                f.timer -= 1;
                if f.timer == 0 {
                    respawn_food.push(idx);
                }
            }
        }
        for idx in respawn_food {
            let (nx, ny) = self.random_walkable();
            self.food[idx].x = nx;
            self.food[idx].y = ny;
            self.food[idx].alive = true;
        }

        let mut respawn_water: Vec<usize> = Vec::new();
        for (idx, w) in self.water.iter_mut().enumerate() {
            if !w.alive && w.timer > 0 {
                w.timer -= 1;
                if w.timer == 0 {
                    respawn_water.push(idx);
                }
            }
        }
        for idx in respawn_water {
            let (nx, ny) = self.random_walkable();
            self.water[idx].x = nx;
            self.water[idx].y = ny;
            self.water[idx].alive = true;
        }
    }

    pub fn step(&mut self, action: usize) -> f64 {
        if self.done { return 0.0; }
        self.steps += 1;

        self.energy -= if self.perishable { 0.02 } else { 0.01 };
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
            self.tick_respawn();
            return -0.5;
        }

        self.agent = (nx as usize, ny as usize);

        if self.perishable {
            if let Some(idx) = self.food.iter().position(|f| f.alive && f.x == self.agent.0 && f.y == self.agent.1) {
                self.tick_respawn();
                return self.consume_food(idx);
            }
            if let Some(idx) = self.water.iter().position(|w| w.alive && w.x == self.agent.0 && w.y == self.agent.1) {
                self.tick_respawn();
                return self.consume_water(idx);
            }
        } else {
            if self.has_food(self.agent.0, self.agent.1) {
                self.tick_respawn();
                return self.collect_resource_reward(10.0);
            }
            if self.has_water(self.agent.0, self.agent.1) {
                self.tick_respawn();
                return self.collect_resource_reward(8.0);
            }
        }

        self.tick_respawn();

        if self.steps >= self.max_steps {
            self.done = true;
            return -1.0;
        }

        -0.02
    }
}
