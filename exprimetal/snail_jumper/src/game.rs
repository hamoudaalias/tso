use rand::Rng;

#[derive(Clone)]
pub struct Obstacle {
    pub x: f64,
    pub y: f64,
}

pub struct GameState {
    pub player_x: f64,
    pub screen_w: f64,
    pub screen_h: f64,
    pub obstacles: Vec<Obstacle>,
    pub score: u64,
    pub alive: bool,
    pub frame: u64,
    spawn_timer: u64,
    speed: f64,
}

impl GameState {
    pub fn new(screen_w: f64, screen_h: f64) -> Self {
        GameState {
            player_x: screen_w / 2.0,
            screen_w,
            screen_h,
            obstacles: Vec::new(),
            score: 0,
            alive: true,
            frame: 0,
            spawn_timer: 0,
            speed: 3.0,
        }
    }

    pub fn step(&mut self, action_left: bool) {
        if !self.alive { return; }
        self.frame += 1;

        if action_left {
            self.player_x = (self.player_x - 4.0).max(10.0);
        } else {
            self.player_x = (self.player_x + 4.0).min(self.screen_w - 10.0);
        }

        self.spawn_timer += 1;
        let mut rng = rand::thread_rng();
        if self.spawn_timer > 20 + rng.gen_range(0..20) {
            self.spawn_timer = 0;
            self.obstacles.push(Obstacle {
                x: rng.gen_range(10.0..self.screen_w - 10.0),
                y: -20.0,
            });
        }

        for obs in &mut self.obstacles {
            obs.y += self.speed;
        }
        self.obstacles.retain(|o| o.y < self.screen_h + 20.0);

        for obs in &self.obstacles {
            let dx = self.player_x - obs.x;
            let dy = self.screen_h - 50.0 - obs.y;
            if dx.abs() < 15.0 && dy.abs() < 20.0 {
                self.alive = false;
                break;
            }
        }

        if self.alive {
            self.score = self.frame / 6;

        if self.frame > 5000 {
            self.alive = false;
        }
        }
    }

    pub fn nearest_obstacle(&self) -> Option<&Obstacle> {
        self.obstacles.iter()
            .filter(|o| o.y < self.screen_h - 30.0)
            .min_by(|a, b| (a.y).partial_cmp(&b.y).unwrap())
    }
}
