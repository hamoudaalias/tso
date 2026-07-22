use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tile {
    Floor,
    Wall,
    Goal,
}

#[derive(Clone)]
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>,
    pub agent_x: usize,
    pub agent_y: usize,
    pub goal_x: usize,
    pub goal_y: usize,
    pub done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

pub const ACTIONS: [Action; 4] = [Action::Up, Action::Down, Action::Left, Action::Right];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepResult {
    Move,
    Collision,
    Goal,
}

impl GridWorld {
    pub fn new(width: usize, height: usize, walls: &[(usize, usize)]) -> Self {
        let mut tiles = vec![Tile::Floor; width * height];
        for &(x, y) in walls {
            tiles[y * width + x] = Tile::Wall;
        }

        let start_x = 0;
        let start_y = 0;
        let goal_x = width - 1;
        let goal_y = height - 1;
        tiles[goal_y * width + goal_x] = Tile::Goal;

        GridWorld {
            width,
            height,
            tiles,
            agent_x: start_x,
            agent_y: start_y,
            goal_x,
            goal_y,
            done: false,
        }
    }

    pub fn reset(&mut self) {
        self.agent_x = 0;
        self.agent_y = 0;
        self.done = false;
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        if self.done {
            return StepResult::Collision;
        }

        let (nx, ny) = match action {
            Action::Up => (self.agent_x, self.agent_y.wrapping_sub(1)),
            Action::Down => (self.agent_x, self.agent_y + 1),
            Action::Left => (self.agent_x.wrapping_sub(1), self.agent_y),
            Action::Right => (self.agent_x + 1, self.agent_y),
        };

        if nx >= self.width || ny >= self.height {
            return StepResult::Collision;
        }

        match self.tiles[ny * self.width + nx] {
            Tile::Wall => StepResult::Collision,
            Tile::Goal => {
                self.agent_x = nx;
                self.agent_y = ny;
                self.done = true;
                StepResult::Goal
            }
            Tile::Floor => {
                self.agent_x = nx;
                self.agent_y = ny;
                StepResult::Move
            }
        }
    }

    pub fn neighbour_states(&self, x: usize, y: usize) -> Vec<(Action, StepResult, usize, usize)> {
        let mut out = Vec::new();
        for &a in &ACTIONS {
            let (nx, ny) = match a {
                Action::Up => (x, y.wrapping_sub(1)),
                Action::Down => (x, y + 1),
                Action::Left => (x.wrapping_sub(1), y),
                Action::Right => (x + 1, y),
            };
            if nx >= self.width || ny >= self.height {
                out.push((a, StepResult::Collision, x, y));
                continue;
            }
            match self.tiles[ny * self.width + nx] {
                Tile::Wall => out.push((a, StepResult::Collision, x, y)),
                Tile::Goal => out.push((a, StepResult::Goal, nx, ny)),
                Tile::Floor => out.push((a, StepResult::Move, nx, ny)),
            }
        }
        out
    }
}

impl fmt::Display for GridWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                if x == self.agent_x && y == self.agent_y {
                    write!(f, "A")?;
                } else {
                    match self.tiles[y * self.width + x] {
                        Tile::Floor => write!(f, ".")?,
                        Tile::Wall => write!(f, "#")?,
                        Tile::Goal => write!(f, "G")?,
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
