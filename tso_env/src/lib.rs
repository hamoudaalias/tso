use ndarray::Array1;
use pyo3::prelude::*;
use std::cell::RefCell;

// GridWorld 5×5 (identique phase1b)
const W: usize = 5;
const H: usize = 5;
const PDIM: usize = 6;
const NA: usize = 4;
const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

/// Python-visible GridWorld environment wrapping TsoEngine.
/// `unsendable` car TsoEngine n'est pas Sync (contient RefCell / encoder Option).
#[pyclass(unsendable)]
struct TsoGridEnv {
    engine: RefCell<tso_engine::tso_engine::TsoEngine>,
    agent: (usize, usize),
    step_count: usize,
    done: bool,
    bfs: Vec<Vec<f64>>,
}

#[pymethods]
impl TsoGridEnv {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        let _seed = seed.unwrap_or(42);
        let mut engine = tso_engine::tso_engine::TsoEngine::with_hidden(PDIM, NA, 4);
        engine.cerebellum.epsilon = 0.0;
        engine.cerebellum.noise_std = 0.0;
        engine.cerebellum.replay_lr = 0.0;
        engine.sleep_every_n_episodes = 0;
        engine.use_stationary_reward = true;
        engine.cogs.delta_clip_max = 5.0;

        let bfs = compute_bfs();
        let agent = random_free_cell();

        TsoGridEnv {
            engine: RefCell::new(engine),
            agent,
            step_count: 0,
            done: false,
            bfs,
        }
    }

    fn reset(&mut self) -> (Vec<f64>, Vec<(String, f64)>) {
        self.agent = random_free_cell();
        self.step_count = 0;
        self.done = false;
        self.engine.borrow_mut().end_episode();
        (self.perceive(), vec![])
    }

    #[pyo3(signature = (action))]
    fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool, bool, Vec<(String, f64)>) {
        if self.done {
            return (self.perceive(), 0.0, true, false, vec![]);
        }

        let (dx, dy) = match action {
            0 => (0, -1), 1 => (0, 1), 2 => (-1, 0), 3 => (1, 0),
            _ => (0, 0),
        };
        let nx = self.agent.0 as isize + dx;
        let ny = self.agent.1 as isize + dy;

        let reward: f64;
        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            reward = -0.5;
            self.step_count += 1;
            if self.step_count >= MAXS { self.done = true; }
        } else {
            self.agent = (nx as usize, ny as usize);
            self.step_count += 1;
            if WATER.contains(&self.agent) {
                reward = 10.0;
                self.done = true;
            } else if self.step_count >= MAXS {
                reward = -1.0;
                self.done = true;
            } else {
                reward = -0.02;
            }
        }

        let obs = self.perceive();
        let p = Array1::from_vec(obs.clone());
        self.engine.borrow_mut().step(&p, reward, Some(self.bfs[self.agent.0][self.agent.1]), &[]);

        if self.done {
            self.engine.borrow_mut().end_episode();
        }

        (obs, reward, self.done, false, vec![])
    }

    fn render(&self) {
        for y in 0..H {
            for x in 0..W {
                if self.agent == (x, y) {
                    eprint!("A ");
                } else if WATER.contains(&(x, y)) {
                    eprint!("~ ");
                } else {
                    eprint!(". ");
                }
            }
            eprintln!();
        }
        eprintln!("Step: {}, done: {}", self.step_count, self.done);
    }

    #[getter]
    fn get_agent(&self) -> (usize, usize) {
        self.agent
    }

    fn get_info(&self) -> (usize, usize, Vec<(usize, usize)>) {
        (W, H, WATER.to_vec())
    }
}

impl TsoGridEnv {
    fn perceive(&self) -> Vec<f64> {
        let (x, y) = self.agent;
        let ix = x as isize;
        let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0;
            let mut cx = ix + dx;
            let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < W as isize && cy < H as isize {
                d += 1;
                cx += dx;
                cy += dy;
            }
            d as f64 / (W.max(H) as f64)
        };
        let mut ws = 0.0;
        for &(wx, wy) in &WATER {
            let d = (((ix - wx as isize).abs().pow(2) + (iy - wy as isize).abs().pow(2)) as f64).sqrt();
            if d <= 2.0 {
                ws = (1.0 - d / 3.0).max(0.0);
                break;
            }
        }
        vec![ray(0, -1), ray(0, 1), ray(-1, 0), ray(1, 0), 0.0, ws]
    }
}

fn random_free_cell() -> (usize, usize) {
    use rand::Rng;
    loop {
        let x = rand::thread_rng().r#gen_range(0..W);
        let y = rand::thread_rng().r#gen_range(0..H);
        if !WATER.contains(&(x, y)) {
            return (x, y);
        }
    }
}

fn compute_bfs() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let dm = ((W - 1) + (H - 1)) as f64;
    let mut pot = vec![vec![0.0; H]; W];
    let mut dist = vec![vec![None::<usize>; H]; W];
    let mut q = VecDeque::new();
    for &(wx, wy) in &WATER {
        dist[wx][wy] = Some(0);
        q.push_back((wx, wy));
    }
    while let Some((cx, cy)) = q.pop_front() {
        let dd = dist[cx][cy].unwrap();
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                if dist[nx][ny].is_none() {
                    dist[nx][ny] = Some(dd + 1);
                    q.push_back((nx, ny));
                }
            }
        }
    }
    for x in 0..W {
        for y in 0..H {
            pot[x][y] = match dist[x][y] {
                Some(dd) => -2.5 * dd as f64 / dm,
                None => -2.5,
            };
        }
    }
    pot
}

#[pymodule]
fn tso_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TsoGridEnv>()?;
    Ok(())
}
