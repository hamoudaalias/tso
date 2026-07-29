/// Comparaison : AttractorField pur (perception 6D) vs PCA (image 25D → latent 4D → AttractorField)
/// sur GridWorld 5×5 avec récompense d'eau.
use tso_engine::tso_engine::TsoEngine;
use tso_engine::environment::{Environment, StepResult};
use tso_engine::encoder::{Encoder, EncodeResult};
use ndarray::Array1;
use std::time::Instant;

// ─── GridWorld 5×5 (render 25D) ───────────────────────────────────────────

const W: usize = 5; const H: usize = 5; const NA: usize = 4; const MAXS: usize = 150;
const WATER: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];

struct GridWorld25 {
    agent: (usize, usize), step: usize, done: bool,
}
impl GridWorld25 {
    fn render(&self) -> Array1<f64> {
        let mut grid = vec![0.0; 25];
        for x in 0..W { for y in 0..H {
            if x == 0 || x == W-1 || y == 0 || y == H-1 { grid[y*W+x] = 0.5; }
        }}
        for &(wx, wy) in &WATER { grid[wy*W+wx] = 1.0; }
        grid[self.agent.1 * W + self.agent.0] = 2.0;
        Array1::from_vec(grid)
    }
}
impl Environment for GridWorld25 {
    fn reset(&mut self) -> Array1<f64> {
        self.agent = (2, 2); self.step = 0; self.done = false; self.render()
    }
    fn step(&mut self, action: usize) -> StepResult {
        if self.done { return StepResult { observation: self.render(), reward: 0.0, done: true }; }
        self.step += 1;
        let (dx, dy) = match action { 0 => (0,-1), 1 => (0,1), 2 => (-1,0), 3 => (1,0), _ => (0,0) };
        let nx = self.agent.0 as isize + dx; let ny = self.agent.1 as isize + dy;
        if nx < 0 || ny < 0 || nx >= W as isize || ny >= H as isize {
            return StepResult { observation: self.render(), reward: -0.5, done: self.step >= MAXS };
        }
        self.agent = (nx as usize, ny as usize);
        if WATER.contains(&self.agent) {
            self.done = true; StepResult { observation: self.render(), reward: 10.0, done: true }
        } else if self.step >= MAXS {
            self.done = true; StepResult { observation: self.render(), reward: -1.0, done: true }
        } else { StepResult { observation: self.render(), reward: -0.02, done: false } }
    }
    fn action_space(&self) -> usize { NA }
    fn observation_dim(&self) -> usize { 25 }
}

// ─── PcaEncoder 25D → 4D ──────────────────────────────────────────────────

struct PcaEncoder {
    mean: Vec<f64>,
    components: Vec<Vec<f64>>,  // [latent_dim × input_dim]
    novelty_threshold: f64,
    centroids: Vec<Vec<f64>>,   // same as VaeEncoder
}
impl PcaEncoder {
    fn new(path: &str, threshold: f64) -> Self {
        let raw = std::fs::read(path).expect("pca file not found");
        let input_dim = 25; let latent_dim = 4;
        let mut off = 0;
        let mut mean = vec![0.0; input_dim];
        for j in 0..input_dim {
            mean[j] = f64::from_le_bytes(raw[off..off+8].try_into().unwrap());
            off += 8;
        }
        let mut components = Vec::with_capacity(latent_dim);
        for _ in 0..latent_dim {
            let mut comp = vec![0.0; input_dim];
            for j in 0..input_dim {
                comp[j] = f64::from_le_bytes(raw[off..off+8].try_into().unwrap());
                off += 8;
            }
            components.push(comp);
        }
        PcaEncoder { mean, components, novelty_threshold: threshold, centroids: Vec::new() }
    }
}
impl Encoder for PcaEncoder {
    fn encode_raw(&mut self, perception: &Array1<f64>) -> EncodeResult {
        // Project 25D → 4D latent
        let mut latent = vec![0.0; 4];
        for k in 0..4 {
            for j in 0..25 { latent[k] += self.components[k][j] * (perception[j] - self.mean[j]); }
        }
        // Centroid matching (same as VaeEncoder)
        if self.centroids.is_empty() {
            self.centroids.push(latent);
            return EncodeResult { category_id: 0, novelty: 0.0, is_new: true };
        }
        let mut best_idx = 0;
        let mut best_dist = latent.iter().zip(self.centroids[0].iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        for (i, c) in self.centroids.iter().enumerate().skip(1) {
            let d = latent.iter().zip(c.iter()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
            if d < best_dist { best_dist = d; best_idx = i; }
        }
        if best_dist > self.novelty_threshold {
            let new_id = self.centroids.len();
            self.centroids.push(latent);
            EncodeResult { category_id: new_id, novelty: best_dist, is_new: true }
        } else {
            EncodeResult { category_id: best_idx, novelty: best_dist, is_new: false }
        }
    }
    fn n_categories(&self) -> usize { self.centroids.len() }
}

// ─── GridWorld 5×5 (perception 6D classique) ──────────────────────────────

struct GridWorld6 {
    inner: GridWorld25,
}
impl GridWorld6 {
    fn perceive(&self) -> Array1<f64> {
        let (x, y) = self.inner.agent;
        let ix = x as isize; let iy = y as isize;
        let ray = |dx: isize, dy: isize| -> f64 {
            let mut d = 0; let mut cx = ix + dx; let mut cy = iy + dy;
            while cx >= 0 && cy >= 0 && cx < W as isize && cy < H as isize {
                d += 1; cx += dx; cy += dy;
            }
            d as f64 / (W.max(H) as f64)
        };
        let mut ws = 0.0;
        for &(wx, wy) in &WATER {
            let d = (((ix - wx as isize).abs().pow(2) + (iy - wy as isize).abs().pow(2)) as f64).sqrt();
            if d <= 2.0 { ws = (1.0 - d / 3.0).max(0.0); break; }
        }
        Array1::from_vec(vec![ray(0,-1), ray(0,1), ray(-1,0), ray(1,0), 0.0, ws])
    }
}
impl Environment for GridWorld6 {
    fn reset(&mut self) -> Array1<f64> { self.inner.reset(); self.perceive() }
    fn step(&mut self, action: usize) -> StepResult {
        let r = self.inner.step(action);
        StepResult { observation: self.perceive(), ..r }
    }
    fn action_space(&self) -> usize { NA }
    fn observation_dim(&self) -> usize { 6 }
}

// ─── Evaluateur ────────────────────────────────────────────────────────────

fn evaluate(mut env: Box<dyn Environment>, mut engine: TsoEngine, label: &str) -> f64 {
    let t0 = Instant::now();
    let mut successes = 0;
    let ep = 200;

    for ep_i in 1..=ep {
        let mut obs = env.reset(); engine.end_episode();
        loop {
            let action = engine.step(&obs, 0.0, None, &[]);
            let r = env.step(action);
            if r.done {
                if r.reward > 0.0 { successes += 1; }
                engine.end_episode();
                break;
            }
            obs = r.observation;
        }
        // ε-annealing
        if ep_i < 100 {
            let frac = ep_i as f64 / 100.0;
            engine.cerebellum.epsilon = 0.8 * (1.0 - frac);
        } else { engine.cerebellum.epsilon = 0.01; }
    }
    let rate = successes as f64 / ep as f64 * 100.0;
    eprintln!("  {:<40} {:3}/{} = {:.1}% [{:.1?}]", label, successes, ep, rate, t0.elapsed());
    rate
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  GridWorld 5×5 — VAE/PCA (image 25D) vs Attractor pur (6D whiskers) ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Config commune
    let n_actions = 4; let hidden = 16;

    // 1) Baseline : AttractorField pur sur perception 6D
    {
        let env = Box::new(GridWorld6 { inner: GridWorld25 { agent: (2,2), step: 0, done: false }});
        let mut engine = TsoEngine::with_hidden(6, n_actions, hidden);
        engine.use_stationary_reward = true;
        engine.cogs.delta_clip_max = 5.0;
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.sleep_every_n_episodes = 0;
        evaluate(env, engine, "Baseline: AttractorField 6D");
    }

    // 2) PCA : image 25D → latent 4D → centroids → TSO
    {
        let env = Box::new(GridWorld25 { agent: (2,2), step: 0, done: false });
        let mut engine = TsoEngine::with_hidden(25, n_actions, hidden);
        engine.encoder = Some(Box::new(PcaEncoder::new("pca_gridworld.bin", 0.5)));
        engine.use_stationary_reward = true;
        engine.cogs.delta_clip_max = 5.0;
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.sleep_every_n_episodes = 0;
        evaluate(env, engine, "PCA 25D→4D + centroids");
    }
    // 3) VAE : image 25D → latent 4D → centroids → TSO
    {
        let env = Box::new(GridWorld25 { agent: (2,2), step: 0, done: false });
        let mut engine = TsoEngine::with_hidden(25, n_actions, hidden);
        let vae_bytes = std::fs::read("vae_gridworld.bin").unwrap_or_default();
        if !vae_bytes.is_empty() {
            let vae: tso_engine::vae::Vae = bincode::deserialize(&vae_bytes).unwrap();
            let mut vae_enc = tso_engine::encoder::VaeEncoder::new(25, 16, 4, 0.5);
            vae_enc.vae = vae;
            vae_enc.deterministic = true;
            vae_enc.freeze = true;
            engine.encoder = Some(Box::new(vae_enc));
        }
        engine.use_stationary_reward = true;
        engine.cogs.delta_clip_max = 5.0;
        engine.cerebellum.epsilon = 0.8;
        engine.cerebellum.noise_std = 0.3;
        engine.cerebellum.replay_lr = 0.05;
        engine.cerebellum.replay_only = true;
        engine.sleep_every_n_episodes = 0;
        evaluate(env, engine, "VAE 25→16→4 + freeze centroids");
    }
}
