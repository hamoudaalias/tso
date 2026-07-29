/// Benchmark e09 : 3 configs d'encodeur sur Zigzag 10×10
///
/// (a) AttractorField (baseline — pas d'encoder)
/// (b) VAE gelé + centroids (encodage discret actuel)
/// (c) VAE continu + gradient RL (apprentissage conjoint end-to-end)
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;
use tso_engine::encoder::{Encoder, VaeEncoder};
use tso_engine::CognitiveConfig;
use tso_engine::zigzag_grid::ZigzagGrid;

/// Crée un TsoEngine pour le benchmark.
/// `cerebellum_dim` = dimension de l'entrée du cervelet (5 pour perception brute,
/// 4 pour latent VAE continu).
fn make_engine(cerebellum_dim: usize) -> TsoEngine {
    // On crée l'engine avec dim=5 (perception brute) mais on recrée
    // le cerebellum avec la dimension latente voulue.
    let mut engine = TsoEngine::with_hidden(5, 4, 16);
    // Réduire le cerebellum à latent_dim si VAE (pas toucher à engine.dim)
    if cerebellum_dim != 5 {
        engine.cerebellum = tso_engine::cerebellum::Cerebellum::new(
            cerebellum_dim, 4, 0.30, 0.1, 0.50, 16);
    }
    engine.cogs = CognitiveConfig {
        attractor: true,
        graph_phi: false,
        attention: false,
        episodic_curiosity: false,
        metabolic_cost: false,
        hypothalamus: false,
        delta_clip_max: 5.0,
        ..CognitiveConfig::default()
    };
    engine.cerebellum.epsilon = 0.8;
    engine.cerebellum.noise_std = 0.3;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = true;
    engine.use_stationary_reward = true;
    engine.sleep_every_n_episodes = 0;
    engine
}

fn run_trial<C>(config_name: &str, cerebellum_dim: usize, configure: C, seeds: usize, episodes_per_seed: usize) -> (f64, f64)
where
    C: Fn(&mut TsoEngine),
{
    let mut rates = Vec::with_capacity(seeds);
    for seed in 0..seeds {
        let mut env = ZigzagGrid::new();
        let mut engine = make_engine(cerebellum_dim);

        // Appliquer la config spécifique
        configure(&mut engine);

        let t0 = Instant::now();
        let mut successes = 0;
        for ep_i in 1..=episodes_per_seed {
            let mut obs = env.reset();
            engine.end_episode();
            loop {
                let action = engine.step(&obs, 0.0, None, &[]);
                let (rew, next_obs) = env.step_env(action);
                if env.done {
                    if rew > 0.0 { successes += 1; }
                    engine.end_episode();
                    break;
                }
                obs = next_obs;
            }
            // Annealing ε
            let frac = (ep_i as f64 / (episodes_per_seed as f64 * 0.5)).min(1.0);
            engine.cerebellum.epsilon = 0.8 * (1.0 - frac * 0.9875);
        }
        let rate = successes as f64 / episodes_per_seed as f64 * 100.0;
        rates.push(rate);
        eprintln!("  {} seed {}/{}... {:.1}% [{:.1?}]",
            config_name, seed + 1, seeds, rate, t0.elapsed());
    }
    let mean = rates.iter().sum::<f64>() / seeds as f64;
    let std = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / seeds as f64).sqrt();
    (mean, std)
}

fn init_vae_encoder() -> VaeEncoder {
    let mut enc = VaeEncoder::new(5, 8, 4, 0.5);
    enc.deterministic = true;
    enc
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  EPIC e09 — Benchmark encodeur sur Zigzag 10×10                      ║");
    eprintln!("║  (a) AttractorField  (b) VAE gelé  (c) VAE conjoint RL              ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    let seeds = 10;
    let episodes = 200;

    // (a) AttractorField baseline — pas d'encoder, perception 5D brute
    let (m_a, s_a) = run_trial("Baseline AF", 5, |_| {}, seeds, episodes);

    // (b) VAE gelé + centroids — encode_raw classique, freeze=true
    let (m_b, s_b) = run_trial("VAE gelé", 4, |engine| {
        let mut vae = init_vae_encoder();
        vae.freeze = true;
        engine.encoder = Some(Box::new(vae));
    }, seeds, episodes);

    // (c) VAE continu + gradient RL — encode_continuous + backprop_td_grad
    let (m_c, s_c) = run_trial("VAE conjoint", 4, |engine| {
        let mut vae = init_vae_encoder();
        vae.freeze = false;      // ← le VAE apprend du gradient RL
        engine.encoder = Some(Box::new(vae));
    }, seeds, episodes);

    // ── Results ────────────────────────────────────────────────────────────
    eprintln!("\n  {}", "=".repeat(60));
    eprintln!("  RÉSULTATS ({} seeds × {} episodes = {} total) :", seeds, episodes, seeds * episodes);
    eprintln!("  {}", "=".repeat(60));
    eprintln!("  (a) AttractorField (baseline)    μ={:6.1}%  σ={:.2}%", m_a, s_a);
    eprintln!("  (b) VAE gelé + centroids         μ={:6.1}%  σ={:.2}%", m_b, s_b);
    eprintln!("  (c) VAE conjoint RL              μ={:6.1}%  σ={:.2}%", m_c, s_c);
    eprintln!();

    let gain_ba = m_b - m_a;
    let gain_ca = m_c - m_a;
    let gain_cb = m_c - m_b;

    eprintln!("  Différences :");
    eprintln!("    VAE gelé vs Baseline       {:+.1}%", gain_ba);
    eprintln!("    VAE conjoint vs Baseline    {:+.1}%", gain_ca);
    eprintln!("    VAE conjoint vs VAE gelé    {:+.1}%", gain_cb);
    eprintln!();

    let mut verdict = String::new();
    if gain_cb > 3.0 {
        verdict.push_str("✅ Apprentissage conjoint VAE+RL améliore significativement — l'epic e09 est validé.");
    } else if gain_cb > 1.0 {
        verdict.push_str("🟡 Apprentissage conjoint VAE+RL améliore légèrement — utile mais modeste.");
    } else if gain_cb < -3.0 {
        verdict.push_str("❌ L'apprentissage conjoint dégrade — le VAE freeze reste meilleur.");
    } else {
        verdict.push_str("⏸️ Différence négligeable — VAE gelé ou conjoint, c'est équivalent.");
    }
    eprintln!("  Verdict : {}", verdict);

    // Recommandation selon le meilleur
    let best = if m_a >= m_b && m_a >= m_c { "(a) AttractorField" }
              else if m_b >= m_c { "(b) VAE gelé + centroids" }
              else { "(c) VAE conjoint RL" };
    eprintln!("  Meilleure config : {}", best);
}
