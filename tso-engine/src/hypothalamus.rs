use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Hypothalamus {
    pub energy: f64,
    pub hydration: f64,
    pub temperature: f64,
    drift_rate: f64,
    /// Cognitive tension (Φ) from the semantic graph — a drive to reduce conflict.
    pub phi: f64,
    /// Sleep debt — accumulates while awake, discharged during sleep [0, 1+].
    /// Drives the organism to rest when high.
    pub sleep_debt: f64,
    pub sleep_drift_rate: f64,
    /// Base metabolic multiplier for all cognitive computation costs.
    pub metabolic_rate: f64,
    /// Last tick's cerebellum computation cost (before habit discount).
    pub cerebellum_cost: f64,
    /// Last tick's graph computation cost (before habit discount).
    pub graph_cost: f64,
    /// Last tick's motor (locomotion) energy cost.
    pub motor_cost: f64,
    /// Motor cost rate — energy consumed per tick for physical movement.
    pub motor_cost_rate: f64,
    /// Last tick's total metabolic energy drain.
    pub total_cost: f64,
}

impl Hypothalamus {
    pub fn new() -> Self {
        Hypothalamus {
            energy: 1.0,
            hydration: 1.0,
            temperature: 0.5,
            drift_rate: 0.005,
            phi: 0.0,
            sleep_debt: 0.0,
            sleep_drift_rate: 0.001,
            metabolic_rate: 0.001,
            cerebellum_cost: 0.0,
            graph_cost: 0.0,
            motor_cost: 0.0,
            motor_cost_rate: 0.001,
            total_cost: 0.0,
        }
    }

    /// Each step, homeostatic variables drift (simulating constant metabolic need).
    /// Uses the default step-based drift.
    pub fn step(&mut self) {
        self.step_dt(0.1);
    }

    /// Real-time drift: variables decay proportional to `dt` (seconds since last tick).
    /// Base drift rate = 0.05 / second (Energy and Hydration go from 1.0 → 0.0 in 20 s).
    /// Sleep debt accumulates ~10× slower than energy depletion.
    pub fn step_dt(&mut self, dt: f64) {
        let drift = self.drift_rate * dt * 10.0;
        self.energy = (self.energy - drift).max(0.0);
        self.hydration = (self.hydration - drift).max(0.0);
        self.temperature += (rand::random::<f64>() - 0.5) * drift * 2.0;
        self.temperature = self.temperature.clamp(0.0, 1.0);
        self.sleep_debt += self.sleep_drift_rate * dt * 10.0;
    }

    /// Perceived value of an external reward, modulated by homeostatic deficits.
    ///
    /// A satiated organism perceives reward at face value.
    /// A deprived organism perceives amplified reward — the deficit makes
    /// the reward more salient (hunger amplifies the value of food).
    /// When deficits are zero the reward passes through unchanged.
    pub fn gate_reward(&self, external_reward: f64) -> f64 {
        let deficit = self.total_deficit();
        let amp = 1.0 + deficit * 2.0;
        external_reward * amp
    }

    /// Consummatory satisfaction when a reward is received.
    /// The satisfaction is proportional to the deficit being reduced —
    /// eating when hungry is itself pleasurable.
    pub fn consummatory_value(&self, external_reward: f64) -> f64 {
        if external_reward > 0.0 {
            self.total_deficit() * 10.0
        } else {
            0.0
        }
    }

    /// Consume resources: reduce deficits when a reward is received,
    /// simulating the effect of eating (energy), drinking (hydration),
    /// or resting (temperature).
    pub fn consume(&mut self, reward: f64) {
        if reward > 0.0 {
            if reward >= 20.0 {
                self.energy = 1.0;
                self.hydration = 1.0;
                self.temperature = 0.5;
            } else {
                let e_def = (0.5 - self.energy).max(0.0);
                let h_def = (0.5 - self.hydration).max(0.0);
                let t_dev = (self.temperature - 0.5).abs();
                self.energy += e_def * 0.3;
                self.hydration += h_def * 0.1;
                if self.temperature > 0.5 {
                    self.temperature -= t_dev * 0.1;
                } else {
                    self.temperature += t_dev * 0.1;
                }
                self.energy = self.energy.min(1.0);
                self.hydration = self.hydration.min(1.0);
                self.temperature = self.temperature.clamp(0.0, 1.0);
            }
        }
    }

    /// Apply metabolic cost of cognitive computation + motor action to energy.
    /// `cerebellum_cost` and `graph_cost` are raw complexity costs.
    /// `habit_efficiency` in [0, 1] reduces graph cost for well-rehearsed paths.
    /// Motor cost is a flat per-tick locomotion expense.
    pub fn apply_metabolic_cost(&mut self, cerebellum_cost: f64, graph_cost: f64, habit_efficiency: f64) {
        self.cerebellum_cost = cerebellum_cost;
        self.graph_cost = graph_cost;
        self.motor_cost = self.motor_cost_rate;
        let total = (cerebellum_cost + graph_cost * (1.0 - habit_efficiency * 0.5)) * self.metabolic_rate
            + self.motor_cost;
        self.total_cost = total;
        self.energy = (self.energy - total).max(0.0);
    }

    /// Set Φ — the current cognitive tension from the semantic graph.
    pub fn set_phi(&mut self, phi: f64) {
        self.phi = phi;
    }

    /// Reset sleep debt after a sleep cycle.
    pub fn reset_sleep(&mut self) {
        self.sleep_debt = 0.0;
    }

    /// Sleep drive intensity [0, 1+] — motivates the organism to rest.
    pub fn sleep_drive(&self) -> f64 {
        self.sleep_debt.min(1.0)
    }

    /// Total homeostatic deficit (without Φ, but includes sleep debt).
    pub fn total_deficit(&self) -> f64 {
        let e_def = (0.5 - self.energy).max(0.0);
        let h_def = (0.5 - self.hydration).max(0.0);
        let t_def = (self.temperature - 0.5).abs();
        let s_def = self.sleep_debt.min(1.0) * 0.5;
        e_def + h_def + t_def + s_def
    }

    /// Primary (most urgent) deficit.
    pub fn primary_deficit(&self) -> f64 {
        let e_def = (0.5 - self.energy).max(0.0);
        let h_def = (0.5 - self.hydration).max(0.0);
        let t_def = (self.temperature - 0.5).abs();
        e_def.max(h_def).max(t_def)
    }

    /// Compound drive signal: homeostatic deficits + Φ-derived tension.
    /// The organism seeks to minimise this total drive.
    pub fn total_drive(&self) -> f64 {
        self.total_deficit() + self.phi * 0.5 + self.sleep_drive()
    }

    /// Sleep debt as fraction [0, 1] for display.
    pub fn sleep_pressure(&self) -> f64 {
        self.sleep_debt.min(1.0)
    }

    pub fn homeostatic_state(&self) -> Vec<f64> {
        vec![self.energy, self.hydration, self.temperature]
    }
}
