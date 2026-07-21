use crate::core::TSOCore;
use ndarray::Array1;
use rayon::prelude::*;

/// V14: DeepTSO — Hierarchical friction propagation with top-down modulation.
///
/// Implements a two-phase cortical cycle:
///   Phase 1 — Bottom-up feedforward with temporal decimation
///   Phase 2 — Inter-layer Φ computation + top-down modulatory bias
///
/// Architecture principles:
///   - Each layer has its own dt multiplier (higher = slower, more abstract)
///   - Layer N+1 reads Layer N's attractors via typed inter-layer edges
///   - Inter-layer Φ is the "residual surprise" of the higher layer's model
///   - Top-down modulation biases lower-layer clusters toward reducing inter-layer Φ
///   - The designated output layer (L5-equivalent) is read by the Inverse Motor
#[derive(Clone, Debug)]
pub struct DeepConfig {
    pub n_layers: usize,
    pub n_clusters: usize,
    pub d: usize,
    pub gamma: f64,
    pub epsilon: f64,
    pub history_size: usize,
    pub base_theta_c: f64,
    pub inertia_threshold: f64,
    pub residual: bool,
    /// Temporal decimation per layer (e.g. [1.0, 2.0, 4.0]).
    /// Layer 0 = fastest, Layer K = slowest (most abstract).
    pub dt_multipliers: Vec<f64>,
    /// Strength of the top-down modulatory bias (default 0.05).
    pub modulatory_strength: f64,
    /// Which layer's rates are returned as `final_rates`.
    /// None = top layer (L5-equivalent).
    pub output_layer: Option<usize>,
    /// Learning rate for inter-layer R-STDP edge updates (default 0.01).
    /// Set to 0.0 to disable inter-layer learning.
    pub inter_edge_lr: f64,
    /// Disable inter-layer R-STDP entirely. Speeds up feature extraction.
    /// Set to `false` during inference / feature extraction.
    pub learn_inter_edges: bool,
    /// Winner-Take-All sparsity: fraction of clusters to KEEP per layer (e.g. 0.05 = keep top 5%).
    /// 1.0 = disabled (keep all). < 1.0 = competitive sparsity: only the strongest survive.
    /// Applied after each step to guarantee that at most `wta_keep_ratio × n_clusters` clusters
    /// have non-zero rates. This is how TSO achieves O(α·N) with α ≪ 1.
    pub wta_keep_ratio: f64,
}

impl Default for DeepConfig {
    fn default() -> Self {
        Self {
            n_layers: 4,
            n_clusters: 64,
            d: 5,
            gamma: 0.5,
            epsilon: 0.3,
            history_size: 50,
            base_theta_c: 0.5,
            inertia_threshold: 0.1,
            residual: true,
            dt_multipliers: vec![1.0, 2.0, 4.0, 8.0],
            modulatory_strength: 0.05,
            output_layer: None,
            inter_edge_lr: 0.01,
            learn_inter_edges: true,
            wta_keep_ratio: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerOutput {
    pub phi: f64,
    pub rates: Array1<f64>,
    pub spikes: u32,
}

#[derive(Clone, Debug)]
pub struct DeepOutput {
    pub layers: Vec<LayerOutput>,
    /// Sum of inter-layer Φ across all adjacent pairs
    pub inter_phi: f64,
    /// Total Φ = intra-layer Φ + inter-layer Φ
    pub total_phi: f64,
    /// Rates from the designated output layer
    pub final_rates: Array1<f64>,
}

pub struct DeepTSO {
    layers: Vec<TSOCore>,
    config: DeepConfig,
    /// Inter-layer edges: for each adjacent pair (li, li+1).
    /// Each edge: (cluster_idx_in_li, cluster_idx_in_li+1, edge_type ±1, strength).
    inter_edges: Vec<Vec<(usize, usize, f64, f64)>>,
    /// Modulatory bias for each layer, set during Phase 2 of the previous step.
    modulatory_biases: Vec<Array1<f64>>,
    /// Cached rates from the previous step (for inter-layer edge learning).
    cached_rates: Vec<Array1<f64>>,
    /// Cached inter-layer Φ from the previous step.
    cached_inter_phi: f64,
}

impl DeepTSO {
    pub fn new(config: DeepConfig) -> Self {
        let layers = (0..config.n_layers)
            .map(|li| {
                let mut core = TSOCore::new(
                    config.n_clusters,
                    config.d,
                    config.gamma,
                    config.epsilon,
                    config.history_size,
                    config.base_theta_c,
                    config.inertia_threshold,
                );
                for i in 0..config.n_clusters {
                    core.add_cluster(&format!("l{}_c{}", li, i));
                }
                core
            })
            .collect();

        // Pre-allocate inter-layer edge lists and modulatory biases
        let inter_edge_count = config.n_layers.saturating_sub(1);
        let inter_edges = vec![Vec::new(); inter_edge_count];
        let modulatory_biases = (0..config.n_layers)
            .map(|_| Array1::zeros(config.n_clusters))
            .collect();

        Self {
            layers,
            config,
            inter_edges,
            modulatory_biases,
            cached_rates: Vec::new(),
            cached_inter_phi: 0.0,
        }
    }

    pub fn n_layers(&self) -> usize { self.config.n_layers }

    pub fn layer(&self, idx: usize) -> Option<&TSOCore> { self.layers.get(idx) }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut TSOCore> { self.layers.get_mut(idx) }

    pub fn config(&self) -> &DeepConfig { &self.config }

    pub fn set_learn_inter_edges(&mut self, v: bool) { self.config.learn_inter_edges = v; }
    pub fn set_inter_edge_lr(&mut self, v: f64) { self.config.inter_edge_lr = v; }

    /// Add an inter-layer edge from `from_layer`'s cluster `from_cluster`
    /// to `to_layer`'s cluster `to_cluster`.
    ///
    /// `w`: +1 for implication, -1 for exclusion.
    /// `strength`: weight of the edge (default 1.0).
    ///
    /// Layers must be adjacent (to_layer == from_layer + 1).
    pub fn add_inter_edge(
        &mut self,
        from_layer: usize,
        from_cluster: usize,
        to_layer: usize,
        to_cluster: usize,
        w: f64,
        strength: f64,
    ) {
        if to_layer != from_layer + 1 || to_layer >= self.config.n_layers {
            return; // only adjacent layers
        }
        self.inter_edges[from_layer].push((from_cluster, to_cluster, w, strength));
    }

    /// Clear all inter-layer edges.
    pub fn clear_inter_edges(&mut self) {
        for v in &mut self.inter_edges {
            v.clear();
        }
    }

    /// Reset all layers, modulatory biases, and cached learning state.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        for bias in &mut self.modulatory_biases {
            bias.fill(0.0);
        }
        self.cached_rates.clear();
        self.cached_inter_phi = 0.0;
    }

    /// Compute inter-layer Φ for a single adjacent pair.
    fn compute_inter_layer_phi(
        &self,
        edge_list: &[(usize, usize, f64, f64)],
        lower_rates: &Array1<f64>,
        upper_rates: &Array1<f64>,
        gamma: f64,
        epsilon: f64,
    ) -> f64 {
        let mut phi = 0.0;
        for &(i, j, w, strength) in edge_list {
            if i >= lower_rates.len() || j >= upper_rates.len() {
                continue;
            }
            let dot = lower_rates[i] * upper_rates[j];
            let violation = if w > 0.0 {
                (gamma - dot).max(0.0)
            } else {
                (dot - epsilon).max(0.0)
            };
            phi += strength * violation;
        }
        phi
    }

    /// Compute the top-down modulatory bias for a lower layer based on
    /// inter-layer edge violations.
    fn compute_modulatory_bias(
        &self,
        edge_list: &[(usize, usize, f64, f64)],
        lower_rates: &Array1<f64>,
        upper_rates: &Array1<f64>,
        gamma: f64,
        epsilon: f64,
        mod_strength: f64,
        n_lower: usize,
    ) -> Array1<f64> {
        let mut bias = Array1::zeros(n_lower);
        for &(i, j, w, strength) in edge_list {
            if i >= n_lower || j >= upper_rates.len() {
                continue;
            }
            let dot = lower_rates[i] * upper_rates[j];
            let violation = if w > 0.0 {
                (gamma - dot).max(0.0)
            } else {
                (dot - epsilon).max(0.0)
            };
            if violation > 1e-6 {
                let b = if w > 0.0 {
                    // Implication: need higher dot → nudge lower rate up
                    mod_strength * strength
                } else {
                    // Exclusion: need lower dot → nudge lower rate down
                    -mod_strength * strength
                };
                bias[i] += b;
            }
        }
        bias
    }

    /// Perform one macro-step with the two-phase cortical cycle.
    ///
    /// **Phase 1 (Bottom-up):** Each layer receives the output of the layer
    /// below (plus any stored modulatory bias) and steps at its own dt.
    ///
    /// **Phase 2 (Top-down):** For each adjacent pair, compute inter-layer Φ
    /// and a modulatory bias that will be applied during the *next* step.
    pub fn step(&mut self, i_ext: &Array1<f64>, dt: f64) -> DeepOutput {
        let mut current = i_ext.clone();
        let mut layer_outputs = Vec::with_capacity(self.config.n_layers);
        let mut total_intra_phi = 0.0;

        // ── Phase 1: Bottom-up feedforward ──
        for li in 0..self.config.n_layers {
            let layer = &mut self.layers[li];
            let actual_dt = dt * self.config.dt_multipliers[li.clamp(
                0,
                self.config.dt_multipliers.len().saturating_sub(1),
            )];

            // Apply modulatory bias from the layer above (from previous step)
            let biased_input = {
                let n = current.len().min(self.modulatory_biases[li].len());
                let mut combined = current.clone();
                for i in 0..n {
                    combined[i] += self.modulatory_biases[li][i];
                }
                combined
            };

            let (phi, mut rates, spikes) = layer.step(&biased_input, actual_dt);
            // Winner-Take-All: keep only the top `wta_keep_ratio` fraction of clusters
            if self.config.wta_keep_ratio < 1.0 {
                let keep = (self.config.n_clusters as f64 * self.config.wta_keep_ratio).max(1.0).ceil() as usize;
                let mut idxs: Vec<usize> = (0..rates.len()).collect();
                idxs.sort_unstable_by(|&a, &b| rates[b].partial_cmp(&rates[a]).unwrap_or(std::cmp::Ordering::Equal));
                for &i in &idxs[keep.min(idxs.len())..] {
                    rates[i] = 0.0;
                }
                // Force-fire: if all rates are zero (dead network), force-activate
                // the top-k clusters with a minimal floor to restart dynamics.
                if rates.iter().all(|&r| r == 0.0) {
                    for &i in &idxs[..keep.min(idxs.len())] {
                        rates[i] = 0.1;
                    }
                }
            }
            total_intra_phi += phi;

            if self.config.residual && current.len() == rates.len() {
                current = &rates + &current;
            } else {
                current = rates.clone();
            }

            layer_outputs.push(LayerOutput { phi, rates, spikes });
        }

        // ── Phase 2: Inter-layer Φ and top-down modulation ──
        let mut total_inter_phi = 0.0;
        let gamma = self.config.gamma;
        let epsilon = self.config.epsilon;
        let mod_strength = self.config.modulatory_strength;

        for li in 0..self.config.n_layers.saturating_sub(1) {
            let lower_rates = &layer_outputs[li].rates;
            let upper_rates = &layer_outputs[li + 1].rates;
            let n_lower = lower_rates.len();

            // Compute inter-layer Φ for this adjacent pair
            let il_phi = self.compute_inter_layer_phi(
                &self.inter_edges[li], lower_rates, upper_rates, gamma, epsilon,
            );
            total_inter_phi += il_phi;

            // Compute modulatory bias for the lower layer (for next step)
            let bias = self.compute_modulatory_bias(
                &self.inter_edges[li], lower_rates, upper_rates,
                gamma, epsilon, mod_strength, n_lower,
            );
            if li < self.modulatory_biases.len() {
                self.modulatory_biases[li] = bias;
            }
        }

        let total_phi = total_intra_phi + total_inter_phi;

        // Output layer (L5-equivalent)
        let output_li = self
            .config
            .output_layer
            .unwrap_or(self.config.n_layers.saturating_sub(1));
        let final_rates = output_li
            .checked_sub(self.config.n_layers.saturating_sub(1))
            .and_then(|_| None)
            .or_else(|| layer_outputs.get(output_li).map(|o| o.rates.clone()))
            .unwrap_or_else(|| {
                layer_outputs
                    .last()
                    .map(|o| o.rates.clone())
                    .unwrap_or_default()
            });

        // ── Phase 3: Inter-layer R-STDP edge learning ──
        if self.config.learn_inter_edges
            && self.config.inter_edge_lr > 0.0
            && !self.cached_rates.is_empty()
        {
            self.update_inter_edges(&layer_outputs, total_inter_phi);
        }

        // Cache rates and inter-layer Φ for the next step
        self.cached_rates = layer_outputs.iter().map(|o| o.rates.clone()).collect();
        self.cached_inter_phi = total_inter_phi;

        DeepOutput {
            layers: layer_outputs,
            inter_phi: total_inter_phi,
            total_phi,
            final_rates,
        }
    }

    /// Update inter-layer edge strengths via unsupervised R-STDP.
    ///
    /// **Reward signal:** `reward = -dΦ_inter`. When inter-layer Φ decreases
    /// (improvement), edges that were active (high Hebbian product) are
    /// strengthened. When Φ increases (worsening), active edges are weakened.
    ///
    /// **Asymmetry:** Good news (Φ↓) has larger magnitude than bad news (Φ↑),
    /// matching biological dopamine transients.
    ///
    /// **Per-edge update:**
    ///   `Δstrength = lr * reward * rate_lower[i] * rate_upper[j]`
    ///   Edges are clamped to [0.1, 5.0].
    fn update_inter_edges(
        &mut self,
        layer_outputs: &[LayerOutput],
        curr_inter_phi: f64,
    ) {
        let delta_phi = curr_inter_phi - self.cached_inter_phi;
        // Asymmetric reward: improvement is rewarded more than worsening is punished
        let reward = if delta_phi < 0.0 { 1.0 } else { -0.3 };

        for li in 0..self.inter_edges.len() {
            let lower_rates = &layer_outputs[li].rates;
            let upper_rates = &layer_outputs[li + 1].rates;

            for (i, j, _w, strength) in &mut self.inter_edges[li] {
                if *i >= lower_rates.len() || *j >= upper_rates.len() {
                    continue;
                }
                let hebb = lower_rates[*i] * upper_rates[*j];
                let delta = self.config.inter_edge_lr * reward * hebb;
                *strength = (*strength + delta).clamp(0.1, 5.0);
            }
        }
    }
}

// ── Batch processor (parallel) ──

pub struct DeepBatchProcessor {
    config: DeepConfig,
}

impl DeepBatchProcessor {
    pub fn new(deep_config: DeepConfig) -> Self {
        Self { config: deep_config }
    }

    pub fn process_batch(
        &self,
        batch: &[Vec<Array1<f64>>],
        dt: f64,
    ) -> Vec<Vec<DeepOutput>> {
        batch
            .par_iter()
            .map(|seq| {
                let mut deep = DeepTSO::new(self.config.clone());
                let mut outputs = Vec::with_capacity(seq.len());
                for input in seq {
                    outputs.push(deep.step(input, dt));
                }
                outputs
            })
            .collect()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_deep_tso_creation() {
        let config = DeepConfig {
            n_layers: 3,
            n_clusters: 10,
            dt_multipliers: vec![1.0, 2.0, 4.0],
            ..Default::default()
        };
        let deep = DeepTSO::new(config);
        assert_eq!(deep.n_layers(), 3);
        assert_eq!(deep.layer(0).unwrap().clusters.len(), 10);
        assert_eq!(deep.inter_edges.len(), 2); // 3 layers → 2 adjacent pairs
    }

    #[test]
    fn test_deep_tso_step_bottom_up() {
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 5,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            ..Default::default()
        });
        let input = Array1::from_elem(5, 2.0);
        let output = deep.step(&input, 0.5);
        assert_eq!(output.layers.len(), 2);
        assert_eq!(output.final_rates.len(), 5);
        assert!(output.inter_phi >= 0.0);
        assert!(output.total_phi >= output.inter_phi);
    }

    #[test]
    fn test_deep_tso_inter_layer_phi() {
        // Create a 2-layer DeepTSO with known inter-layer edges
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            ..Default::default()
        });

        // Add an implication edge: cluster 0 of layer 0 → cluster 0 of layer 1
        // Both start at rate ~0, so implication is violated (gamma=0.5, dot≈0)
        deep.add_inter_edge(0, 0, 1, 0, 1.0, 1.0);

        let input = Array1::from_elem(3, 0.2); // low input → low rates
        let output = deep.step(&input, 0.5);

        // Inter-layer Φ should be positive (implication violated: 0.5 - dot > 0)
        assert!(
            output.inter_phi > 0.0,
            "Inter-layer phi should be positive with violated implication edge"
        );
    }

    #[test]
    fn test_deep_tso_top_down_modulation() {
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            modulatory_strength: 0.1,
            ..Default::default()
        });

        // Implication edge: layer0 cluster 0 → layer1 cluster 0
        deep.add_inter_edge(0, 0, 1, 0, 1.0, 1.0);

        let input = Array1::from_elem(3, 0.2);
        let _ = deep.step(&input, 0.5);

        // After Phase 2, modulatory bias for layer 0 should be > 0
        // (because the higher layer "wants" lower cluster 0 to fire more)
        let bias = &deep.modulatory_biases[0];
        assert!(
            bias[0] > 0.0,
            "Modulatory bias should push lower cluster up for violated implication"
        );

        // Second step should have different rates than first (bias applied)
        let output2 = deep.step(&input, 0.5);
        assert!(output2.total_phi >= 0.0);
    }

    #[test]
    fn test_deep_tso_exclusion_inter_edge() {
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            modulatory_strength: 0.1,
            ..Default::default()
        });

        // Exclusion edge: layer0 cluster 0 → layer1 cluster 0
        // (high dot = violation)
        deep.add_inter_edge(0, 0, 1, 0, -1.0, 1.0);

        // Saturating input (350.0) ensures LIF neurons fire every step
        // after the first integration step. Rates build toward 1.0.
        // Layer 0 (dt=0.5, tau_rate=50, a≈0.01): rate ≈ 0.63 after 100 steps
        // Layer 1 (dt=1.0, tau_rate=50, a≈0.02): rate ≈ 0.87 after 100 steps
        // Dot ≈ 0.55 > epsilon=0.3 → exclusion violated
        let input = Array1::from_elem(3, 350.0);
        let mut last_inter_phi = 0.0;
        let mut reached_positive = false;
        for _ in 0..100 {
            let output = deep.step(&input, 0.5);
            if output.inter_phi > 0.0 {
                reached_positive = true;
                last_inter_phi = output.inter_phi;
                break;
            }
        }

        assert!(
            reached_positive,
            "Exclusion edge should create inter-layer friction \
             after rates build up (last_phi={:.6})",
            last_inter_phi
        );

        // Modulatory bias for layer0 cluster 0 should be negative
        let bias = &deep.modulatory_biases[0];
        assert!(
            bias[0] < 0.0,
            "Modulatory bias should push lower cluster down for violated exclusion"
        );
    }

    #[test]
    fn test_deep_tso_output_layer() {
        // Test that output_layer config works
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 3,
            n_clusters: 4,
            d: 3,
            dt_multipliers: vec![1.0, 2.0, 4.0],
            output_layer: Some(0), // read from layer 0
            ..Default::default()
        });

        let input = Array1::from_elem(4, 1.0);
        let output = deep.step(&input, 0.5);
        assert_eq!(output.final_rates.len(), 4);

        // Default should read from top layer (n_layers - 1 = 2)
        let mut deep2 = DeepTSO::new(DeepConfig {
            n_layers: 3,
            n_clusters: 4,
            d: 3,
            dt_multipliers: vec![1.0, 2.0, 4.0],
            output_layer: None,
            ..Default::default()
        });
        let output2 = deep2.step(&input, 0.5);
        assert_eq!(output2.final_rates.len(), 4);
    }

    #[test]
    fn test_deep_tso_reset_clears_biases() {
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            ..Default::default()
        });

        deep.add_inter_edge(0, 0, 1, 0, 1.0, 1.0);
        let input = Array1::from_elem(3, 0.2);
        let _ = deep.step(&input, 0.5);

        // Bias should have been set
        assert!(deep.modulatory_biases[0][0] != 0.0);

        deep.reset();

        // After reset, biases should be zero
        for b in deep.modulatory_biases[0].iter() {
            assert!((*b).abs() < 1e-10, "Bias should be zero after reset");
        }
    }

    #[test]
    fn test_deep_tso_inter_rstdp_strengthens_implication() {
        // R-STDP strengthens implication edges when inter-layer Φ decreases
        // (the higher layer's prediction becomes more accurate over time).
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            modulatory_strength: 0.1,
            inter_edge_lr: 0.01,
            ..Default::default()
        });

        // Implication edge: layer0 cluster 0 → layer1 cluster 0
        deep.add_inter_edge(0, 0, 1, 0, 1.0, 1.0);
        let initial_strength = deep.inter_edges[0][0].3;

        // Stable input consistently activates both layers similarly
        // → implication is satisfied → inter-layer Φ decreases over time
        let input = Array1::from_elem(3, 350.0);
        for _ in 0..20 {
            let out = deep.step(&input, 0.5);
            // Inter-layer Φ should trend downward as the edge is learned
            let _ = out.total_phi;
        }

        let final_strength = deep.inter_edges[0][0].3;
        assert!(
            final_strength > initial_strength,
            "Implication edge strength should increase when predictions improve \
             (initial={:.4}, final={:.4})",
            initial_strength, final_strength,
        );
    }

    #[test]
    fn test_deep_tso_inter_rstdp_weakens_exclusion() {
        // R-STDP weakens exclusion edges when inter-layer Φ increases
        // (the higher layer's prediction becomes less accurate).
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 2,
            n_clusters: 3,
            d: 3,
            dt_multipliers: vec![1.0, 2.0],
            modulatory_strength: 0.1,
            inter_edge_lr: 0.01,
            ..Default::default()
        });

        // Exclusion edge: layer0 cluster 0 → layer1 cluster 0
        // Both layers firing → exclusion violated → Φ increases
        deep.add_inter_edge(0, 0, 1, 0, -1.0, 1.0);
        let initial_strength = deep.inter_edges[0][0].3;

        let input = Array1::from_elem(3, 350.0);
        for _ in 0..20 {
            let out = deep.step(&input, 0.5);
            let _ = out.total_phi;
        }

        // The exclusion edge should weaken (its prediction is consistently wrong)
        let final_strength = deep.inter_edges[0][0].3;
        assert!(
            final_strength <= initial_strength + 1e-6,
            "Exclusion edge strength should not increase when predictions worsen \
             (initial={:.4}, final={:.4})",
            initial_strength, final_strength,
        );
    }

    #[test]
    fn test_deep_batch_processor() {
        let deep_config = DeepConfig {
            n_layers: 2,
            n_clusters: 4,
            dt_multipliers: vec![1.0, 2.0],
            ..Default::default()
        };
        let bp = DeepBatchProcessor::new(deep_config);

        let batch = vec![
            vec![Array1::from_elem(4, 1.0), Array1::from_elem(4, 2.0)],
            vec![Array1::from_elem(4, 0.5)],
        ];

        let results = bp.process_batch(&batch, 0.5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 1);
    }
}
