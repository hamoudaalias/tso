use crate::core::TSOCore;
use ndarray::Array1;
use rayon::prelude::*;

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
    pub total_phi: f64,
    pub final_rates: Array1<f64>,
}

pub struct DeepTSO {
    layers: Vec<TSOCore>,
    config: DeepConfig,
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

        Self { layers, config }
    }

    pub fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    pub fn layer(&self, idx: usize) -> Option<&TSOCore> {
        self.layers.get(idx)
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut TSOCore> {
        self.layers.get_mut(idx)
    }

    pub fn config(&self) -> &DeepConfig {
        &self.config
    }

    pub fn step(&mut self, i_ext: &Array1<f64>, dt: f64) -> DeepOutput {
        let mut current = i_ext.clone();
        let mut layer_outputs = Vec::with_capacity(self.config.n_layers);
        let mut total_phi = 0.0;

        for layer in &mut self.layers {
            let (phi, rates, spikes) = layer.step(&current, dt);
            total_phi += phi;

            if self.config.residual && current.len() == rates.len() {
                current = &rates + &current;
            } else {
                current = rates.clone();
            }

            layer_outputs.push(LayerOutput { phi, rates, spikes });
        }

        DeepOutput { layers: layer_outputs, total_phi, final_rates: current }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_deep_tso_creation() {
        let config = DeepConfig {
            n_layers: 3,
            n_clusters: 10,
            ..Default::default()
        };
        let deep = DeepTSO::new(config);
        assert_eq!(deep.n_layers(), 3);
        assert_eq!(deep.layer(0).unwrap().clusters.len(), 10);
    }

    #[test]
    fn test_deep_tso_step() {
        let config = DeepConfig {
            n_layers: 2,
            n_clusters: 5,
            d: 3,
            ..Default::default()
        };
        let mut deep = DeepTSO::new(config);
        let input = Array1::from_elem(5, 2.0);
        let output = deep.step(&input, 0.5);
        assert_eq!(output.layers.len(), 2);
        assert_eq!(output.final_rates.len(), 5);
    }

    #[test]
    fn test_deep_tso_residual() {
        let mut deep = DeepTSO::new(DeepConfig {
            n_layers: 3,
            n_clusters: 4,
            d: 3,
            residual: true,
            ..Default::default()
        });
        let input = Array1::from_elem(4, 0.5);
        let out1 = deep.step(&input, 0.5);

        let mut deep2 = DeepTSO::new(DeepConfig {
            n_layers: 3,
            n_clusters: 4,
            d: 3,
            residual: false,
            ..Default::default()
        });
        let out2 = deep2.step(&input, 0.5);

        assert!(out1.final_rates != out2.final_rates);
    }

    #[test]
    fn test_deep_tso_reset() {
        let config = DeepConfig {
            n_layers: 2,
            n_clusters: 5,
            ..Default::default()
        };
        let mut deep = DeepTSO::new(config);
        let input = Array1::from_elem(5, 3.0);
        let _ = deep.step(&input, 0.5);
        let _ = deep.step(&input, 0.5);
        deep.reset();
        for l in 0..deep.n_layers() {
            assert!(deep.layer(l).unwrap().phi_history.is_empty());
        }
    }

    #[test]
    fn test_deep_batch_processor() {
        let deep_config = DeepConfig {
            n_layers: 2,
            n_clusters: 4,
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
