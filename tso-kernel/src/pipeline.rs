use crate::core::TSOCore;
use ndarray::Array1;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub phi: f64,
    pub rates: Array1<f64>,
    pub total_spikes: u32,
}

#[derive(Clone, Debug)]
pub struct SequenceOutput {
    pub steps: Vec<StepOutput>,
    pub final_phi: f64,
    pub mean_phi: f64,
    pub total_spikes: u32,
}

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub max_clusters: usize,
    pub d: usize,
    pub gamma: f64,
    pub epsilon: f64,
    pub history_size: usize,
    pub base_theta_c: f64,
    pub inertia_threshold: f64,
    pub batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_clusters: 100,
            d: 5,
            gamma: 0.5,
            epsilon: 0.3,
            history_size: 50,
            base_theta_c: 0.5,
            inertia_threshold: 0.1,
            batch_size: 32,
        }
    }
}

pub struct BatchProcessor {
    config: BatchConfig,
}

impl BatchProcessor {
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    fn make_core(&self) -> TSOCore {
        TSOCore::new(
            self.config.max_clusters,
            self.config.d,
            self.config.gamma,
            self.config.epsilon,
            self.config.history_size,
            self.config.base_theta_c,
            self.config.inertia_threshold,
        )
    }

    fn process_single(&self, core: &mut TSOCore, seq: &[Array1<f64>], dt: f64) -> SequenceOutput {
        let mut steps = Vec::with_capacity(seq.len());
        let mut total_spikes = 0;

        for input in seq {
            let (phi, rates, spikes) = core.step(input, dt);
            total_spikes += spikes;
            steps.push(StepOutput { phi, rates, total_spikes: spikes });
        }

        let final_phi = steps.last().map(|s| s.phi).unwrap_or(0.0);
        let mean_phi = if !steps.is_empty() {
            steps.iter().map(|s| s.phi).sum::<f64>() / steps.len() as f64
        } else {
            0.0
        };

        SequenceOutput { steps, final_phi, mean_phi, total_spikes }
    }

    pub fn process_batch(
        &self,
        batch: &[Vec<Array1<f64>>],
        dt: f64,
    ) -> Vec<SequenceOutput> {
        batch
            .par_iter()
            .map(|seq| {
                let mut core = self.make_core();
                self.process_single(&mut core, seq, dt)
            })
            .collect()
    }

    pub fn process_with_cores(
        &self,
        cores: &mut [TSOCore],
        batch: &[Vec<Array1<f64>>],
        dt: f64,
    ) -> Vec<SequenceOutput> {
        assert_eq!(cores.len(), batch.len(),
            "BatchProcessor: cores and batch must have same length ({} vs {})",
            cores.len(), batch.len());

        cores
            .par_iter_mut()
            .zip(batch.par_iter())
            .map(|(core, seq)| self.process_single(core, seq, dt))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_input(n_clusters: usize, val: f64) -> Array1<f64> {
        Array1::from_elem(n_clusters, val)
    }

    #[test]
    fn test_process_single_sequence() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config);

        let n_clusters = 5;
        let seq = vec![
            make_input(n_clusters, 1.0),
            make_input(n_clusters, 2.0),
            make_input(n_clusters, 3.0),
        ];

        let result = processor.process_batch(&[seq], 0.5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].steps.len(), 3);
    }

    #[test]
    fn test_process_batch_parallel() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config);

        let n_clusters = 5;
        let batch: Vec<Vec<Array1<f64>>> = (0..10)
            .map(|i| {
                vec![
                    make_input(n_clusters, i as f64 * 0.5),
                    make_input(n_clusters, i as f64 * 0.3),
                ]
            })
            .collect();

        let results = processor.process_batch(&batch, 0.5);
        assert_eq!(results.len(), 10);
        for r in &results {
            assert_eq!(r.steps.len(), 2);
        }
    }

    #[test]
    fn test_process_with_cores_reuse() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config.clone());

        let n_clusters = 5;
        let batch: Vec<Vec<Array1<f64>>> = (0..4)
            .map(|i| {
                vec![
                    make_input(n_clusters, i as f64),
                ]
            })
            .collect();

        let mut cores: Vec<TSOCore> = (0..4)
            .map(|_| {
                let mut c = TSOCore::new(
                    config.max_clusters, config.d, config.gamma, config.epsilon,
                    config.history_size, config.base_theta_c, config.inertia_threshold,
                );
                for j in 0..5 {
                    c.add_cluster(&format!("c{}", j));
                }
                c
            })
            .collect();

        let results = processor.process_with_cores(&mut cores, &batch, 0.5);
        assert_eq!(results.len(), 4);
    }
}
