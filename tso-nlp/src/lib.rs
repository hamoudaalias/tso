pub mod bpe;
pub mod cluster_map;
pub mod dataset;
pub mod distributional;
pub mod graph_builder;
pub mod metrics;
pub mod som;
pub mod syntagmatic;
pub mod tokenizer;

use cluster_map::ClusterMap;
use graph_builder::TokenGraph;
use ndarray::Array1;
use tso_kernel::deep::{DeepConfig, DeepOutput, DeepTSO};
use tokenizer::TSOTokenizer;

pub struct TSONLP {
    pub tokenizer: TSOTokenizer,
    pub graph: TokenGraph,
    pub cluster_map: ClusterMap,
    pub n_clusters: usize,
}

impl TSONLP {
    pub fn new(
        tokenizer: TSOTokenizer,
        window_size: usize,
        n_clusters: usize,
    ) -> Self {
        Self {
            tokenizer,
            graph: TokenGraph::new(window_size),
            cluster_map: ClusterMap::new(n_clusters),
            n_clusters,
        }
    }

    pub fn learn_from_corpus(&mut self, texts: &[&str]) {
        for &text in texts {
            let ids = self.tokenizer.encode_with_words(text, false);
            self.graph.process_sequence(&ids);
        }
        self.graph.normalize_edges();
        for (&tid, _) in &self.graph.token_to_node {
            self.cluster_map.register(tid);
        }
    }

    pub fn text_to_clusters(&mut self, text: &str) -> Vec<usize> {
        let ids = self.tokenizer.encode(text, false);
        self.cluster_map.token_ids_to_clusters(&ids)
    }

    pub fn text_to_currents(&mut self, text: &str) -> Vec<Array1<f64>> {
        let clusters = self.text_to_clusters(text);
        cluster_map::clusters_to_currents(&clusters, self.n_clusters)
    }

    pub fn process_phrase(
        &mut self,
        deep: &mut DeepTSO,
        text: &str,
        dt: f64,
    ) -> Vec<DeepOutput> {
        let currents = self.text_to_currents(text);
        let mut outputs = Vec::with_capacity(currents.len());
        for input in &currents {
            outputs.push(deep.step(input, dt));
        }
        outputs
    }

    pub fn make_deep_tso(&self, config: DeepConfig) -> DeepTSO {
        assert_eq!(
            config.n_clusters, self.n_clusters,
            "DeepConfig.n_clusters ({}) must match TSONLP.n_clusters ({})",
            config.n_clusters, self.n_clusters
        );
        DeepTSO::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tso_kernel::deep::DeepConfig;

    #[test]
    fn test_end_to_end() {
        let mut nlp = TSONLP::new(TSOTokenizer::whitespace(), 2, 10);
        nlp.learn_from_corpus(&["the cat sat on the mat", "a dog is an animal"]);

        let currents = nlp.text_to_currents("cat sat");
        assert_eq!(currents.len(), 2);
        assert_eq!(currents[0].len(), 10);
        assert!(currents[0].iter().any(|&v| v == 1.0));
    }

    #[test]
    fn test_empty_text() {
        let mut nlp = TSONLP::new(TSOTokenizer::whitespace(), 2, 10);
        let currents = nlp.text_to_currents("");
        assert!(currents.is_empty());
    }

    #[test]
    fn test_full_pipeline_text_to_deep_tso() {
        let mut nlp = TSONLP::new(TSOTokenizer::whitespace(), 2, 8);
        nlp.learn_from_corpus(&[
            "the cat sat on the mat",
            "a dog is an animal",
            "the cat is not a dog",
        ]);

        let mut deep = nlp.make_deep_tso(DeepConfig {
            n_layers: 3,
            n_clusters: 8,
            d: 4,
            residual: true,
            ..DeepConfig::default()
        });

        let outputs = nlp.process_phrase(&mut deep, "cat sat on mat", 0.5);
        assert_eq!(outputs.len(), 4);
        for out in &outputs {
            assert_eq!(out.layers.len(), 3);
            assert_eq!(out.final_rates.len(), 8);
        }
        assert!(outputs.last().unwrap().total_phi > 0.0 || outputs.last().unwrap().layers[0].phi == 0.0);
    }
}
