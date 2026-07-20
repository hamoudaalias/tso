use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

#[derive(Clone)]
pub struct TokenGraph {
    pub graph: DiGraph<String, f64>,
    pub token_to_node: HashMap<u32, NodeIndex>,
    pub node_to_token: HashMap<NodeIndex, u32>,
    pub window_size: usize,
}

impl TokenGraph {
    pub fn new(window_size: usize) -> Self {
        Self {
            graph: DiGraph::new(),
            token_to_node: HashMap::new(),
            node_to_token: HashMap::new(),
            window_size,
        }
    }

    fn get_or_add_node(&mut self, token_id: u32, word: &str) -> NodeIndex {
        if let Some(&n) = self.token_to_node.get(&token_id) {
            return n;
        }
        let n = self.graph.add_node(word.to_string());
        self.token_to_node.insert(token_id, n);
        self.node_to_token.insert(n, token_id);
        n
    }

    pub fn add_cooccurrence(&mut self, token_id: u32, word: &str, context: &[(u32, &str)]) {
        let node = self.get_or_add_node(token_id, word);
        for &(ctx_id, ctx_word) in context {
            if ctx_id == token_id {
                continue;
            }
            let ctx_node = self.get_or_add_node(ctx_id, ctx_word);
            if let Some(e) = self.graph.find_edge(node, ctx_node) {
                *self.graph.edge_weight_mut(e).unwrap() += 1.0;
            } else {
                self.graph.add_edge(node, ctx_node, 1.0);
            }
        }
    }

    pub fn process_sequence(&mut self, words: &[(u32, String)]) {
        for (i, &(tid, ref word)) in words.iter().enumerate() {
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(words.len());
            let context: Vec<(u32, &str)> = words[start..end]
                .iter()
                .map(|(id, w)| (*id, w.as_str()))
                .collect();
            self.add_cooccurrence(tid, word, &context);
        }
    }

    pub fn normalize_edges(&mut self) {
        let edges: Vec<EdgeIndex> = self.graph.edge_indices().collect();
        let mut out_weight: HashMap<NodeIndex, f64> = HashMap::new();
        for &e in &edges {
            let (src, _) = self.graph.edge_endpoints(e).unwrap();
            let w = *self.graph.edge_weight(e).unwrap();
            *out_weight.entry(src).or_insert(0.0) += w;
        }
        for &e in &edges {
            let (src, _) = self.graph.edge_endpoints(e).unwrap();
            if let Some(&total) = out_weight.get(&src) {
                if total > 0.0 {
                    *self.graph.edge_weight_mut(e).unwrap() /= total;
                }
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn to_friction_graph(&self) -> std::collections::HashMap<String, std::collections::HashMap<String, f64>> {
        let mut hg: std::collections::HashMap<String, std::collections::HashMap<String, f64>> =
            std::collections::HashMap::new();
        for node in self.graph.node_indices() {
            let name = self.graph[node].clone();
            let mut neighbors = std::collections::HashMap::new();
            for edge in self.graph.edges_directed(node, petgraph::Direction::Outgoing) {
                let target = self.graph[edge.target()].clone();
                neighbors.insert(target, *edge.weight());
            }
            hg.insert(name, neighbors);
        }
        hg
    }

    pub fn token_ids(&self) -> Vec<u32> {
        self.node_to_token.values().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_empty() {
        let g = TokenGraph::new(2);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_process_sequence() {
        let mut g = TokenGraph::new(1);
        let seq: Vec<(u32, String)> = vec![
            (0, "the".into()),
            (1, "cat".into()),
            (2, "sat".into()),
            (3, "down".into()),
        ];
        g.process_sequence(&seq);
        assert_eq!(g.node_count(), 4);
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_cooccurrence_creates_nodes() {
        let mut g = TokenGraph::new(1);
        g.add_cooccurrence(42, "hello", &[(7, "world"), (8, "test")]);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_normalize_edges() {
        let mut g = TokenGraph::new(1);
        let seq: Vec<(u32, String)> = vec![
            (0, "a".into()),
            (1, "b".into()),
            (0, "a".into()),
            (2, "c".into()),
            (0, "a".into()),
            (3, "d".into()),
        ];
        g.process_sequence(&seq);
        g.normalize_edges();
        for e in g.graph.edge_indices() {
            let w = *g.graph.edge_weight(e).unwrap();
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_to_friction_graph_uses_words() {
        let mut g = TokenGraph::new(1);
        let seq: Vec<(u32, String)> = vec![
            (0, "hello".into()),
            (1, "world".into()),
        ];
        g.process_sequence(&seq);
        let fg = g.to_friction_graph();
        assert!(fg.contains_key("hello"), "should contain 'hello' not 't0'");
        assert!(fg.contains_key("world"), "should contain 'world' not 't1'");
    }
}
