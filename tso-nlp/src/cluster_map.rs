use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub struct ClusterMap {
    n_clusters: usize,
    token_to_cluster: HashMap<u32, usize>,
    node_to_cluster: HashMap<NodeIndex, usize>,
}

impl ClusterMap {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            token_to_cluster: HashMap::new(),
            node_to_cluster: HashMap::new(),
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn modular(token_id: u32, n_clusters: usize) -> usize {
        (token_id as usize) % n_clusters
    }

    pub fn register(&mut self, token_id: u32) -> usize {
        let cid = Self::modular(token_id, self.n_clusters);
        self.token_to_cluster.insert(token_id, cid);
        cid
    }

    pub fn register_node(&mut self, node: NodeIndex) -> usize {
        let cid = node.index() % self.n_clusters;
        self.node_to_cluster.insert(node, cid);
        cid
    }

    pub fn token_to_cluster(&self, token_id: u32) -> Option<usize> {
        self.token_to_cluster.get(&token_id).copied()
    }

    pub fn node_to_cluster(&self, node: NodeIndex) -> Option<usize> {
        self.node_to_cluster.get(&node).copied()
    }

    pub fn token_to_cluster_or_default(&self, token_id: u32) -> usize {
        self.token_to_cluster
            .get(&token_id)
            .copied()
            .unwrap_or_else(|| Self::modular(token_id, self.n_clusters))
    }

    pub fn node_to_cluster_or_default(&self, node: NodeIndex) -> usize {
        self.node_to_cluster
            .get(&node)
            .copied()
            .unwrap_or(node.index() % self.n_clusters)
    }

    pub fn token_ids_to_clusters(&self, token_ids: &[u32]) -> Vec<usize> {
        token_ids
            .iter()
            .map(|&tid| self.token_to_cluster_or_default(tid))
            .collect()
    }
}

/// Convert a sequence of cluster IDs to current vectors for DeepTSO input.
///
/// Each token becomes a vector of length `n_clusters` with 0.0 everywhere
/// except 1.0 at the active cluster index.
pub fn clusters_to_currents(cluster_ids: &[usize], n_clusters: usize) -> Vec<ndarray::Array1<f64>> {
    cluster_ids
        .iter()
        .map(|&cid| {
            let mut v = ndarray::Array1::zeros(n_clusters);
            if cid < n_clusters {
                v[cid] = 1.0;
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_mapping() {
        assert_eq!(ClusterMap::modular(0, 128), 0);
        assert_eq!(ClusterMap::modular(128, 128), 0);
        assert_eq!(ClusterMap::modular(129, 128), 1);
        assert_eq!(ClusterMap::modular(255, 128), 127);
    }

    #[test]
    fn test_register_and_lookup() {
        let mut cm = ClusterMap::new(64);
        let cid = cm.register(42);
        assert_eq!(cid, 42 % 64);
        assert_eq!(cm.token_to_cluster(42), Some(42 % 64));
    }

    #[test]
    fn test_token_ids_to_clusters() {
        let cm = ClusterMap::new(10);
        let ids = vec![0, 10, 5, 15];
        let clusters = cm.token_ids_to_clusters(&ids);
        assert_eq!(clusters, vec![0, 0, 5, 5]);
    }

    #[test]
    fn test_clusters_to_currents() {
        let currents = clusters_to_currents(&[0, 3, 7], 10);
        assert_eq!(currents.len(), 3);
        assert_eq!(currents[0][0], 1.0);
        assert_eq!(currents[0][1], 0.0);
        assert_eq!(currents[1][3], 1.0);
        assert_eq!(currents[2][7], 1.0);
    }
}
