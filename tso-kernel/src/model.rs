use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub window_size: usize,
    pub top_k: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self { window_size: 5, top_k: 20 }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainedNLIClassifier {
    pub config: ModelConfig,
    pub sorted_neighbors: HashMap<String, Vec<String>>,
    pub centroids: HashMap<usize, Vec<f64>>,
}

impl TrainedNLIClassifier {
    pub fn new(
        config: ModelConfig,
        sorted_neighbors: HashMap<String, Vec<String>>,
        centroids: HashMap<usize, Vec<f64>>,
    ) -> Self {
        Self { config, sorted_neighbors, centroids }
    }

    pub fn sorted_neighbors_as_hashsets(&self) -> HashMap<String, HashSet<String>> {
        self.sorted_neighbors
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    }

    pub fn save_bin(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path.as_ref(), encoded)?;
        Ok(())
    }

    pub fn load_bin(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path.as_ref())?;
        let model: Self = bincode::deserialize(&bytes)?;
        Ok(model)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), json)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let model: Self = serde_json::from_str(&text)?;
        Ok(model)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FrictionGraphCheckpoint {
    pub config: ModelConfig,
    pub graph: HashMap<String, HashMap<String, f64>>,
}

impl FrictionGraphCheckpoint {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path.as_ref(), encoded)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path.as_ref())?;
        let ckpt: Self = bincode::deserialize(&bytes)?;
        Ok(ckpt)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CentroidsCheckpoint {
    pub centroids: HashMap<usize, Vec<f64>>,
    pub counts: HashMap<usize, usize>,
    pub total_samples: usize,
}

impl CentroidsCheckpoint {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), json)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path.as_ref())?;
        let ckpt: Self = serde_json::from_str(&text)?;
        Ok(ckpt)
    }
}
