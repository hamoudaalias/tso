pub mod attractor;
pub mod core;
pub mod decoder;
pub mod deep;
pub mod friction;
pub mod model;
pub mod neurons;
pub mod operators;
pub mod pipeline;
pub mod plasticity;

pub use attractor::SharpAttractorField;
pub use core::TSOCore;
pub use deep::{DeepConfig, DeepBatchProcessor, DeepOutput, DeepTSO, LayerOutput};
pub use friction::{
    build_typed_graph, compute_trifriction, compute_trifriction_fast, compute_typed_trifriction,
    compute_weighted_trifriction, predict_from_phi, prepare_sorted_neighbors, rstdp_update_edges,
    FrictionCalculator, TypedEdge,
};
pub use neurons::{LIFCluster, LIFNeuron};
pub use operators::TopographicOperator;
pub use pipeline::{BatchConfig, BatchProcessor, SequenceOutput, StepOutput};
pub use plasticity::{EligibilityTrace, RSTDPPlasticity};
