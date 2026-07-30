pub mod core;
pub mod neurons;
pub mod attractor;
pub mod episodic;
pub mod memory;
pub mod perceptual_belt;
pub mod working_memory;
pub mod action;
pub mod replay_buffer;
pub mod cerebellum;
pub mod hypothalamus;
pub mod attention;
#[cfg(feature = "rstdp")]
pub mod plasticity;
pub mod tso_engine;
pub use tso_engine::CognitiveConfig;
pub use tso_engine::TsoEngine;
pub use tso_engine::SleepReport;
pub use neurogenesis::{Neurogenesis, NeurogenesisConfig, NeurogenesisOutcome};
pub mod grid_world;
pub mod grid_cells;

pub mod sokoban;
pub mod terrarium;
pub mod encoder;
pub mod environment;
// pub mod vae; — removed: d=0.02 vs attractor seul (paper.md §6.1)

pub mod rotating_t;
pub mod minigrid_env;
pub mod zigzag_grid;
#[cfg(feature = "active-inference")]
pub mod efe;
#[cfg(feature = "active-inference")]
pub mod fpi;
pub mod model;
#[cfg(feature = "active-inference")]
pub mod inference;
pub mod learning;
pub mod neurogenesis;

pub mod baselines;
