# TSO — Temporal-Semantic Orchestration Engine

A biologically-inspired cognitive architecture integrating perception, categorization, episodic memory, semantic reasoning, and Hebbian reinforcement learning — no backpropagation, no GPU, no experience replay.

## Core Results

| Benchmark | Train | Test (greedy) | Path Length | Condition |
|-----------|-------|---------------|-------------|-----------|
| T-Maze (shaped reward) | 99.4% | 100% | 2–4 steps | One-hot position encoding |
| T-Maze (pure delayed) | 88.4% | 88.4% | 2–4 steps | Terminal reward only |
| T-Maze (reversal) | 87.4% | 87.4% | 2–4 steps | Mid-training rule flip |
| T-Maze (POMDP delay) | 84.4% | 84.4% | 2–12 steps | Cue visible 1 step only |
| Empty Room 5×5 (whiskers) | 100% | 100% | 4 steps | 4 local ray sensors |
| L-Maze 7×7 (whiskers) | 90.2% | 99.0% | ~12 steps | Perceptual aliasing |
| **Zigzag 10×10 (whiskers)** | **66.4%** | **70.0%** | **28 steps** | Long-horizon + aliasing |

## Architecture

```
tso-engine/
├── attractor.rs      # LVQ1 attractor field (Euclidean distance)
├── cerebellum.rs     # Reward-modulated Hebbian learning (REINFORCE + eligibility traces)
├── core.rs           # Graph, Φ energy, resolve (constraint satisfaction)
├── episodic.rs       # Episodic memory (suffix-prefix matching)
├── grid_world.rs     # 2D navigation environments
├── main.rs           # Benchmark runner
├── neurons.rs        # LIF, Dual-LIF temporal integrators
├── tso_engine.rs     # 7-step cognitive cycle orchestration
├── working_memory.rs # DualLIF + associative memory
└── action.rs         # ActionMotor
```

## Key Mechanisms

- **Online categorization** via Euclidean-distance LVQ1 (no training/test split)
- **Cognitive map** through concept value iteration (trans_log + propagate_values)
- **BFS gradient bias** in action logits (innate navigation prior)
- **Raw perception** as cerebellum's decision state (no LIF smoothing)
- **Eligibility traces** (γλ=0.9702) for 28-step credit assignment
- **ε-greedy exploration** with exponential decay (0.50 → 0.05)

## Quick Start

```bash
cargo run --release -p tso-engine
```

## Paper

See [`TSO_PAPER.md`](TSO_PAPER.md) for the full technical paper.
