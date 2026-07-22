# TSO — Topographic Stabilization Operator

A neuromorphic cognitive engine where topographic friction (Φ) replaces attention — no backpropagation, no Transformers, no GPU.

## Architecture

```
tso-engine/        # Cognitive engine (modality-independent)
├── core           # Graph, Φ, Critic, Actor, resolve
├── neurons        # LIF, Dual-LIF reservoirs
├── attractor      # LVQ1 attractor field (one-shot capable)
└── episodic       # Episodic memory

exprimetal/
├── nlp/           # NLP implementation
│   ├── paper.md   # SNLI benchmark paper
│   └── snli_1.0/  # dataset
└── game/          # RL game implementation
    └── paper.md   # GridWorld RL paper
```

## Key Results

| Domain | Benchmark | Result | vs Baseline |
|--------|-----------|--------|-------------|
| NLP | SNLI (550k train, 10k test) | **43.9%** | 33.3% (majority) |
| Continual Learning | 3 sequential SNLI tasks | **0.8 pt forgetting** | 15-20 pts (dense nets) |
| RL | GridWorld 8×6 (100 ep) | **73% success** | 0% (random) |
| Graph Resolution | 61k nodes, 1.66M edges | **0.84s** | — |

## Properties

- **Zero backpropagation**: all learning is local (R-STDP, LVQ1 attraction/repulsion)
- **Zero GPU**: pure Rust, CPU only
- **Zero catastrophic forgetting**: decoupled representation (immutable) + classifier (plastic)
- **One-shot learning**: LVQ1 creates new classes from a single example
- **Modality-independent**: same engine drives NLP and RL

## Quick Start

```bash
# NLP benchmark (SNLI)
cargo run -p tso-nlp --release

# RL game (GridWorld)
cargo run -p tso-game --release
```

## Repository Structure

- `tso-engine/` — standalone cognitive engine (no modality-specific code)
- `exprimetal/nlp/` — NLP implementation (PPMI → SVD → Φ → LVQ1)
- `exprimetal/game/` — RL implementation (GridWorld with Q-learning + TSO)

## Paper

See [`exprimetal/nlp/paper.md`](exprimetal/nlp/paper.md) for the full theoretical foundation and SNLI benchmark.
