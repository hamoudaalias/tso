# TSO — Topographic Stabilization Operator

A neuromorphic cognitive engine where topographic friction (Φ) replaces attention — no backpropagation, no Transformers, no GPU.

## Architecture

```
tso-engine/        # Cognitive engine (modality-independent)
├── core           # Graph, Φ, Critic, Actor, resolve
├── neurons        # LIF, Dual-LIF reservoirs
├── attractor      # LVQ1 attractor field (one-shot capable)
├── episodic       # Episodic memory
└── memory         # Associative memory (k-NN cosine)

exprimetal/
├── nlp/           # NLP implementation (SNLI)
├── game/          # RL — GridWorld + MiniGrid
├── snail_jumper/  # Neuroevolution
├── tso_pyo3/      # Python bindings (maturin + PyO3)
└── minigrid_py/   # MiniGrid agent & OneShot-v0 environment
```

## Validation Report

### ✅ Cognitive Proofs (Memory & One-Shot)

| Test | NORMAL | AMNÉSIQUE | Δ | Status |
|------|--------|-----------|---|--------|
| **OneShot-v0** — object matching via AssociativeMemory | **100%** | 0% | **+100%** | Definitive |
| **T-Maze** — episodic recall via EpisodicMemory | **100%** | 0% | **+100%** | Definitive |
| **DoorMatch-v0** — color recall via AssociativeMemory | **100%** | 48% | **+52%** | Definitive |
| **KeyMatch-v0** — key→door via WorkingMemory | **100%** | 51% | **+49%** | Definitive |
| **Procgen Heist** — working memory + exploration | **87%** | 50% | **+37%** | Strong |
| **Recursive Maze** — procedural maze episodic recall | **59%** | 22% | **+37%** | Validated |

### ⚙️ RL & Motor Learning

| Benchmark | Result | Baseline | Status |
|-----------|--------|----------|--------|
| SNLI (NLP without backprop) | **43.9%** | 33.3% (majority) | Functional |
| Continual Learning (3 tasks) | **0.8 pt forgetting** | 15-20 pts (dense nets) | Functional |
| GridWorld 8×6 | **73% success** | 0% (random) | Functional |
| Procgen 16-env (V2) | **6/16 wins** | 0/16 (random) | Partial |
| Procgen 16-env (V3) | **1/16 wins** | 0/16 (random) | Partial |
| MiniGrid MemoryS7 | **14% success** | — | Weak |
| Atari Pong/Breakout | ≤ Random | — | Chain valid |
| Snail Jumper (neuroevolution) | **722 score** | — | Functional |

### Architecture

```
tso-engine/        # Cognitive engine (modality-independent)
├── core           # Graph, Φ, Critic, Actor, resolve
├── neurons        # LIF, Dual-LIF reservoirs
├── attractor      # LVQ1 attractor field (one-shot capable)
├── episodic       # Episodic memory (suffix-prefix matching)
├── memory         # Associative memory (k-NN cosine)
├── working_memory # DualLIF + AssociativeMemory
├── action         # ActionMotor (alignment-based selection)
└── cerebellum     # Reward-modulated Hebbian learning

exprimetal/
├── nlp/           # NLP implementation (SNLI)
├── game/          # RL — GridWorld
├── snail_jumper/  # Neuroevolution
├── tso_pyo3/      # Python bindings (maturin + PyO3)
├── minigrid_py/   # MiniGrid agents & OneShot-v0
├── procgen/       # Procgen 16-env benchmark
└── atari_py/      # Atari Pong/Breakout validation
```

## Properties

- **Zero backpropagation**: all learning is local (R-STDP, LVQ1 attraction/repulsion)
- **Zero GPU**: pure Rust, CPU only
- **Zero catastrophic forgetting**: decoupled representation (immutable) + classifier (plastic)
- **One-shot learning**: LVQ1 creates new classes from a single example
- **Geometric working memory**: AssociativeMemory (k-NN cosine) for one-shot object matching
- **Modality-independent**: same engine drives NLP, RL, and visual object matching
- **Python bindings**: PyO3 bridge exposes all primitives via `pip install tso-pyo3`

## Quick Start

```bash
# NLP benchmark (SNLI)
cargo run -p tso-nlp --release

# RL game (GridWorld)
cargo run -p tso-game --release

# MiniGrid with Python bindings
cd exprimetal/tso_pyo3 && maturin build --release && pip install ../target/wheels/tso_pyo3*.whl
python3 exprimetal/minigrid_py/train_symbolic.py MiniGrid-MemoryS7-v0

# OneShot-v0 test (100% one-shot matching)
python3 exprimetal/minigrid_py/test_oneshot.py
```

## Repository Structure

- `tso-engine/` — standalone cognitive engine (no modality-specific code)
- `exprimetal/nlp/` — NLP implementation (PPMI → SVD → Φ → LVQ1)
- `exprimetal/game/` — RL: GridWorld + MiniGrid (MemoryS7, OneShot-v0)
- `exprimetal/snail_jumper/` — Neuroevolution implementation
- `exprimetal/tso_pyo3/` — PyO3 Python bindings for tso-engine
- `exprimetal/minigrid_py/` — MiniGrid agent scripts and custom environments

## Theoretical Highlights

1. **DualLIF ≠ Working Memory**: The DualLIF is a temporal integrator that averages inputs — it cannot discriminate simultaneously visible objects. This is a theoretical limitation of leaky integration for one-shot tasks.

2. **AssociativeMemory = Geometric Binding**: The vector-keyed AssociativeMemory (k-NN cosine) stores exact vector copies and retrieves by similarity, enabling perfect one-shot object matching. This is the visual equivalent of the Inverse Motor in NLP generation.

3. **100% One-Shot Discrimination** (ablation-confirmed): On custom OneShot-v0 (target-randomized, blind-corridor, positions randomized), TSO's AssociativeMemory achieves 100% discrimination (matching sim=1.000, distractor ≤0.268). Ablation (random similarity) collapses to 54% (chance). No training, no backprop, no attention.

## Paper

See [`tso-engine/paper.md`](tso-engine/paper.md) for the full engine specification.
See [`exprimetal/game/paper.md`](exprimetal/game/paper.md) for RL results including OneShot-v0.
See [`exprimetal/nlp/paper.md`](exprimetal/nlp/paper.md) for the SNLI benchmark.
