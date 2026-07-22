# TSO Snail Jumper (Python)

Neuroevolution game where TSO (Topographic Stabilization Operator) replaces the neural network.

## Requirements

```bash
pip install pygame numpy
```

## How to run

```bash
python run.py
```

Click "Start With TSO Neuroevolution" to see TSO agents evolve.
The game shows the TSO brain's state: LIF activation, predicted class, distance to nearest prototype.

## How it works

- Each agent has a TSO brain (LIF reservoir + LVQ1 attractor field)
- The brain sees obstacle positions → encodes as vector → LIF integrates → LVQ1 classifies → action
- Evolution selects fittest agents, crosses over their prototype vectors, mutates them
- No backpropagation, no neural network weights — pure geometric prototypes
