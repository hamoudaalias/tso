# TSO-RL: Topographic Stabilization Operator for Reinforcement Learning

## A neuromorphic game-playing agent with zero catastrophic forgetting

**Author:** Hamouda ALIAS
**Engine:** tso-engine (Cognitive Dissipation Theory)
**Implementation:** GridWorld navigation with one-shot danger recognition

---

This paper documents the second implementation of the TSO cognitive engine:
reinforcement learning in a grid-world environment. The agent uses the same
engine primitives (Graph, Φ, LIF, LVQ1) as the NLP variant, with no architectural
changes — only different encoders and action spaces.

**State encoding:** The agent receives a 16-dimensional state vector containing
a one-hot encoding of its current (x, y) position and a coarse bucket encoding
of the goal position, acting as a compass signal (goal-conditioned RL).
The wall layout is not encoded — collisions are discovered through
environment interaction and remembered via one-shot LVQ1 classification
(`add_class` on first collision).

### Results (100 episodes, 8×6 GridWorld)

| Metric | Value |
|--------|-------|
| Success rate | **73%** (73/100) |
| Best episode | 19 steps, 3 collisions |
| Average steps | 107 |
| Average collisions/ep | 11 |
| Learning method | Q-learning (tabular) + TSO Graph edges |
| Danger detection | LVQ1 one-shot (class 1 on first collision) |

The TSO engine components demonstrated concurrently:
- **Graph** stores  transition topology (Move=+1, Collision=-1, Goal=+2)
- **AttractorField** learns dangerous states in one shot via `add_class`
- **Dual-LIF** encodes temporal state sequences
- **Q-table** (external, 8×6×4 = 192 entries) enables goal-reaching behavior

This validates that the TSO cognitive engine is modality-independent:
the same `tso-engine` primitives that drive NLP inference also support
RL navigation, with zero architectural changes.
