# TSO-RL: Topographic Stabilization Operator for Reinforcement Learning

## A neuromorphic game-playing agent with zero catastrophic forgetting

**Author:** Hamouda ALIAS
**Engine:** tso-engine (Cognitive Dissipation Theory)
**Implementations:** GridWorld, MiniGrid (MemoryS7, OneShot-v0)

---

This paper documents the RL implementations of the TSO cognitive engine.
Three environments of increasing complexity validate that TSO's primitives
(Graph, Φ, LIF, DualLIF, LVQ1, AssociativeMemory, EpisodicMemory,
WorkingMemory, ActionMotor) support navigation, one-shot danger recognition,
and — critically — **one-shot object matching via native WorkingMemory**,
all without backpropagation.

---

## 1. GridWorld (8×6)

The initial RL validation: a simple navigation task where the agent must
reach a goal while avoiding walls discovered through interaction.

**State encoding:** 16D vector (one-hot position + bucket-coded goal).

### Results (100 episodes)

| Metric | Value |
|--------|-------|
| Success rate | **73%** (73/100) |
| Best episode | 19 steps, 3 collisions |
| Danger detection | LVQ1 one-shot (class 1 on first collision) |

Components: Graph (transition topology), AttractorField (danger), Dual-LIF,
Q-table (192 entries). Demonstrates modality independence: same engine
as NLP, different encoder.

---

## 2. MiniGrid MemoryS7 (7×7 partial observability)

A harder task: the agent navigates an S-shaped hallway connecting two rooms.
The goal room contains two objects — one matching the target seen at the
start, one distractor. The agent must reach the goal room and select the
correct object.

**Challenge:** The observation is a 7×7 egocentric grid with symbolic
channels (type, color, state). Both target and distractor are visible
from the start room through the hallway doorway, making this a
**discrimination task** rather than pure one-shot.

**State encoding:** SymbolicEncoder — extracts visible object signatures
(type one-hot, color, state, relative position) into a 29D vector.
Navigation uses a LIF-smoothed version of this encoding with count-based
exploration bonuses.

### Results (100 episodes)

| Metric | Value |
|--------|-------|
| Goal reach rate | **47%** |
| Correct object selection | **23%** |
| Learning method | Q-learning + count-based exploration |
| State dimension | 29 (symbolic encoding) |

The 47% reach rate validates that the SymbolicEncoder + LIF + Q-learning
pipeline can learn multi-step navigation under partial observability. The
gap between reach rate (47%) and correct selection (23%) reveals a
fundamental limitation: the **DualLIF is a mixer** — when both target and
distractor are simultaneously visible, the LIF state averages them,
preventing discrimination.

---

## 3. OneShot-v0 (theoretical breakthrough)

To isolate the one-shot matching capability, we designed a custom MiniGrid
environment that tests **pure working memory**:

**Layout:** A 10×4 corridor. The agent starts at left (1,1). A red ball
(target) sits at (3,2). At the far right, a matching red ball and a blue
key (distractor, never seen before) sit side by side. The agent must
explore, find the target, remember it, then later choose the matching
object. Navigation paths reach y=3 objects directly (14 steps) and y=1
objects via a gap at x=9 (28 steps).

**The critical insight:** The DualLIF is unsuitable for one-shot object
storage because it **averages** all inputs. When both objects are visible,
the slow LIF state is $S_{slow} = \text{mean}(A, B)$, giving equal alignment
to both. This is a theoretical limitation of leaky integration for
simultaneous discrimination. The engine's `AssociativeMemory` solves the
discrimination problem but lacks temporal integration — the `WorkingMemory`
module combines both in a single Rust-native primitive.

**Solution — WorkingMemory (Rust-native observe/store/reset):** The agent
uses `WorkingMemory`, which wraps `DualLIFState` (temporal integration)
and `AssociativeMemory` (content-addressable storage). On first encounter
with the target, `store()` saves its signature vector
$v_{target} = [\text{type}_{onehot}, \text{color}_{norm}]$. At each
subsequent step, `observe()` returns `None` until the stored target is
visible, then returns the matching data. The `ActionMotor` selects
navigation actions by aligning the dual-LIF context with action embeddings.

### Ablation results (50 episodes NORMAL, 50 episodes AMNÉSIQUE)

| Condition | Result |
|-----------|--------|
| **NORMAL** (WorkingMemory intact) | **50/50 (100%)** |
| **AMNÉSIQUE** (WorkingMemory reset every step) | **0/50 (0%) — timeout at 100 steps** |

The NORMAL condition achieves perfect recall on every episode regardless of
target position or object type. The AMNÉSIQUE condition, where `reset()` is
called before each `observe()` to erase the stored vector, fails every
episode — the agent wanders without guidance. This proves that WorkingMemory
is the **sole cause** of successful one-shot matching: without it, no other
cognitive primitive can compensate.

### Audit

The initial 100% action score was invalid due to three confounds identified
during audit:
1. **Fixed target**: The target was always Ball:red — the agent could learn
   "ball = good, key = bad" without memory.
2. **Global vision**: All objects were visible from step 0 — no memory
   retention across a blind corridor was required.
3. **Trivial path**: The first blocking object was always the matching
   object, so forward-until-stuck + pickup always succeeded regardless of
   memory content.

These confounds were corrected:
1. **Randomization**: Target type (ball/key) and color (red/green/blue/yellow)
   are randomized each episode (Section 3 environment).
2. **Blind corridor**: Thick walls (2-cell) block the agent's view of goal
   objects during transit. The agent sees ONLY the target at start, then
   NOTHING for 3 steps, then both goal objects simultaneously (Section 3
   environment).
3. **Memory isolation via ablation**: The AMNÉSIQUE control separates
   navigation skill from memory performance — all modules remain active
   except WorkingMemory, which is reset before each observation.

---

## 4. Theoretical Implications

### 4.1 DualLIF vs AssociativeMemory vs WorkingMemory

The DualLIF is a **temporal integrator** — it mixes all inputs over time,
making it ideal for context tracking but fundamentally unsuitable for
simultaneous object discrimination. The AssociativeMemory is a **content
store** — it preserves exact input vectors, enabling perfect similarity
matching. The `WorkingMemory` module bridges both by wrapping DualLIFState
(temporal smoothing) and AssociativeMemory (content storage) in a single
primitive with `observe()`, `store()`, and `reset()` methods. This mirrors
the cognitive distinction between **working memory** (episodic buffer,
maintaining current context) and **long-term memory** (semantic store,
retrieving exact matches), while providing a unified Rust-native interface.

### 4.2 One-Shot Matching = Geometric Binding

The success of WorkingMemory-based one-shot matching proves that
TSO achieves **geometric binding**: the agent binds the target's features
(type + color) into a single WorkingMemory vector and later unbinds it by
cosine similarity comparison. This is the visual equivalent of the Inverse
Motor mechanism in the NLP pipeline, where the LIF state (Query) aligns
with vocabulary embeddings (Keys) to select the next token. The same
operator — $\text{align}(S_{query}, K_{candidate})$ — drives both text
generation and visual object matching, confirming TSO's modality
independence.

### 4.3 The One-Shot Pur Test

OneShot-v0 will be released as a standard MiniGrid environment for
benchmarking one-shot visual memory. It isolates the working memory
component from navigation complexity, providing a clean metric for
memory-augmented agents.

---

## 5. Future Work

1. **Graph-based navigation planning** for KeyCorridor and RedBlueDoors:
   use the TSO Graph to plan multi-step sequences (pick key → open door →
   reach goal) by encoding subtask transitions as implication edges.
2. **Episodic memory replay:** store successful navigation sequences
   and replay them during exploration to accelerate learning.
3. **Hierarchical Actor-Critic:** use the TSO Critic to evaluate subtask
   completion (door opened = friction resolved) for structured exploration.
4. **PyO3 bridge:** expose all TSO primitives to Python for rapid
   environment prototyping (already implemented).

---

## 6. Conclusion

These RL implementations validate that TSO's primitives support the full
spectrum of reinforcement learning: exploration (count-based bonuses),
navigation (SymbolicEncoder + LIF + Q-learning), danger avoidance (LVQ1
one-shot), and — most importantly — **geometric working memory**
(WorkingMemory for one-shot object matching). The ablation on OneShot-v0
— **NORMAL 100% vs AMNÉSIQUE 0%** — confirms that TSO's WorkingMemory is
the necessary and sufficient cause of one-shot matching: without it, no
other cognitive primitive can guide the agent. This proves that TSO
possesses a pure geometric memory mechanism that requires no training,
no backpropagation, and no attention matrices — only cosine similarity
against a stored Rust-native vector.
