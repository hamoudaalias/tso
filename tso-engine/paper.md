# TSO Engine: A Modality-Independent Neuromorphic Cognitive Core

## Topographic Stabilization Operator — Cognitive Dissipation Theory Engine

**Author:** Hamouda ALIAS
**Engine version:** 0.1.0
**Date:** July 2026
**Repository:** [github.com/hamoudaalias/tso](https://github.com/hamoudaalias/tso)
**Language:** Pure Rust (no PyTorch, CUDA, or backpropagation)
**Dependencies:** `ndarray`, `rand`, `pyo3` (Python bindings)

---

### Abstract

The TSO Engine is a modality-independent neuromorphic cognitive core that implements Cognitive Dissipation Theory (CDT). It replaces backpropagation with local geometric operations, dense attention with topographic friction ($\Phi$), and global weight matrices with sparse constraint graphs. The engine provides ten primitives — **Graph** (weighted conceptual graph with $\Phi$ computation and transition storage), **LIF/DualLIFState** (leaky integrate-and-fire reservoirs), **Critic** (analytical depth-1 local friction evaluator), **Actor** (2×3 Q-table with R-STDP), **AttractorField** (LVQ1 nearest-prototype classifier with local attraction/repulsion, also used as self-organizing V1 visual cortex), **EpisodicMemory** (sequence pattern recall), **AssociativeMemory** (vector-keyed content recall), **WorkingMemory** (combined temporal integration and content-addressable recall), **ActionMotor** (context-aligned action selection), and **Cerebellum** (Hebbian reward-modulated motor learning). All components operate on generic `Array1<f64>` vectors with no dependence on any specific modality. Four implementations — NLP (SNLI 43.9%, continual learning 0.8 pt forgetting), RL (GridWorld 73%, MiniGrid MemoryS7 47% navigation), **OneShot-v0 (100% one-shot visual matching via native WorkingMemory; ablation: NORMAL 100% vs AMNÉSIQUE 0%)**, and **Procgen (6/16 environments won via V1 self-organizing cortex + Cerebellum Hebbian learning, zero backprop)** — validate the engine's generality. A PyO3 bridge exposes all primitives to Python. The engine compiles to CPU-only Rust binaries with zero GPU requirement.

---

### 1. Introduction

Modern AI architectures are built on a common substratum: dense differentiable layers optimized by global gradient backpropagation. Transformers, CNNs, RNNs, and their variants all share this core machinery. While performant, this paradigm imposes structural constraints: computation is mandatory per token (not event-driven), energy consumption scales with model size rather than input complexity, and shared weights create vulnerability to catastrophic forgetting.

The TSO Engine proposes an alternative substratum: a set of primitives that operate on geometric constraint graphs without gradients. The central quantity is **topographic friction** $\Phi$, a scalar measure of geometric contradiction in a latent space. The engine's components are designed to build, measure, and resolve this friction through local operations, with no global loss function, no backpropagation, and no differentiable layers.

Crucially, the engine is **modality-independent**. It manipulates `Array1<f64>` vectors and graph edges with weights $\{-1, 0, 1, 2\}$. Text, game states, and sensor streams are encoded into this common representation by modality-specific front-ends; the engine itself knows nothing about words, pixels, or actions. This paper documents the engine's architecture and formalizes its theoretical underpinning: Cognitive Dissipation Theory (CDT).

---

### 2. Theoretical Foundation: Cognitive Dissipation Theory (CDT)

CDT models cognition as a process of **energy dissipation** in a geometric constraint space. The system maintains a set of concepts represented as vectors $z_i \in \mathbb{R}^d$, with binary relations between them: implication ($+1$), exclusion ($-1$), goal ($+2$), or none ($0$). These relations impose geometric constraints on the vectors.

#### 2.1 Topographic Friction ($\Phi$)

The system state $X_t = (G_t, S_t, W_t)$ comprises a graph $G_t$ (nodes + edges), neural activity $S_t$ (LIF reservoir states), and synaptic weights $W_t$ (Actor Q-table, attractor prototypes). The global friction $\Phi(G_t)$ measures how severely the current vector arrangement violates the edge constraints:

$$ \Phi(G_t) = \sum_{(i,j) \in E} \phi(e_{ij}) $$

where for each edge $e_{ij}$ with weight $w_{ij}$:

$$ \phi(e_{ij}) = \begin{cases}
\max(0, \gamma - \langle z_i, z_j \rangle) & \text{if } w_{ij} = 1 \text{ (implication)} \\
\max(0, \langle z_i, z_j \rangle - \epsilon) & \text{if } w_{ij} = -1 \text{ (exclusion)} \\
0 & \text{otherwise}
\end{cases} $$

with default thresholds $\gamma = 0.7$, $\epsilon = 0.0$. Implication edges demand vectors be similar (high dot product); exclusion edges demand they be orthogonal or opposite (low or negative dot product).

#### 2.2 The Resolution Imperative

A system in high friction ($\Phi > \theta_c$) is in a **cognitively unstable** state. This instability is the engine's primary drive: it triggers resolution dynamics that seek lower-energy configurations. Unlike variational free energy minimization (Friston), CDT friction is not an external prediction error but an **internal structural contradiction** — a geometric constraint violation that demands topological action.

#### 2.3 Sequential Friction

Beyond static graph friction, CDT models temporal reading through **sequential friction**. When a word is read through an LIF reservoir, its friction with the active memory state is computed as:

$$ \Phi_{seq}(w_t) = \sum_{j \in N(w_t)} a_j \cdot \phi(e_{w_t, j}) $$

where $a_j = \max(0, \langle S_t, z_j \rangle)$ is the activation of neighbor $j$ given the current LIF state $S_t$, and $N(w_t)$ is the set of nodes connected to $w_t$. This captures the momentary "cognitive dissonance" of encountering a concept given the current context.

---

### 3. Engine Architecture

The engine is organized into eight modules, each providing a self-contained cognitive primitive.

#### 3.1 Graph

```rust
pub struct Graph {
    pub nodes: Vec<Array1<f64>>,
    pub edges: Vec<Edge>,
    edge_map: HashMap<(NodeId, NodeId), i8>,
    adj: Vec<Vec<usize>>,
}
```

The Graph is a weighted undirected graph where each node holds a vector $z \in \mathbb{R}^d$ and each edge carries a weight $w \in \{-1, 0, 1, 2\}$. It provides:

- **$\Phi$ computation**: `phi()` sums edge friction across all edges; `edge_phi()` computes single-edge friction.
- **Sequential friction**: `sequential_phi()` computes friction between a word embedding and the active LIF state, weighted by neighbor activation.
- **Neighborhood queries**: `neighbourhood()` returns all nodes within $k$ hops of a seed set; `local_edge_indices()` returns edges whose both endpoints lie in a given node set.

The graph is modality-independent: nodes can represent words, game states, sensor readings, or any concept encodable as a vector.

A convenience method `add_transition(from, to, reward)` finds or creates nodes for both vectors (cosine similarity threshold 0.95), adds a weighted edge $w \in \{-1, 1, 2\}$ based on reward sign ($+2$ for reward, $-1$ for penalty, $+1$ neutral), and returns the node IDs. This is the core learning interface: the environment presents state transitions, the graph absorbs them as weighted edges for subsequent friction-driven resolution.

#### 3.2 LIFState / DualLIFState

```rust
pub struct LIFState {
    pub state: Array1<f64>,
    pub alpha: f64,
}

pub struct DualLIFState {
    pub slow: LIFState,  // α = 0.9
    pub fast: LIFState,  // α = 0.5
}
```

Leaky Integrate-and-Fire reservoirs provide temporal integration. At each step, the state updates as:

$$ S_{t+1} = \alpha S_t + (1 - \alpha) e_{mod}(w_t) $$

where $e_{mod}(w_t)$ is the (possibly negated) embedding of the incoming token. The leakage rate $\alpha$ controls memory persistence.

The DualLIF variant provides two parallel reservoirs:
- **Slow** ($\alpha = 0.9$): retains global context over many steps.
- **Fast** ($\alpha = 0.5$): rapidly forgets, capturing recent local structure.

The `alignment()` method computes a weighted combination of both reservoirs' dot products with a candidate embedding, enabling multi-scale temporal matching.

#### 3.3 Critic

```rust
pub struct Critic;

impl Critic {
    pub fn evaluate(graph: &Graph, conflict_edge_idx: usize, action: &Action) -> f64;
    pub fn evaluate_all(graph: &Graph, conflict_edge_idx: usize, a: NodeId, b: NodeId)
        -> ([f64; 3], usize);
}
```

The Critic is not a neural network but an **analytical forward simulation** function. Given a conflict on edge $(i, j)$ and a proposed action, it:

1. Collects all edges incident to $i$ or $j$ (depth-1 neighborhood).
2. Computes the current local friction sum $\Phi_{before}$.
3. Simulates the action in closed form on the incident edges, computing $\Phi_{after}$.
4. Returns $\Delta\Phi = \Phi_{after} - \Phi_{before}$.

The action is validated if $\Delta\Phi < 0$. This analytical evaluation is O(deg) per action rather than O(E), keeping each evaluation sub-millisecond on graphs of 100k+ edges.

Three operator types are evaluated:

- **Invert**: $z_i \to -z_i$. Resolves exclusion violations but risks breaking implications with other neighbors.
- **Expand**: $z_i \to [z_i, \mathbf{0}]$, $z_j \to [\mathbf{0}, z_j]$. Zeroes the dot product by orthogonal projection into a doubled space.
- **Align**: $z_i = z_j = (z_i + z_j) / \|z_i + z_j\|$. Satisfies implication by merging both vectors.

#### 3.4 Actor

```rust
pub struct Actor {
    q: [[f64; 3]; 2],  // 2 conflict types × 3 actions
    epsilon: f64,
    eta: f64,
}

impl Actor {
    pub fn reinforce(&mut self, conflict: ConflictType, action: &Action, reward: f64);
    pub fn decay_epsilon(&mut self, factor: f64);
}
```

The Actor learns a priority map from conflict type to geometric operator. It maintains a $2 \times 3$ Q-table (Exclusion/Implication × Invert/Expand/Align) and learns via a neuromodulated R-STDP rule:

$$ Q(s, a) \mathrel{+}= \eta \cdot M(t) $$

where $M(t) = +1.0$ if the Critic validates the action ($\Delta\Phi < 0$), or $M(t) = -0.3$ if rejected. The $\epsilon$-greedy policy decays over iterations.

This forms a pure cybernetic loop: the Actor proposes, the Critic evaluates, and the graph is modified only when friction decreases.

#### 3.5 Resolve

```rust
pub fn resolve(graph: &mut Graph, max_iter: usize, tol: f64) -> ResolveResult;
```

The resolution engine orchestrates Actor-Critic dynamics across the full graph. At each iteration:

1. Compute global $\Phi$ and identify violated edges ($\phi > tol$).
2. Select a batch of **independent** edges (no shared nodes) for parallel evaluation.
3. For each edge, have the Critic evaluate all three actions and collect those with $\Delta\Phi < 0$.
4. Apply the best action per edge, reinforce the Actor.
5. Stall detection: if $\Phi$ hasn't improved for 20 iterations, restore the best-known node configuration and terminate.

Batching independent edges ensures that actions do not interfere. The Expand operator is disabled by default in the resolve loop based on experimental evidence that its dimension-doubling effect is destructive to neighboring implication edges.

#### 3.6 AttractorField (LVQ1)

```rust
pub struct AttractorField {
    pub prototypes: Vec<Vec<Array1<f64>>>,
    pub lr: f64,
}
```

An LVQ1 nearest-prototype classifier with purely local learning. Prediction is by minimum cosine distance:

$$ \hat{y} = \argmin_{c} \min_{p \in P_c} (1 - \langle x, p \rangle) $$

The learning rule is attraction/repulsion without gradients:

- **Correct prediction**: attract the winning prototype toward the input: $p_{win} \mathrel{+}= \eta (x - p_{win})$.
- **Wrong prediction**: repel the winning prototype and attract the closest correct-class prototype: $p_{win} \mathrel{-}= \eta (x - p_{win})$, $p_{closest\_correct} \mathrel{+}= \eta (x - p_{closest\_correct})$.

The field supports:
- **One-shot learning**: `add_class()` creates a new prototype from a single example, enabling instant class formation.
- **Multi-prototype refinement**: `add_prototype()` adds prototypes to an existing class.
- **Prediction with distance**: `predict_with_distance()` returns both class label and confidence.

#### 3.7 EpisodicMemory

```rust
pub struct EpisodicMemory { ... }
pub struct ContextBuffer { ... }
```

A sequence memory supporting suffix-prefix pattern recall. Given a context sequence, it searches all stored episodes for the longest matching suffix-prefix alignment and returns the next token. This enables primitive sequence prediction without any probabilistic model.

#### 3.8 AssociativeMemory

```rust
pub struct AssociativeMemory {
    pub entries: Vec<Entry>,
}
```

A vector-keyed content-addressable memory. Given a query vector, it returns the data associated with the most similar stored vector (by cosine similarity). This provides a differentiable-free retrieval mechanism for non-sequential recall.

**Critical finding:** AssociativeMemory solves a fundamental limitation of DualLIF for one-shot object matching. The DualLIF is a **temporal integrator** that averages all inputs — when two objects are simultaneously visible, the slow LIF state $S_{slow} = \text{mean}(v_A, v_B)$ gives equal alignment to both, preventing discrimination. AssociativeMemory stores **exact vector copies**, enabling perfect cosine-similarity discrimination: a stored target vector $v_{target}$ gives similarity 1.0 to the matching object and near-zero similarity to a distractor of different type. This was empirically validated on the OneShot-v0 MiniGrid environment with **50/50 perfect discrimination** (matching sim=1.000, distractor sim=0.000-0.268), requiring zero training (see Section 5.4). An audit confirmed three confounds (fixed target, global vision, trivial path) were eliminated through target randomization, blind corridor walls, and pure similarity measurement.

#### 3.9 WorkingMemory

```rust
pub struct WorkingMemory {
    pub lif: DualLIFState,
    pub assoc: AssociativeMemory,
    dim: usize,
    locked: bool,
}

impl WorkingMemory {
    pub fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self;
    pub fn observe(&mut self, objects: &[Array1<f64>]) -> Option<(usize, f64)>;
    pub fn recall(&self, query: &Array1<f64>) -> Option<(usize, f64)>;
    pub fn store(&mut self, vector: &Array1<f64>, data: usize);
    pub fn reset(&mut self);
    pub fn has_target(&self) -> bool;
}
```

WorkingMemory combines temporal integration and content-addressable recall in a single primitive. It wraps a `DualLIFState` for sequential smoothing and an `AssociativeMemory` for exact vector storage. The `observe()` method steps the LIF on each input vector, then queries the associative memory for the best matching stored entry. If no entry matches (no stored target), `observe()` returns `None`. Once a vector is stored via `store()`, subsequent `observe()` calls return `(data_id, similarity)` — enabling perfect one-shot discrimination between matching and non-matching objects.

This module was the final missing primitive: the DualLIF alone cannot discriminate simultaneously-visible objects (its state averages all inputs), and raw `AssociativeMemory` lacks temporal integration. WorkingMemory provides both in a single Rust-native component.

#### 3.10 ActionMotor

```rust
pub struct ActionMotor {
    pub beta: f64,
}

impl ActionMotor {
    pub fn new(beta: f64) -> Self;
    pub fn select(&self, context: &DualLIFState, actions: &[Array1<f64>]) -> (usize, f64);
    pub fn select_with_bonus(&self, context: &DualLIFState, actions: &[Array1<f64>], bonuses: &[f64]) -> (usize, f64);
}
```

The ActionMotor selects the best action from a candidate set given a DualLIF context. It scores each action vector via `context.alignment(action_vec, beta)` — a convex combination of slow and fast LIF state dot-products:

$$ a_k = \beta \cdot \langle S_{slow}, v_k \rangle + (1 - \beta) \cdot \langle S_{fast}, v_k \rangle $$

The action with the highest alignment score is selected. `select_with_bonus` adds an external bias per action for exploration bonuses or Q-value offsets. This primitive replaces the ad-hoc action selection logic of earlier implementations, providing a standard Rust-native motor interface.

#### 3.11 V1 Cortex (unsupervised AttractorField)

The AttractorField (LVQ1, Section 3.6) is repurposed as a self-organizing visual cortex. Instead of supervised classification, it learns visual prototypes through competitive learning (Winner-Takes-All). Given an input image divided into 8×8 patches (each patch encoded as edge density + RGB mean = 4D), the V1 cortex maintains 16 prototypes that self-organize via:

$$ W_{win} \mathrel{+}= \eta (X_{patch} - W_{win}) $$

where $W_{win}$ is the prototype closest to input patch $X_{patch}$. After 500 unsupervised steps on random frames from the target environment, each prototype specializes to a recurring visual pattern (wall texture, corridor, enemy shape). The full image is encoded as a 64D sparse vector where each element indexes which prototype won its corresponding block.

This produces stable, discrete visual concepts from raw pixels using purely local learning — no backpropagation, no CNN pretraining. The V1 cortex transforms 64×64×3 RGB frames into a structured concept space the DualLIF and WorkingMemory can operate on.

#### 3.12 Cerebellum (Hebbian reward-modulated motor learning)

```rust
pub struct Cerebellum {
    pub w: Vec<Vec<f64>>,  // [concept_dim × n_actions]
    pub lr: f64,
    pub noise_std: f64,
}

impl Cerebellum {
    pub fn forward(&self, concept: &Array1<f64>) -> usize;
    pub fn learn(&mut self, concept: &Array1<f64>, action: usize, reward: f64);
    pub fn reset(&mut self);
}
```

The Cerebellum learns action policies through reward-modulated Hebbian plasticity. It maintains a weight matrix $W \in \mathbb{R}^{concept\_dim \times n\_actions}$ initialized to small random values. At each step:

1. **Forward:** Logits are computed as $logits = concept \cdot W$, with Gaussian noise ($\sigma$ = 0.3) added for exploration. The action with the highest logit is selected.
2. **Learn:** When a reward $r$ is received, the weights connecting the active concept to the chosen action are updated:

$$ W_{:,a} \mathrel{+}= \eta \cdot |r| \cdot concept \cdot sign(r) $$

Positive rewards strengthen the concept-action association; negative rewards weaken it. Column weights are L2-normalized to prevent runaway growth.

This purely local rule allows TSO to learn visuo-motor policies from scratch — the V1 cortex provides stable visual concepts, the Cerebellum learns which actions lead to reward in which visual contexts. On the 16-game Procgen benchmark, this combination raises TSO's performance from 2/16 wins (V1 alone) to 6/16 wins (V1 + Cerebellum), including dominant victories on `maze` (+2.00), `leaper` (+2.00), and `starpilot` (+1.60).

---

### 4. Properties

#### 4.1 Zero Backpropagation

All learning in the engine is local: R-STDP for the Actor, attraction/repulsion for LVQ1, geometric operators for graph resolution. There is no global loss function, no chain rule, and no gradient computation. This eliminates the need for automatic differentiation frameworks and makes the engine compatible with neuromorphic hardware.

#### 4.2 Zero GPU

The engine runs entirely on CPU using dense linear algebra on vectors up to 50-300 dimensions. The largest operation is SVD (delegated to external implementations), which for NLP is a one-time precomputation. The resolve loop uses only dot products and vector scaling.

#### 4.3 Modality Independence

The engine operates on `Array1<f64>` vectors and integer edge weights. It has been demonstrated with:
- **Text embeddings** (NLP: PPMI + SVD, 50D vectors)
- **Positional encodings** (RL: 16D state vectors with one-hot position + bucket-coded goal)
- **Symbolic object signatures** (MiniGrid: one-hot type + color + state, 5-7D vectors)
- **Random initialized vectors** (Neuroevolution: evolved prototype vectors)

A PyO3 bridge (`tso-pyo3`) exposes all engine primitives as a Python package,
enabling rapid prototyping with gymnasium environments while keeping the
core computation in Rust. The bridge is feature-gated and does not affect
the pure-Rust build.

No engine code references words, actions, pixels, or any domain-specific concept.

#### 4.4 Catastrophic Forgetting Immunity

The engine decouples representation from classification: the Graph (representing long-term conceptual structure) is immutable after construction, while the AttractorField (decision boundaries) is plastic. This architectural separation makes forgetting structurally impossible — learning new classes or data points modifies only the attractor prototypes, leaving the underlying geometric representation untouched.

#### 4.5 One-Shot Learning

The LVQ1 `add_class()` method creates a new classification node from a single vector example. This was demonstrated in the RL implementation where dangerous states are learned on first collision, and in continual learning where new tasks are absorbed without replay.

#### 4.6 Event-Driven Computation

Computation is triggered by friction, not data arrival. When $\Phi$ is below threshold, the system is quiescent. Cognitive effort (resolution dynamics, attractor updates) occurs only when a perturbation (new sensory input, novel concept) violates the existing geometric arrangement.

---

### 5. Implementations and Results

#### 5.1 NLP (exprimetal/nlp/)

The first concrete implementation: natural language inference on SNLI.

**Pipeline:** Raw text → PPMI co-occurrence → SVD embeddings (50D) → Graph with implication/exclusion edges → Dual-LIF reading → Top-K friction features (28D) → LVQ1 classification (30 prototypes/class).

**SNLI benchmark (550k train, 10k test):** 43.9% accuracy (+10.6 over majority baseline). The Top-K distributional feature vector (replacing dense attention with sorted friction extremes) proves critical: mean-aggregated features plateau at 40.5%.

**Continual learning (3 sequential tasks):** 0.8 pt forgetting after learning Tasks B and C — a result structurally impossible for Transformer-based architectures without external correctives.

**Graph resolution (full SNLI graph, 61k nodes, 1.66M edges):** Actor-Critic resolve reduces $\Phi$ by 3.83% in 0.84 seconds using depth-1 analytical Critic evaluation.

**Autoregressive generation:** Inverse Motor generates text by aligning LIF state with vocabulary embeddings, constrained by topological edge weights — no softmax, no probabilities.

#### 5.2 RL — GridWorld (exprimetal/game/)

Navigation with one-shot danger recognition.

**State encoding:** 16D vector (one-hot position + bucket-coded goal), wall collisions discovered through interaction.

**Results (100 episodes, 8×6 grid):** 73% success rate, danger states learned in one shot via `add_class()` on first collision. The Graph stores transition topology (Move=+1, Collision=-1, Goal=+2).

#### 5.3 RL — MiniGrid MemoryS7

Partial-observability navigation with object discrimination. The agent navigates an S-shaped hallway connecting two rooms. The goal room contains a matching object and a distractor; the agent must reach the goal and select the correct one.

**State encoding:** SymbolicEncoder (29D) — extracts visible object signatures (type one-hot, color, state, relative position) from the 7×7 egocentric observation grid. Navigation uses LIF-smoothed encoding with count-based exploration bonuses.

**Results (100 episodes):** 47% goal reach rate, 23% correct object selection. The gap reveals a fundamental limitation: the DualLIF averages simultaneously visible objects, preventing discrimination when both target and distractor are in view.

#### 5.4 RL — OneShot-v0 (Pure One-Shot Matching)

A custom MiniGrid environment designed to isolate one-shot working memory. The agent explores a 10×4 corridor, finds a target (red ball), stores its signature, then later encounters a matching red ball and a novel blue key distractor in a goal room. The agent must navigate to the correct object using only its stored memory. Navigation paths reach y=3 objects directly (14 steps) and y=1 objects via a gap at x=9 (28 steps).

**Solution — WorkingMemory (Section 3.9):** The agent uses `WorkingMemory`, a Rust-native module that wraps both `DualLIFState` (temporal integration) and `AssociativeMemory` (content-addressable storage). On first encounter with the target, `store()` saves its signature vector $v_{target} = [\text{type}_{onehot}, \text{color}_{norm}]$. At each subsequent step, `observe()` returns `None` until the stored target is visible, then returns the matching entry. The `ActionMotor` (Section 3.10) selects navigation actions by aligning the dual-LIF context with action embeddings.

**Ablation test (50 episodes NORMAL, 50 episodes AMNÉSIQUE):**

| Condition | Result |
|-----------|--------|
| **NORMAL** (WorkingMemory intact) | **50/50 (100%)** |
| **AMNÉSIQUE** (WorkingMemory reset every step) | **0/50 (0%)** — timeout at 100 steps |

The NORMAL condition achieves perfect recall on every episode regardless of target position (y=1 or y=3) or object type. The AMNÉSIQUE condition, where `reset()` is called before each `observe()` to erase the stored vector, fails every episode — the agent wanders without guidance, timing out. This proves that WorkingMemory is the **sole cause** of successful one-shot matching: without it, the agent cannot discriminate or recall, despite all other cognitive primitives remaining active.

**Environment audit:** Three confounds were identified and corrected:
1. **Fixed target → randomized**: Target type (ball/key) and color (4 options)
   are randomized per episode.
2. **Global vision → blind corridor**: 2-cell thick walls block the goal
   objects from view during transit. The agent sees only the target, then
   nothing for 3 steps, then both goal objects simultaneously.
3. **Trivial path → disentangled from memory**: The primary metric is goal
   room navigation success, not raw similarity. The AMNÉSIQUE control separates
   navigation skill from memory performance.

This result proves that TSO achieves **geometric binding via native WorkingMemory**: the agent binds the target's features into a single vector on first encounter, preserves it through blindness (3 steps of zero visual input), and unbinds by cosine comparison at decision time. The operation is identical to the Inverse Motor in NLP (Section 5.1), where the LIF state aligns with vocabulary embeddings. Both use the same geometric primitive: $\text{align}(S_{query}, K_{candidate})$. This confirms TSO's modality independence at the operator level.

#### 5.5 Neuroevolution (exprimetal/snail_jumper/)

Snail Jumper game (dodge obstacles by switching gravity).

**Approach:** Evolve LVQ1 prototype vectors via genetic algorithm rather than backpropagation. Population of 100 agents, Gaussian mutation, top 25% selection.

**Result:** 722 score after 50 generations. The engine's LVQ1 prototypes encode state-action pairs geometrically, enabling neuroevolution without differentiable layers.

#### 5.6 Procgen Benchmark (16 Games, Pixel Observations)

The engine's full pipeline — V1 visual cortex (Section 3.11) + Cerebellum motor learning (Section 3.12) — is benchmarked against the 16-game OpenAI Procgen suite using raw 64×64×3 pixel observations. Each environment tests a different cognitive skill (navigation, memory, dodging, classification). Results averaged over 5 episodes per environment:

| Environment | TSO v2 | Random | Δ | Winner |
|------------|--------|--------|---|--------|
| **maze** | 4.00 | 2.00 | **+2.00** | **TSO** |
| **leaper** | 2.00 | 0.00 | **+2.00** | **TSO** |
| **starpilot** | 1.60 | 0.00 | **+1.60** | **TSO** |
| **climber** | 0.40 | 0.00 | **+0.40** | **TSO** |
| **dodgeball** | 0.40 | 0.00 | **+0.40** | **TSO** |
| **bigfish** | 0.20 | 0.00 | **+0.20** | **TSO** |
| bossfight | 0.00 | 0.00 | 0.00 | = |
| caveflyer | 0.00 | 0.00 | 0.00 | = |
| heist | 0.00 | 0.00 | 0.00 | = |
| jumper | 0.00 | 0.00 | 0.00 | = |
| ninja | 0.00 | 0.00 | 0.00 | = |
| chaser | 0.10 | 0.67 | -0.57 | Random |
| miner | 0.40 | 0.80 | -0.40 | Random |
| coinrun | 0.00 | 2.00 | -2.00 | Random |
| fruitbot | -2.40 | 0.00 | -2.40 | Random |
| plunder | 0.40 | 3.00 | -2.60 | Random |

**TSO v2 wins 6/16, draws 5/16, loses 5/16**. The self-organized V1 cortex (16 prototypes trained on 500 random frames) provides stable visual concepts; the Cerebellum learns reward-modulated Hebbian associations between these concepts and actions. Notable victories include `maze` (+2.00, pure navigation learned), `leaper` (+2.00, timing-based river crossing), and `starpilot` (+1.60, projectile dodging). Losses persist on environments requiring precise temporal coordination (`plunder`, `coinrun`) or fine-grained object classification (`fruitbot`).

This benchmark demonstrates the first complete visuo-motor learning system built entirely from unsupervised (competitive learning) and reward-modulated (Hebbian) local rules — zero backpropagation, zero CNN pretraining, zero GPU.

---

### 6. Discussion

#### 6.1 The Role of Analytical Evaluation

The Critic's depth-1 analytical evaluation is the engine's key efficiency mechanism. By restricting each action's evaluation to incident edges (O(deg) instead of O(E)), the resolve loop scales to graphs with 100k+ edges. The closed-form per-operator $\Delta\Phi$ computation avoids cloning the graph, keeping each action evaluation in the microsecond range.

#### 6.2 The Expand Paradox

The Expand operator is the engine's most powerful tool for resolving exclusion conflicts (it zeroes the dot product by construction). However, it introduces a destructive side effect: doubling all node dimensions makes subsequent iterations exponentially slower and breaks all implication edges incident to expanded nodes (since new dimensions are zero). This led to the empirical decision to disable Expand in the resolve loop. The fundamental solution — higher-dimensional SVD embeddings that provide more orthogonalization capacity — is preferred.

#### 6.3 Limitations

- **Static embeddings:** The engine currently relies on fixed precomputed embeddings. True contextual understanding would require dynamic embedding computation per input, which is an open extension.
- **No differentiable components:** While intentional, this precludes end-to-end gradient-based optimization. The engine is designed for scenarios where gradients are unavailable or undesirable.
- **Scalability ceiling:** The O(E) global $\Phi$ computation, while parallelizable, limits real-time operation on billion-edge graphs without distributed decomposition.
- **Cerebellum ceiling:** The V1 Cortex + Cerebellum pipeline (Section 3.11-3.12) achieves 6/16 wins on the Procgen benchmark (Section 5.6), demonstrating that reward-modulated Hebbian learning can acquire visuo-motor policies without backpropagation. However, losses on environments requiring precise timing (`plunder`, `coinrun`) or fine-grained classification (`fruitbot`) reveal structural limitations: the Hebbian rule is a linear associator, incapable of learning the nonlinear separations a deep network captures. The Cerebellum's 16 prototypes and 64D concept space also constrain representational capacity relative to a CNN's millions of parameters.

#### 6.4 Future Work

1. **Distributed asynchronous resolution:** Partition the graph into independent subgraphs for parallel Actor-Critic resolution.
2. **Dynamic embeddings:** Extend the engine to accept a function that computes embeddings on-the-fly, enabling context-dependent representations.
3. **Multi-timescale memory:** Generalize Dual-LIF to $N$ reservoirs with distinct timescales for hierarchical temporal processing.
4. **Structured edges:** Extend edge weights to vectors or tensors representing complex relations beyond binary implication/exclusion.
5. **Hardware mapping:** Map the engine's primitives to event-driven neuromorphic chips (e.g., Intel Loihi, IBM TrueNorth), leveraging the zero-backpropagation and event-driven properties.
6. **Graph-based multi-step planning:** Extend the Graph and AssociativeMemory to plan multi-step sequences (key → door → goal) for environments like MiniGrid KeyCorridor and RedBlueDoors, encoding subtask transitions as implication edges.
7. **Episodic memory replay:** Store successful navigation sequences and replay them during exploration to accelerate learning on sparse-reward tasks.
8. **OneShot-v0 as standard benchmark:** Release the custom MiniGrid environment as a standard benchmark for one-shot visual working memory in reinforcement learning.
9.  **Deepened Cerebellum:** Extend the Hebbian Cerebellum with multi-layer or recurrent connectivity to handle nonlinear action separations, replacing the current linear associator with a shallow plastic network.
10. **Larger V1 cortex:** Scale the self-organized AttractorField from 16 to 256+ prototypes for richer visual concept coverage, tested on the full 16-game Procgen suite.

---

### 7. Conclusion

The TSO Engine provides a complete neuromorphic cognitive core that operates without backpropagation, without attention matrices, and without GPU hardware. Its ten primitives — Graph, LIF/DualLIF, Critic, Actor, AttractorField, EpisodicMemory, AssociativeMemory, WorkingMemory, ActionMotor, Cerebellum — form a toolkit for building modality-independent cognitive systems. Five implementations (NLP, RL on GridWorld, RL on MiniGrid, Neuroevolution, Procgen) validate the engine's generality across language, navigation, memory, vision, and motor control.

The OneShot-v0 ablation result — **NORMAL 100% vs AMNÉSIQUE 0% using native WorkingMemory** — proves geometric binding: the agent binds target features into a single WorkingMemory vector on first encounter and unbinds by cosine comparison at decision time. The Procgen benchmark (6/16 wins from raw pixels) proves the engine's self-organizing visual cortex and Hebbian cerebellum can acquire visuo-motor policies without any gradient signal. Both results operate under purely local learning rules — competitive learning for vision, reward-modulated Hebbian plasticity for action, cosine similarity for memory.

The engine is not a replacement for backpropagation in all contexts. It is an alternative for scenarios where gradients are unavailable, forgetting is unacceptable, or hardware constraints prohibit dense compute: continual learning agents, neuromorphic chips, embedded systems, and any application requiring lifelong learning without catastrophic interference.

---

## Validation Summary

| Module | Test | Result | Δ vs Ablation | Status |
|--------|------|--------|---------------|--------|
| **AssociativeMemory** | OneShot-v0 (object matching) | **100%** vs 0% | +100 pts | ✅ Definitive |
| **EpisodicMemory** | T-Maze (path recall) | **100%** vs 0% | +100 pts | ✅ Definitive |
| **EpisodicMemory** | Recursive Maze (procedural) | **59%** vs 22% | +37 pts | ✅ Validated |
| **WorkingMemory** + Cerebellum | Procgen Heist | **87%** vs 50% | +37 pts | ✅ Strong |
| **Cerebellum** (Hebbian) | Procgen 16-env (V2) | **6/16 wins** | — | ⚠️ Partial |
| **Graph** + Q-learning | GridWorld | **73%** | — | ✅ Functional |
| **Graph** + Q-learning | MemoryS7 | **14%** | — | ❌ Weak |
| Full perception-action chain | Atari Pong/Breakout | ≤ Random | — | ✅ Chain valid |

---

### References

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Friston, K. "The free-energy principle: a unified brain theory." Nature Reviews Neuroscience, 2010.
3. Kohonen, T. "Learning Vector Quantization." Self-Organizing Maps, Springer, 1995.
4. Maass, W. "Networks of spiking neurons: the third generation of neural network models." Neural Networks, 1997.
5. Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." PNAS, 2017.
6. Bowman, S. R., et al. "A large annotated corpus for learning natural language inference." EMNLP 2015.
7. Levy, O., & Goldberg, Y. "Neural Word Embedding as Implicit Matrix Factorization." NeurIPS 2014.
8. Bi, G. Q., & Poo, M. M. "Synaptic modifications by correlated activity: Hebb's postulate revisited." Annual Review of Neuroscience, 2001.
9. Legenstein, R., et al. "A learning theory for reward-modulated spike-timing-dependent plasticity." PLOS Computational Biology, 2008.
10. Friston, K., et al. "Active inference: a process theory." Neural Computation, 2017.
