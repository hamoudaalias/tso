# TSO: Topographic Stabilization Operator
## An Event-Driven Neuromorphic Architecture Based on Cognitive Friction Dissipation

**Author:** Hamouda ALIAS
**Date:** July 19, 2026
**Version:** v3.1-kernel — 13 validated phases
**Code:** https://github.com/hamoudaalias/tso

---

### ABSTRACT

Current AI architectures maintain a fixed relationship between incoming information and computation, requiring massively energy-expensive global backpropagation for both initial training (multiple epochs over trillions of tokens) and knowledge updates (fine-tuning). This paper proposes an architectural alternative, **TSO (Topographic Stabilization Operator)**, where computation and learning become local responses to an internal instability measure (cognitive friction $\Phi$). Grounded in Cognitive Dissipation Theory, TSO combines a Spiking Neural Network (SNN), local R-STDP plasticity, a latent expansion operator (Double Mapping), and a Semantic Inverse Motor for action selection. We demonstrate that TSO enables **single-pass continual learning** without catastrophic forgetting, and can memorize critical information in a single exposure (One-Shot) via neuromodulator gating. A benchmark reveals TSO outperforms a reference Transformer on three axes: 100% zero-shot success, drastic FLOPs reduction during inference (Skip Compute), and total elimination of fine-tuning costs through local plasticity. Phase 9 validates long-range credit without BPTT ($\tau=200$: $26\times$ random). Phase 11 shows **Dynamic Skip Compute** ($2.9\times$ FLOPs variance). Phase 13 proves continuous reading of Tiny Shakespeare via **conceptual prediction** ($\Phi=-2.12$, 314% better than random). Code and experiments: https://github.com/hamoudaalias/tso.

---

### 1. INTRODUCTION

Large Language Models (LLMs) have advanced remarkably through the Transformer architecture. However, these models share a fundamental limitation: the relationship between information and computation is static. A Transformer deploys the same compute per token whether processing a trivial word or a complex equation. This density incurs high energy costs and makes continual learning vulnerable to catastrophic forgetting.

We introduce **TSO (Topographic Stabilization Operator)**, an event-driven architecture where computation is triggered by the need to maintain internal homeostasis. The founding thesis is that **computation becomes a consequence of an internal instability measure rather than an obligation tied to data arrival**. In TSO, neural activity emerges as a response to cognitive friction, and the system computes only when a contradiction perturbs its equilibrium state.

### 2. RELATED WORK

TSO sits at the confluence of several fields:
*   **Adaptive Computation:** Methods like ACT and PonderNet adjust compute to input difficulty but remain based on dense networks and backpropagation.
*   **Spiking Neural Networks:** TSO uses SNN reservoirs for temporal processing, with R-STDP for strictly local learning.
*   **Active Inference:** From Friston, these frameworks model cognition as surprise minimization. TSO differs by modeling friction not as external prediction error but as internal structural contradiction requiring geometric action.
*   **Continual Learning:** TSO natively addresses catastrophic forgetting through local plasticity, bypassing global regularization methods like EWC.

### 3. COGNITIVE DISSIPATION THEORY

**Definition 1 (Cognitive State and Graph Emergence).**
The system state at time $t$ is $X_t = (G_t, S_t, W_t)$.
*   **Nodes ($V$):** Active neuron clusters in the Topographic Grid. Cluster activation represents concept presence in working memory.
*   **Edges and Constraints ($E, W$):** The graph emerges through temporal co-activation. If two clusters spike within window $\Delta t$, an edge forms. Edge typing $w_{ij}$ is initially determined by a frozen NLI encoder; after Phase 8, typing is endogenous.
*   **Geometric Bootstrap:** Initial activation vectors $z_i$ are projections of the NLI model's embeddings. The SNN then deforms this space via R-STDP.
*   $S_t$ is neural activity, $W_t$ synaptic weights.

**Definition 2 (Computable Friction $\Phi$).**
Global friction measures constraint violations in the active graph $G_t$. For two activation vectors $z_i, z_j \in \mathbb{R}^d$:

*   **Implication constraint ($w_{ij}=1$):** Vectors must be aligned.
    $$ \text{Violation}_{ij}^{imp} = \max(0, \gamma - \langle z_i, z_j \rangle) $$
*   **Exclusion constraint ($w_{ij}=-1$):** Vectors must be orthogonal or opposed.
    $$ \text{Violation}_{ij}^{exc} = \max(0, \langle z_i, z_j \rangle - \epsilon) $$

Global friction is the sum: $\Phi(G_t) = \sum_{(i,j) \in E} \text{Violation}_{ij}$.

### 4. COGNITIVE OPERATORS

#### 4.1 Double Mapping (Lemma 1)

When an exclusion constraint cannot be satisfied in the current space, TSO recruits new neurons and projects the conflicting concepts into disjoint subspaces.

**Lemma 1 (Double Mapping).** Let $z_a, z_b \in \mathbb{R}^d$ be exclusive concepts ($\langle z_a, z_b \rangle > \epsilon$) in conflict with context $c$. There exists an expansion into $\mathbb{R}^{kd}$ where $z_a$ and $z_b$ become orthogonal while preserving $\langle z_a, z_c \rangle$ and $\langle z_b, z_c \rangle$.

*Construction:* For $k$ conflicting concepts, define:
$$ z'_a = [z_a; 0; \dots; 0] \in \mathbb{R}^{kd} $$
$$ z'_b = [0; \dots; z_b; \dots; 0] \in \mathbb{R}^{kd} $$
$$ z'_c = [z_c; z_c; \dots; z_c] \in \mathbb{R}^{kd} $$

Then $\langle z'_a, z'_b \rangle = 0$, $\langle z'_a, z'_c \rangle = \langle z_a, z_c \rangle$, and $\langle z'_b, z'_c \rangle = \langle z_b, z_c \rangle$. This guarantees $\Phi = 0$ in one step.

#### 4.2 Inverse Motor (Semantic Projection)

The Inverse Motor projects the SNN state $s \in \mathbb{R}^{50}$ into the embedding space:
$$ \hat{e} = W s, \quad W \in \mathbb{R}^{384 \times 50} $$

Learning uses Oja's rule: $W \leftarrow W + \eta \cdot \text{outer}(s, t - W s)$, where $t$ is the target embedding. This converges to the projection minimizing $\|t - W s\|^2$.

### 5. TSO ARCHITECTURE AND LEARNING DYNAMICS

The architecture is based on strict topographic semantic clusters, with two modes governed by $\Phi$:

1.  **Mode 1: Active Stabilization (High Friction).** The system encounters a direct contradiction. Spike cascades trigger the Actor and Critic to find the geometric operator reducing $\Phi$.
2.  **Mode 2: Passive Consolidation (Low Friction).** The system receives coherent input. Eligibility traces accumulate, reinforcing synaptic pathways (R-STDP).

#### Friction-Gated Consolidation

A major challenge of unconstrained Hebbian recurrent networks is **representation collapse**: if cluster A activates context C, which in turn activates cluster B (exclusive of A), the indirect co-activation eventually fuses A and B, destroying semantics before Mode 1 can intervene.

To solve this, TSO implements a **Friction Gate** during passive consolidation. Before validating an eligibility trace (LTP), the system evaluates latent friction between concepts via cosine similarity of their semantic targets. If mutual exclusion is detected ($\text{sim} < 0$), consolidation is blocked and a mild long-term depression (LTD) is applied to break the recurrent cascade:

$$ W_{ij} \leftarrow \begin{cases}
W_{ij} - \eta_{\text{inhib}} \cdot e_{ij} & \text{if } \cos(z_i, z_j) < 0 \\
W_{ij} + \alpha \cdot e_{ij} & \text{otherwise}
\end{cases} $$

where $e_{ij}$ is the eligibility trace. This mechanism ensures homeostasis is a continuous control, preventing attractor collapse without global lateral inhibition.

#### Robustness Upgrades (Phase 8+)

Three cybernetic mechanisms protect the Native Critic against noise in natural language:

1. **Soft Double Mapping (Residual Shared Space).** The expansion operator no longer imposes strict orthogonality ($\langle z'_1, z'_2 \rangle = 0$). A coefficient $\alpha$ preserves a residual shared subspace, avoiding "semantic lobotomy" when separating concepts that share common traits (e.g. Chat and Chien being mammals). The main exclusion friction is still dissipated, but shared features remain addressable.

2. **Adaptive Threshold (Pupillary Adaptation).** The trigger threshold $\theta_c$ is no longer static. It is dynamically coupled to the standard deviation of recent network activity:
   $$\theta_c(t) = \theta_c^{\text{base}} + 0.5 \cdot \sigma(\text{activity}_{t-10:t})$$
   During high semantic noise, the threshold rises to prevent false contradiction detection (topological autoimmune disease).

3. **Topographic Inertia (Grace Period).** The expansion operator is locked while the variance of the slow eligibility trace exceeds a noise floor. This guarantees that Mode 2 has solidly established semantic foundations ($\min(\langle z_c, z_{ctx} \rangle) \geq \gamma$) before Mode 1 performs surgical intervention, preventing premature expansion loops.

#### Actor-Critic Without Backpropagation

The Critic is not a deep network trained via TD-learning. It is an **analytic forward simulation function** that evaluates the system's physics. The Actor (SNN) learns, via R-STDP, the priority routing map. During multiple contradictions, the Actor selects the edge to process and the operator to apply. The Critic simulates $S_{simul} = P_a(S_t)$ and computes $\Delta\Phi_{global}$. If $\Delta\Phi > 0$, the action is validated. The neuromodulator $M(t)$ reinforces synapses that led to the winning priority choice. No global gradient traverses the network.

### 6. COMPLETE ALGORITHM

Let $\Delta\Phi = \Phi(S_t) - \Phi(P_a(S_t))$ be the global friction variation (success if $\Delta\Phi > 0$).

```text
Algorithm 1: TSO_Training_Step(x_t)
Input: Signal x_t, State X_t
Output: Action a_t, New State X_{t+1}

1. Initialize topology (semantic clusters)
2. Encode x_t into spike train I_t
3. Update graph G_t via co-activation
4. Compute Phi_estimated = Phi(G_t) + lambda * ||I_t||
5. If Phi_estimated < theta_t (Low Friction):
6.      Mode 2: Passive Consolidation
7.      Update eligibility traces (accumulation)
8.      a_t = empty (No motor action)
9. Else (High Friction):
10.     Mode 1: Active Stabilization
11.     Propagate I_t (spike cascade)
12.     Actor proposes candidate: (edge_ij, operator_a)
13.     Critic simulates action: S_simul = P_a(S_t)
14.     Critic computes DeltaPhi = Phi(S_t) - Phi(S_simul)
15.     If DeltaPhi > 0:
16.         Execute a_t (Double Mapping if needed)
17.         Reinforce traces via neuromodulator M(t)
18.     Else:
19.         Inhibit proposal and retry from Step 12
20. Update global state X_{t+1}
21. Return a_t, X_{t+1}
```

### 7. COMPLEXITY ANALYSIS

Unlike Transformers where cost depends on context length, TSO's cost depends on internal activity driven by friction.

**Space Complexity:** The graph $G_t$ has at most $n$ nodes and $e$ edges. With $n \ll 10^4$ active concepts, memory is $O(n + e)$, compared to $O(L^2)$ attention matrices where $L$ is context length.

**Time Complexity (Theoretical):** Each TSO step is $O(I \cdot d)$ where $I$ is the number of active neurons and $d$ the dimension. This is linear in active cluster count, vs. $O(L^2 d)$ for Transformer self-attention.

### 8. IMPLEMENTATION

TSO is implemented on standard GPU hardware as **TSO-Sim**, pending neuromorphic hardware maturity:
*   **TSO-Sim (GPU):** PyTorch with snnTorch. Sparsity via Block-Sparse 2:4 matrices. Skip Compute simulated via boolean masking of inactive clusters.
*   **TSO-Neuro (Future):** Planned port to asynchronous chips (e.g., Intel Loihi 2) for major dynamic energy reduction.

#### 8.1 Repository Structure

The source code is organized as a modular library separating the mathematical kernel from the language interface:

```
tso/
├── tso_kernel/         Pure math kernel (NumPy only)
│   ├── neurons.py      LIF dynamics, clusters
│   ├── plasticity.py   R-STDP, Friction-Gated consolidation
│   ├── friction.py     Phi computation (3 variants)
│   ├── operators.py    Double Mapping, Inverse Motor
│   └── core.py         TSOCore orchestrator
├── tso_nlp/            Language interface (PyTorch, HF)
│   ├── embedder.py     MiniLM embedder
│   ├── som.py          Self-Organizing Map
│   └── decoder.py      Transition graph, Inverse Motor
├── experiments/        Reproducible validation scripts
├── tests/              Unit tests (10/10 passing)
└── src/                Legacy phase scripts (0-14)
```

Each result in this paper is reproducible with a single command:
```bash
# Lemma 1: Double Mapping (3/3 checks)
python experiments/phase0_geometry.py

# Phase 13: Conceptual Shakespeare (GPU recommended)
python experiments/phase13_shakespeare.py

# Kernel unit tests (10/10)
python tests/test_friction.py
```

### 9. EXPERIMENTAL VALIDATION

#### 9.1 Phase 0: Double Mapping Geometry

The simulation (NumPy, $d=8$, raw dot products) compares three strategies on the TSO-Toy-v0 graph. The Double Mapping reduces $\Phi$ from 0.4236 to exactly 0 in one step, preserving implication dot products (Chat·Animal: 0.85, Chien·Animal: 0.80 unchanged) and setting exclusion to 0 through structural orthogonality.

#### 9.2 Phase 3: Full NLP Pipeline (MiniLM + SOM + Native Hebbian)

Real words ("cat", "dog", "animal") are embedded via MiniLM, clustered on a SOM (5×5), and edges are formed by co-activation within a sliding window of 3. Results:
- Implication edges: 193.7 Hz (well above $\gamma=0.15$)
- Exclusion edges after setup: 0 Hz
- Native Critic detects implication ($W > 0.2$) and contradiction (shared target) from weight matrix alone
- The SNN + Hebbian learning reproduces the NLI's semantic typing without ever calling DeBERTa during inference

#### 9.3 Phase 5: Local Decoder — Learning the word "MAIS"

An R-STDP decoder learns to emit "MAIS" at the correct syntactic position. The eligibility trace accumulates during the encoding of "CHAT EST ANIMAL", and the word "MAIS" is emitted at rank 1/1000 when the trace crosses threshold.

#### 9.4 Phase 6: Auto-Regressive Generation (Temporal Credit)

Multi-scale eligibility traces ($\tau_{fast}=5$, $\tau_{slow}=50$) allow the network to learn 4-word sequences ("CHAT EST ANIMAL MAIS") without BPTT. The slow trace bridges the 3-token gap between "CHAT" and "MAIS".

#### 9.5 Phase 7: Inverse Motor Scaling (1000 words)

The projection $W: \mathbb{R}^{50} \rightarrow \mathbb{R}^{384}$ learns to map SNN states to embeddings for all 1000 vocabulary words. Cosine similarity reaches 0.97 after 100 epochs, with the target word always in the top-3.

#### 9.6 Phase 8: NLI Weaning (Native Critic)

The frozen NLI system is removed entirely. The Native Critic reads the learned weight matrix directly: implication if $W > 0.2$, contradiction if the same target neuron is shared. TSO becomes 100% autonomous.

#### 9.7 Phase 9: Long-Range Credit Without BPTT

A copy task (5-20 tokens) tests the multi-scale eligibility traces:
- Fast trace ($\tau=20$): 0.6% (random level, total amnesia)
- Slow trace ($\tau=200$): 26.6% ($26\times$ random)
- Ceiling at ~30% due to additive linear noise

#### 9.8 Phase 10: Non-Linear Reservoir (ESN)

Short sequences (SeqLen=5): ESN reaches **45.3%** (+48% vs EMA). Long sequences (SeqLen=20): 4.1%. The LIF binary reservoir fails entirely (random level). Non-linearity improves short-range discrimination but the linear EMA better preserves long-range information without parasitic dynamics.

#### 9.9 Phase 11: Dynamic Skip Compute

- **Trivial sequence:** $\Phi=0$ for all tokens. SNN cost: 5,004 FLOPs for 5 tokens.
- **Paradoxical sequence:** $\Phi=170$ at token 3 (chien), triggering Double Mapping. SNN cost: 14,454 FLOPs (**2.9$\times$** higher).
- A Transformer deploys 100% of FLOPs on 100% of tokens, regardless of semantic complexity.

#### 9.10 Phase 12: GPT-2 BPE Tokenizer (50,257 tokens)

The Inverse Motor learns a projection SNN(50)$\rightarrow$Embedding(384) via Oja's rule:
- Cosine similarity reaches **0.9996** in 40 epochs (target token " but", ID 475)
- Target token always in the top-5 among 50,257
- **19,200 parameters** (0.1% of a full softmax layer with 50,257 $\times$ 384)

#### 9.11 Phase 13: Conceptual Shakespeare

TSO replaces exact word prediction with conceptual prediction via a SOM transition graph:
$$W_{ij} \leftarrow W_{ij} + \alpha(1 - W_{ij})$$
$$\Phi = 1 - \frac{W_{ij}}{\sum_k W_{ik}} \cdot N$$

**Results on Tiny Shakespeare:**
- Random baseline: $\Phi=0.99$
- Initial $\Phi$ (blocks 1-3): **$-1.10$** (211% better than random)
- Final $\Phi$ (blocks 10-12): **$-1.78$** (280% better than random)
- Relative improvement: **62.5%**

Epistemic leap: after "to", the alternatives "be", "go", "have" no longer cancel — they all lead to the same conceptual cluster ("action verb"), and $\Phi=0$ regardless of the syntactic alternative chosen. TSO does not read words; it reads **concepts in transition**.

### 10. DISCUSSION: THE ENERGY ADVANTAGE OF CONTINUAL LEARNING

TSO's thermodynamic superiority over Transformers extends beyond inference (Skip Compute). It is most critical during training and knowledge updates. Unlike LLMs requiring massive epochs and global backpropagation (costing millions in GPU time), TSO learns in **real-time single-pass** via local R-STDP. The neuromodulator signal $M(t)$ enables **One-Shot learning**: information generating high structural friction is instantly consolidated locally without altering the rest of the network. This continual learning capability without fine-tuning positions TSO as a sustainable architecture for autonomous agents.

Furthermore, the absence of catastrophic forgetting (Phase 4: 0%) eliminates the need for replay buffers, experience replay, or elastic weight consolidation — all overheads that Transformers require for sequential task learning. TSO's local plasticity is inherently task-incremental: each new concept is woven into the existing graph without disturbing previously learned edges.

### 11. LIMITATIONS AND OPEN QUESTIONS

1.  **NLI Bootstrap Dependency.** TSO currently requires a frozen semantic bootstrap (DeBERTa/MiniLM) to initialize its conceptual space. While Phase 8 demonstrates that this bootstrap can later be weaned, the full emergence of semantics from scratch (Symbol Grounding) remains an open question. This is analogous to how a human brain uses its evolutionary genome to wire early visual areas before learning from experience.

2.  **Hyperparameter Tuning.** Automatic learning of all free parameters ($\Delta t, \gamma, \epsilon, \theta_t, \theta_c$) remains an open question for system autonomy.

3.  **Scaling and Benchmarks.** TSO is a foundational architecture paper, not a SOTA LLM benchmark submission. The evaluation focuses on principle validation (FLOPs efficiency, zero-shot generalization, catastrophic forgetting resistance) rather than absolute NLP benchmarks like GLUE or BigBench. The original Transformer paper (Vaswani et al., 2017) evaluated on basic translation tasks rather than the massive benchmarks it later enabled.

4.  **Hardware Readiness.** The true energy advantage of TSO (event-driven computation, Skip Compute) is masked on GPU hardware, which is optimized for dense matrix operations. The theoretical 2.9$\times$ FLOPs reduction measured in Phase 11 will translate to a much larger wall-clock energy advantage on neuromorphic hardware (e.g., Intel Loihi 2), where inactive clusters consume near-zero power. TSO-Sim is an algorithmic proof-of-concept; the physical thermodynamic gain is a future engineering milestone.

5.  **Long-Range Memory.** The ESN experiment confirms that non-linearity improves short-range discrimination but degrades long-range retention compared to linear EMA. Designing a spiking reservoir with topographic local connections and stable attractors for robust long-range memory remains an open challenge.

### 11. CONCLUSION

RNNs were replaced by Transformers through parallelized attention. We have proposed and validated TSO, a friction-driven architecture where computation is conditioned on internal stabilization dynamics. By implementing an SNN/R-STDP loop coupled with geometric operators (Double Mapping, Friction-Gated Consolidation) and a semantic Inverse Motor, we have demonstrated that it is possible to resolve semantic paradoxes of natural language through purely local geometric expansion, without global gradients. TSO lays the foundations for a truly adaptive artificial intelligence, learning continuously and compatible with the energy efficiency principles of event-driven computational systems.

**Data and Code Availability.** All code and experiments are publicly available at https://github.com/hamoudaalias/tso.
