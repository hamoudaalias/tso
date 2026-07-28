use ndarray::Array1;
use rand::Rng;
use crate::core::Graph;
use crate::attractor::AttractorField;

// ---------------------------------------------------------------------------
// CONSTRAINT REDIRECTION
//
// Strategy: instead of removing or weakening edges, resolve conflict by
// adjusting node embeddings (vector space) to satisfy constraints
// simultaneously using gradient descent on Φ. This is energy-based learning:
// move node vectors to reduce total Φ without touching edges.
//
// Requirements:
//  1. No edges deleted.
//  2. Node vectors drift in latent space (may break existing concept boundaries).
//  3. Requires differentiable node representation (provided by Array1<f64>).
//  4. Compatible with AttractorField prototypes via post-hoc syncing.
// ---------------------------------------------------------------------------

/// Configuration for the constraint-redirection gradient descent solver.
#[derive(Clone, Debug)]
pub struct RedirectionConfig {
    /// Learning rate for gradient descent on node vectors.
    pub lr: f64,
    /// Maximum number of gradient steps.
    pub max_steps: usize,
    /// If total Φ drops below this threshold, early-stop.
    pub tol: f64,
    /// Re-normalize each node to unit length after every step.
    /// Keeps the geometry on the unit hypersphere (existing Graph convention).
    pub unit_normalize: bool,
    /// Optional momentum coefficient (Nesterov-like). 0.0 = plain SGD.
    pub momentum: f64,
    /// When momentum > 0, prints a progress line each `report_every` steps.
    /// 0 = no progress reporting.
    pub report_every: usize,
    /// Minimum norm threshold for the gradient on one node.
    /// If any node's gradient norm falls below this, a small random
    /// perturbation is added to break deadlock (e.g. exactly opposite
    /// vectors on the sphere produce zero Riemannian gradient).
    pub min_grad_norm: f64,
    /// Magnitude of the random jitter when gradient is near-zero.
    pub jitter_scale: f64,
}

impl Default for RedirectionConfig {
    fn default() -> Self {
        RedirectionConfig {
            lr: 0.1,
            max_steps: 100,
            tol: 1e-4,
            unit_normalize: true,
            momentum: 0.0,
            report_every: 0,
            min_grad_norm: 1e-12,
            jitter_scale: 0.01,
        }
    }
}

/// Outcome of a constraint-redirection solve.
#[derive(Clone, Debug)]
pub struct RedirectionResult {
    /// Number of gradient steps taken.
    pub steps: usize,
    /// Total Φ before the first step.
    pub phi_initial: f64,
    /// Total Φ after the last step.
    pub phi_final: f64,
    /// Infinity-norm of the accumulated gradient vector at the last step.
    pub max_gradient_norm: f64,
    /// Whether the solver terminated by reaching `tol` (true) or
    /// `max_steps` (false).
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Accumulate **Riemannian gradients** on the unit sphere for all edges in tension.
///
/// Because graph nodes are always on the unit sphere (||v|| = 1), the Euclidean
/// gradient of `phi` w.r.t. v_a must be projected onto the tangent space of the
/// sphere at v_a so that gradient descent stays on the manifold.
///
/// For implication edges (weight = 1 or 2):
///   phi  = factor * max(0, gamma  - v_a · v_b)   where factor ∈ {1, 2}
///   ∇_a_phi_euclid  = -factor * v_b
///   ∇_a_phi_riemann = ∇_a_phi_euclid  - (v_a · ∇_a_phi_euclid)  * v_a
///                   = -factor * v_b + factor * (v_a · v_b) * v_a
///
/// For exclusion edges (weight = -1):
///   phi  = max(0, v_a · v_b - epsilon)
///   ∇_a_phi_euclid  = v_b
///   ∇_a_phi_riemann = v_b - (v_a · v_b) * v_a
///
/// This matches the formula used in `Action::Repel` from core.rs, but here the
/// step size is controlled by the gradient magnitude and learning rate rather
/// than a fixed REPEL_STRIDE.
///
/// Returns (gradients, max_gradient_inf_norm).
fn compute_phi_gradient(graph: &Graph) -> (Vec<Array1<f64>>, f64) {
    let n = graph.nodes.len();
    let dim = graph.nodes[0].len();
    let mut grads = vec![Array1::zeros(dim); n];
    let mut max_norm = 0.0;

    for e in &graph.edges {
        let va = &graph.nodes[e.from];
        let vb = &graph.nodes[e.to];
        let dot_ab = va.dot(vb);

        // Determine the Euclidean gradient (before projection) and the scalar factor
        let (active, euclid_scale) = match e.weight {
            1 => {
                if dot_ab < graph.gamma {
                    (true, -1.0)
                } else {
                    (false, 0.0)
                }
            }
            2 => {
                if dot_ab < graph.gamma {
                    (true, -2.0)
                } else {
                    (false, 0.0)
                }
            }
            -1 => {
                if dot_ab > graph.epsilon {
                    (true, 1.0)
                } else {
                    (false, 0.0)
                }
            }
            _ => (false, 0.0),
        };

        if active {
            // Riemannian gradient for v_a:  euclid_scale * (v_b - dot_ab * v_a)
            // Riemannian gradient for v_b:  euclid_scale * (v_a - dot_ab * v_b)
            let direction = vb - va * dot_ab; // v_b - (v_a·v_b)·v_a
            let grad_a = &direction * euclid_scale;
            grads[e.from] = &grads[e.from] + &grad_a;

            let direction_b = va - vb * dot_ab; // v_a - (v_a·v_b)·v_b  (same length, opposite
                                                // sign after swap — but we preserve correct
                                                // projection per node)
            let grad_b = &direction_b * euclid_scale;
            grads[e.to] = &grads[e.to] + &grad_b;

            let na = grad_a.dot(&grad_a).sqrt();
            let nb = grad_b.dot(&grad_b).sqrt();
            if na > max_norm {
                max_norm = na;
            }
            if nb > max_norm {
                max_norm = nb;
            }
        }
    }

    (grads, max_norm)
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// Resolve graph conflicts by gradient descent on node vectors.
///
/// No edges are deleted or modified.  Only `graph.nodes` is mutated.
/// The graph's edges, gamma, and epsilon remain untouched.
///
/// Returns a `RedirectionResult` describing the solve trajectory.
pub fn resolve_by_redirection(graph: &mut Graph, config: &RedirectionConfig) -> RedirectionResult {
    let phi_initial = graph.phi();
    let dim = graph.nodes[0].len();
    let n = graph.nodes.len();

    // Velocity buffer for momentum
    let mut velocity: Vec<Array1<f64>> = vec![Array1::zeros(dim); n];

    let mut phi = phi_initial;
    let mut max_gradient_norm = 0.0;

    for step in 0..config.max_steps {
        // Compute gradient of total Φ w.r.t. each node vector
        let (grad, grad_max) = compute_phi_gradient(graph);
        max_gradient_norm = grad_max;

        // Apply gradient step with optional momentum and jitter
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let mut g = grad[i].clone();
            let gn = g.dot(&g).sqrt();

            // Break deadlock: if Riemannian gradient is near-zero (can happen
            // for exactly opposite vectors) add a small random perturbation.
            if gn < config.min_grad_norm {
                let mut jitter = Array1::zeros(dim);
                for j in 0..dim {
                    jitter[j] = rng.gen_range(-config.jitter_scale..config.jitter_scale);
                }
                g = g + &jitter;
            }

            if config.momentum > 0.0 {
                velocity[i] = &velocity[i] * config.momentum - &g * config.lr;
                graph.nodes[i] = &graph.nodes[i] + &velocity[i];
            } else {
                graph.nodes[i] = &graph.nodes[i] - &(&g * config.lr);
            }

            // Re-normalise to unit sphere
            if config.unit_normalize {
                let norm = graph.nodes[i].dot(&graph.nodes[i]).sqrt().max(1e-12);
                graph.nodes[i] = &graph.nodes[i] / norm;
            }
        }

        // Recompute total Φ
        phi = graph.phi();

        if config.report_every > 0 && (step + 1) % config.report_every == 0 {
            eprintln!(
                "[constraint_redirection] step {:4}  Φ = {:.6}  |∇|_∞ = {:.6}",
                step + 1,
                phi,
                max_gradient_norm,
            );
        }

        if phi < config.tol || (phi_initial - phi).abs() < 1e-12 && step > 0 {
            // Converged: Φ is below tolerance, or stalled
            return RedirectionResult {
                steps: step + 1,
                phi_initial,
                phi_final: phi,
                max_gradient_norm,
                converged: true,
            };
        }
    }

    RedirectionResult {
        steps: config.max_steps,
        phi_initial,
        phi_final: phi,
        max_gradient_norm,
        converged: phi < config.tol,
    }
}

// ---------------------------------------------------------------------------
// Prototype syncing
// ---------------------------------------------------------------------------

/// Copy graph node vectors back to the attractor field's first prototype of
/// each corresponding concept.
///
/// Graph node `i` (0..N) is synced to `attractor.prototypes[i][0]` — the
/// primary prototype of concept `i`.  Side-effect prototypes (k>0) are left
/// unchanged because they represent alternative cluster centroids that may
/// still be valid even after the primary drifts.
///
/// If the graph has more nodes than attractor classes, only the first
/// `attractor.n_classes()` nodes are synced (the extras are graph-only
/// scratch nodes without a concept mapping).
///
/// This is the bridge that keeps (requirement 4): compatibility with
/// AttractorField prototypes.  Call it after `resolve_by_redirection()`.
pub fn sync_prototypes_from_graph(graph: &Graph, attractor: &mut AttractorField) {
    let n_sync = attractor.n_classes().min(graph.nodes.len());
    for i in 0..n_sync {
        if !attractor.prototypes[i].is_empty() {
            attractor.prototypes[i][0] = graph.nodes[i].clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Graph;

    #[test]
    fn test_no_phi_no_change() {
        let mut graph = Graph::with_params(0.7, 0.1);
        // Two nodes at 90°: dot ≈ 0.  No edges → no phi.
        let v1 = ndarray::arr1(&[1.0, 0.0]);
        let v2 = ndarray::arr1(&[0.0, 1.0]);
        graph.add_node(v1.clone());
        graph.add_node(v2.clone());
        // No edges — phi should be 0
        assert!(graph.phi() < 1e-12);

        let config = RedirectionConfig::default();
        let result = resolve_by_redirection(&mut graph, &config);
        assert!(result.phi_initial < 1e-12);
        assert!(result.phi_final < 1e-12);
        assert!(result.converged);
    }

    #[test]
    fn test_implication_conflict_resolved() {
        let mut graph = Graph::with_params(0.7, 0.1);
        // Two nodes that are almost opposite (dot ≈ -0.96) → conflict for implication
        // because gamma=0.7, dot < 0.7
        let v1 = ndarray::arr1(&[0.9, 0.43589]);
        let v2 = ndarray::arr1(&[-0.9, 0.43589]); // dot ≈ -0.62 < 0.7
        let id1 = graph.add_node(v1.clone());
        let id2 = graph.add_node(v2.clone());
        graph.add_edge(id1, id2, 1); // implication

        let phi_before = graph.phi();
        assert!(phi_before > 0.0);

        let config = RedirectionConfig {
            lr: 0.1,
            max_steps: 200,
            tol: 1e-4,
            unit_normalize: true,
            momentum: 0.0,
            report_every: 0,
            min_grad_norm: 1e-12,
            jitter_scale: 0.01,
        };
        let result = resolve_by_redirection(&mut graph, &config);

        let phi_after = graph.phi();
        // Must not delete edges
        assert_eq!(graph.edges.len(), 1);
        // Phi should drop significantly
        assert!(phi_after < phi_before - 0.1);
        // Nodes must still be unit vectors
        for n in &graph.nodes {
            let norm = n.dot(n).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
        // Result must report initial/final
        assert!((result.phi_initial - phi_before).abs() < 1e-9);
    }

    #[test]
    fn test_exclusion_conflict_resolved() {
        let mut graph = Graph::with_params(0.7, 0.1);
        // Two nodes with dot ≈ 0.96 > 0.1 → conflict for exclusion
        let v1 = ndarray::arr1(&[1.0, 0.0]);
        let v2 = ndarray::arr1(&[0.96, 0.28]);
        let id1 = graph.add_node(v1.clone());
        let id2 = graph.add_node(v2.clone());
        graph.add_edge(id1, id2, -1); // exclusion

        let phi_before = graph.phi();
        assert!(phi_before > 0.0);

        let config = RedirectionConfig {
            lr: 0.1,
            max_steps: 200,
            tol: 1e-4,
            unit_normalize: true,
            momentum: 0.0,
            report_every: 0,
            min_grad_norm: 1e-12,
            jitter_scale: 0.01,
        };
        let result = resolve_by_redirection(&mut graph, &config);

        let phi_after = graph.phi();
        assert_eq!(graph.edges.len(), 1); // no edges deleted
        assert!(phi_after < phi_before - 0.1);
        for n in &graph.nodes {
            let norm = n.dot(n).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
        assert!((result.phi_initial - phi_before).abs() < 1e-9);
    }

    #[test]
    fn test_momentum_terminates() {
        let mut graph = Graph::with_params(0.7, 0.1);
        let v1 = ndarray::arr1(&[0.9, 0.43589, 0.0]);
        let v2 = ndarray::arr1(&[-0.9, 0.43589, 0.0]);
        let v3 = ndarray::arr1(&[0.0, 1.0, 0.0]);
        let id1 = graph.add_node(v1);
        let id2 = graph.add_node(v2);
        graph.add_edge(id1, id2, 1);
        graph.add_node(v3); // extra free node, no edges

        let config = RedirectionConfig {
            lr: 0.15,
            max_steps: 150,
            tol: 1e-4,
            unit_normalize: true,
            momentum: 0.7,
            report_every: 0,
            min_grad_norm: 1e-12,
            jitter_scale: 0.01,
        };
        let result = resolve_by_redirection(&mut graph, &config);

        assert_eq!(graph.edges.len(), 1);
        assert!(result.phi_final < result.phi_initial - 0.1);
        assert!(result.max_gradient_norm >= 0.0);
    }

    #[test]
    fn test_sync_prototypes() {
        use crate::attractor::AttractorField;
        let mut graph = Graph::with_params(0.7, 0.1);
        // 2 concepts, each a 2D unit vector
        let v1 = ndarray::arr1(&[1.0, 0.0]);
        let v2 = ndarray::arr1(&[0.0, 1.0]);
        graph.add_node(v1.clone());
        graph.add_node(v2.clone());
        graph.add_edge(0, 1, 1);

        let mut attractor = AttractorField::new(2, 2, 2, 0.01); // 2 classes, 2 prototypes each
        // Set first prototype of each class
        attractor.prototypes[0][0] = ndarray::arr1(&[0.0, 0.0]); // distinct from graph nodes
        attractor.prototypes[1][0] = ndarray::arr1(&[0.0, 0.0]);

        sync_prototypes_from_graph(&graph, &mut attractor);

        // First prototypes should now match graph nodes
        assert!((&attractor.prototypes[0][0] - &v1).dot(&(&attractor.prototypes[0][0] - &v1)).sqrt() < 1e-9);
        assert!((&attractor.prototypes[1][0] - &v2).dot(&(&attractor.prototypes[1][0] - &v2)).sqrt() < 1e-9);
        // Second prototypes should remain as original
        assert!((&attractor.prototypes[0][1]).dot(&attractor.prototypes[0][1]).sqrt() > 0.0);
    }

    #[test]
    fn test_no_edge_deletion() {
        let mut graph = Graph::with_params(0.7, 0.1);
        // Multiple edges forming a small clique with conflicts
        let v1 = ndarray::arr1(&[1.0, 0.0, 0.0]);
        let v2 = ndarray::arr1(&[-0.8, 0.6, 0.0]); // dot ≈ -0.8 < 0.7 → implication conflict
        let v3 = ndarray::arr1(&[0.0, 1.0, 0.0]);
        let id1 = graph.add_node(v1);
        let id2 = graph.add_node(v2);
        let id3 = graph.add_node(v3);
        graph.add_edge(id1, id2, 1);  // implication: dot ≈ -0.8 < gamma=0.7 → conflict
        graph.add_edge(id2, id3, -1); // exclusion: dot=0.6 > epsilon=0.1 → conflict
        graph.add_edge(id1, id3, 1);  // implication: dot=0 < gamma=0.7 → conflict

        let before = graph.edges.len();
        let phi_before = graph.phi();

        let config = RedirectionConfig {
            lr: 0.1,
            max_steps: 300,
            tol: 1e-4,
            unit_normalize: true,
            momentum: 0.8,
            report_every: 0,
            min_grad_norm: 1e-12,
            jitter_scale: 0.01,
        };
        let result = resolve_by_redirection(&mut graph, &config);

        assert_eq!(graph.edges.len(), before, "Edge count must not change");
        assert!(result.phi_final < phi_before, "Phi must decrease");
    }
}
