use ndarray::Array1;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

pub type NodeId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConflictType {
    Exclusion,
    Implication,
}

impl ConflictType {
    pub fn from_weight(weight: i8) -> Self {
        match weight {
            -1 => ConflictType::Exclusion,
            _  => ConflictType::Implication,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            ConflictType::Exclusion => 0,
            ConflictType::Implication => 1,
        }
    }
}

const REPEL_STRIDE: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Invert(NodeId),
    Align(NodeId, NodeId),
    Repel(NodeId, NodeId),
}

impl Action {
    pub fn index(&self) -> usize {
        match self {
            Action::Invert(_) => 0,
            Action::Align(_, _) => 1,
            Action::Repel(_, _) => 2,
        }
    }

    pub fn apply_to_graph(&self, graph: &mut Graph) {
        match self {
            Action::Invert(id) => {
                graph.nodes[*id].mapv_inplace(|x| -x);
            }
            Action::Align(a, b) => {
                let sum = &graph.nodes[*a] + &graph.nodes[*b];
                let norm = sum.dot(&sum).sqrt();
                if norm > 1e-12 {
                    graph.nodes[*a] = &sum / norm;
                    graph.nodes[*b] = &sum / norm;
                } else {
                    let unit = &graph.nodes[*a] / graph.nodes[*a].dot(&graph.nodes[*a]).sqrt().max(1e-12);
                    graph.nodes[*a] = unit.clone();
                    graph.nodes[*b] = unit;
                }
            }
            Action::Repel(a, b) => {
                let va = graph.nodes[*a].clone();
                let vb = graph.nodes[*b].clone();
                let dot_ab = va.dot(&vb);
                let grad_a = -&vb + dot_ab * &va;
                let na = grad_a.dot(&grad_a).sqrt();
                if na > 1e-12 {
                    let moved = &va + &(grad_a / na * REPEL_STRIDE);
                    graph.nodes[*a] = moved.clone() / moved.dot(&moved).sqrt().max(1e-12);
                } else {
                    graph.nodes[*a] = -&va;
                }
                if na < 1e-12 && (va.dot(&vb).abs() - 1.0).abs() < 1e-9 {
                    graph.nodes[*b] = vb.clone();
                } else {
                    let grad_b = -&va + dot_ab * &vb;
                    let nb = grad_b.dot(&grad_b).sqrt();
                    if nb > 1e-12 {
                        let moved = &vb + &(grad_b / nb * REPEL_STRIDE);
                        graph.nodes[*b] = moved.clone() / moved.dot(&moved).sqrt().max(1e-12);
                    } else {
                        graph.nodes[*b] = vb.clone();
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: i8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Array1<f64>>,
    pub edges: Vec<Edge>,
    edge_map: HashMap<(NodeId, NodeId), i8>,
    adj: Vec<Vec<usize>>,
    pub gamma: f64,
    pub epsilon: f64,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            adj: Vec::new(),
            gamma: 0.7,
            epsilon: 0.0,
        }
    }

    pub fn with_params(gamma: f64, epsilon: f64) -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_map: HashMap::new(),
            adj: Vec::new(),
            gamma,
            epsilon,
        }
    }

    pub fn add_node(&mut self, z: Array1<f64>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(z);
        self.adj.push(Vec::new());
        id
    }

    pub fn find_similar_node(&self, z: &Array1<f64>, tol: f64) -> Option<NodeId> {
        for (i, n) in self.nodes.iter().enumerate() {
            let d = n.dot(z);
            let na = n.dot(n).sqrt().max(1e-12);
            let nb = z.dot(z).sqrt().max(1e-12);
            let sim = d / (na * nb);
            if sim > tol {
                return Some(i);
            }
        }
        None
    }

    pub fn add_transition(&mut self, from: &Array1<f64>, to: &Array1<f64>, reward: f64) -> (NodeId, NodeId) {
        let from_id = self.find_similar_node(from, 0.95)
            .unwrap_or_else(|| self.add_node(from.clone()));
        let to_id = self.find_similar_node(to, 0.95)
            .unwrap_or_else(|| self.add_node(to.clone()));
        let weight = if reward > 0.5 { 2 }
                     else if reward < -0.1 { -1 }
                     else { 1 };
        self.add_edge(from_id, to_id, weight);
        (from_id, to_id)
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: i8) {
        if self.edge_map.contains_key(&(from, to)) {
            return;
        }
        let idx = self.edges.len();
        self.edges.push(Edge { from, to, weight });
        self.adj[from].push(idx);
        self.adj[to].push(idx);
        self.edge_map.insert((from, to), weight);
        self.edge_map.insert((to, from), weight);
    }

    pub fn edge_weight(&self, a: NodeId, b: NodeId) -> Option<i8> {
        self.edge_map.get(&(a, b)).copied()
    }

    pub fn phi(&self) -> f64 {
        let mut total = 0.0;
        for e in &self.edges {
            total += self.edge_phi(e);
        }
        total
    }

    pub fn edge_phi(&self, e: &Edge) -> f64 {
        let dot = self.nodes[e.from].dot(&self.nodes[e.to]);
        match e.weight {
            1 => (self.gamma - dot).max(0.0),
            2 => (self.gamma - dot).max(0.0) * 2.0,
            -1 => (dot - self.epsilon).max(0.0),
            _ => 0.0,
        }
    }

    pub fn sequential_phi(&self, lif_state: &Array1<f64>, word_id: NodeId, negate: bool) -> f64 {
        let e = if negate { -&self.nodes[word_id] } else { self.nodes[word_id].clone() };
        let mut total = 0.0;
        for edge in &self.edges {
            let other_id = if edge.from == word_id {
                edge.to
            } else if edge.to == word_id {
                edge.from
            } else {
                continue;
            };
            let activation = lif_state.dot(&self.nodes[other_id]).max(0.0);
            if activation > 1e-12 {
                let dot = e.dot(&self.nodes[other_id]);
                let phi = match edge.weight {
                    1 => (self.gamma - dot).max(0.0),
                    -1 => (dot - self.epsilon).max(0.0),
                    _ => 0.0,
                };
                total += activation * phi;
            }
        }
        total
    }

    pub fn neighbourhood(&self, seeds: &[NodeId], depth: usize) -> Vec<NodeId> {
        let mut set: HashSet<NodeId> = seeds.iter().cloned().collect();
        let mut frontier: Vec<NodeId> = seeds.to_vec();
        for _ in 0..depth {
            let mut next: Vec<NodeId> = Vec::new();
            for &f in &frontier {
                for &ei in &self.adj[f] {
                    let e = &self.edges[ei];
                    let other = if e.from == f { e.to } else { e.from };
                    if !set.contains(&other) {
                        set.insert(other);
                        next.push(other);
                    }
                }
            }
            frontier = next;
        }
        set.into_iter().collect()
    }

    /// Remove edges whose phi contribution is below `min_phi`.
    /// Returns the number of edges removed.
    /// Remove an edge by its endpoints. Returns the phi contribution that was removed.
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) -> f64 {
        let pos = self.edges.iter().position(|e| (e.from == from && e.to == to) || (e.from == to && e.to == from));
        match pos {
            Some(idx) => {
                let e = &self.edges[idx];
                let saved = self.edge_phi(e);
                self.edges.swap_remove(idx);
                self.edge_map.remove(&(from, to));
                self.edge_map.remove(&(to, from));
                self.adj[from].retain(|&i| i != idx);
                self.adj[to].retain(|&i| i != idx);
                saved
            }
            None => 0.0,
        }
    }

    /// Flag an edge as resolved: remove it and return the amount of Φ eliminated.
    /// Φ drops immediately when a conflicting edge is "flagged" (removed).
    pub fn flag_edge(&mut self, from: NodeId, to: NodeId) -> f64 {
        self.remove_edge(from, to)
    }

    /// Décroissance exponentielle du poids : multiplie le poids par `factor`.
    /// Les poids i8 sont traités comme f64 le temps de la décroissance,
    /// puis arrondis à l'entier le plus proche. Si |weight| < 0.5, l'arête est supprimée.
    /// Retourne le Φ éliminé (0 si l'arête survit, >0 si supprimée).
    pub fn exp_decay_edge_weight(&mut self, from: NodeId, to: NodeId, factor: f64) -> f64 {
        let pos = self.edges.iter().position(|e| (e.from == from && e.to == to) || (e.from == to && e.to == from));
        if let Some(idx) = pos {
            let saved = self.edge_phi(&self.edges[idx]);
            let e = &mut self.edges[idx];
            let old = e.weight as f64;
            let decayed = old * factor;
            if decayed.abs() < 0.5 {
                self.remove_edge(from, to);
                saved
            } else {
                e.weight = decayed.round() as i8;
                self.edge_map.insert((from, to), e.weight);
                self.edge_map.insert((to, from), e.weight);
                0.0
            }
        } else { 0.0 }
    }

    /// Décroissance graduelle du poids d'une arête (inhibition latérale).
    /// Au lieu de supprimer instantanément, on réduit la valeur absolue du poids
    /// de `decay` à chaque violation. Si le poids passe à 0, l'arête est supprimée.
    /// Les poids sont i8 : +1/+2 (implication), -1 (exclusion).
    /// Retourne le Φ éliminé (0 si l'arête survit, >0 si supprimée).
    pub fn decay_edge_weight(&mut self, from: NodeId, to: NodeId, decay: i8) -> f64 {
        let pos = self.edges.iter().position(|e| (e.from == from && e.to == to) || (e.from == to && e.to == from));
        if let Some(idx) = pos {
            // Capture phi contribution *before* we mutate the weight
            let saved = self.edge_phi(&self.edges[idx]);
            let e = &mut self.edges[idx];
            let old_weight = e.weight;
            // Réduit vers 0 par pas de `decay`
            if old_weight > 0 {
                e.weight = (old_weight - decay).max(0);
            } else if old_weight < 0 {
                e.weight = (old_weight + decay).min(0);
            }
            // Sync edge_map so edge_weight() returns the updated value
            if e.weight != 0 {
                self.edge_map.insert((from, to), e.weight);
                self.edge_map.insert((to, from), e.weight);
                0.0
            } else {
                self.remove_edge(from, to);
                saved
            }
        } else {
            0.0
        }
    }

    /// Bulk-prune exclusion edges whose phi contribution is below `min_phi`.
    /// Returns (exclusion_removed, implication_removed, total_phi_saved).
    pub fn prune_exclusion_edges(&mut self, min_phi: f64) -> (usize, usize, f64) {
        // Compute phi for each edge before draining
        let phis: Vec<f64> = self.edges.iter().map(|e| self.edge_phi(e)).collect();
        let mut kept = Vec::new();
        let mut excl_removed = 0usize;
        let mut impl_removed = 0usize;
        let mut phi_saved = 0.0;
        for (e, phi) in self.edges.drain(..).zip(phis) {
            if phi < min_phi {
                phi_saved += phi;
                match e.weight {
                    -1 => excl_removed += 1,
                    _ => impl_removed += 1,
                }
            } else {
                kept.push(e);
            }
        }
        let removed = excl_removed + impl_removed;
        self.edges = kept;
        if removed > 0 {
            self.edge_map.clear();
            self.adj = vec![Vec::new(); self.nodes.len()];
            for (idx, e) in self.edges.iter().enumerate() {
                self.edge_map.insert((e.from, e.to), e.weight);
                self.edge_map.insert((e.to, e.from), e.weight);
                self.adj[e.from].push(idx);
                self.adj[e.to].push(idx);
            }
        }
        (excl_removed, impl_removed, phi_saved)
    }

    /// Inject many random exclusion edges between random node pairs.
    /// Used to stress-test the resolution and pruning machinery.
    /// Returns the number of edges added.
    pub fn inject_exclusion_edges(&mut self, count: usize) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut added = 0usize;
        for _ in 0..count {
            if self.nodes.len() < 2 { break; }
            let a = rng.gen_range(0..self.nodes.len());
            let b = rng.gen_range(0..self.nodes.len());
            if a == b { continue; }
            if self.edge_map.contains_key(&(a, b)) { continue; }
            self.add_edge(a, b, -1);
            added += 1;
        }
        added
    }

    pub fn remove_low_phi_edges(&mut self, min_phi: f64) -> usize {
        let before = self.edges.len();
        let keep: Vec<bool> = self.edges.iter().map(|e| self.edge_phi(e) >= min_phi).collect();
        self.edges = self.edges.drain(..).enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, e)| e)
            .collect();
        let removed = before - self.edges.len();
        if removed > 0 {
            self.edge_map.clear();
            self.adj = vec![Vec::new(); self.nodes.len()];
            for (idx, e) in self.edges.iter().enumerate() {
                self.edge_map.insert((e.from, e.to), e.weight);
                self.edge_map.insert((e.to, e.from), e.weight);
                self.adj[e.from].push(idx);
                self.adj[e.to].push(idx);
            }
        }
        removed
    }

    /// Remove all edges while keeping nodes intact.
    /// Used by concept pruning to rebuild edges after reindexing nodes.
    pub fn clear_edges(&mut self) {
        self.edges.clear();
        self.edge_map.clear();
        self.adj = vec![Vec::new(); self.nodes.len()];
    }

    /// Metabolic cost per tick.
    /// Proportional to graph size — more edges means more computation
    /// for phi() and resolve(). Each edge and node incurs a small cost.
    pub fn compute_cost(&self) -> f64 {
        self.edges.len() as f64 * 0.1 + self.nodes.len() as f64 * 0.05
    }

    pub fn local_edge_indices(&self, node_set: &[NodeId]) -> Vec<usize> {
        let set: HashSet<NodeId> = node_set.iter().cloned().collect();
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for &n in node_set {
            for &ei in &self.adj[n] {
                let e = &self.edges[ei];
                if set.contains(&e.from) && set.contains(&e.to) && seen.insert(ei) {
                    result.push(ei);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Critic
// ---------------------------------------------------------------------------
pub struct Critic;

pub const CRITIC_DEPTH: usize = 1;

impl Critic {
    pub fn evaluate(graph: &Graph, conflict_edge_idx: usize, action: &Action) -> f64 {
        let e = &graph.edges[conflict_edge_idx];
        let a = e.from;
        let b = e.to;

        let mut seen = HashSet::new();
        let mut incident = Vec::new();
        for &ei in &graph.adj[a] {
            if seen.insert(ei) { incident.push(ei); }
        }
        for &ei in &graph.adj[b] {
            if seen.insert(ei) { incident.push(ei); }
        }

        let phi_before: f64 = incident.iter()
            .map(|&idx| graph.edge_phi(&graph.edges[idx]))
            .sum();

        let phi_after: f64 = match action {
            Action::Invert(id) => {
                let mut inv = if *id == a { Some(-&graph.nodes[a]) } else { None };
                if *id == b { inv = Some(-&graph.nodes[b]); }
                let inv = inv.unwrap();
                incident.iter().map(|&idx| {
                    let ee = &graph.edges[idx];
                    let dot = if ee.from == *id { inv.dot(&graph.nodes[ee.to]) }
                              else if ee.to == *id { graph.nodes[ee.from].dot(&inv) }
                              else { graph.nodes[ee.from].dot(&graph.nodes[ee.to]) };
                    match ee.weight { 1 => (graph.gamma - dot).max(0.0), 2 => (graph.gamma - dot).max(0.0) * 2.0, -1 => (dot - graph.epsilon).max(0.0), _ => 0.0 }
                }).sum()
            }
            Action::Align(a, b) => {
                let u = &graph.nodes[*a];
                let v = &graph.nodes[*b];
                let sum = u + v;
                let norm = sum.dot(&sum).sqrt();
                let (nu, nv): (Array1<f64>, Array1<f64>) = if norm > 1e-12 {
                    (&sum / norm, &sum / norm)
                } else {
                    let unit = u / u.dot(u).sqrt().max(1e-12);
                    (unit.clone(), unit)
                };
                incident.iter().map(|&idx| {
                    let ee = &graph.edges[idx];
                    let dot = if ee.from == *a && ee.to == *b { nu.dot(&nv) }
                              else if ee.from == *a { nu.dot(&graph.nodes[ee.to]) }
                              else if ee.to == *a { graph.nodes[ee.from].dot(&nu) }
                              else if ee.from == *b { nv.dot(&graph.nodes[ee.to]) }
                              else if ee.to == *b { graph.nodes[ee.from].dot(&nv) }
                              else { graph.nodes[ee.from].dot(&graph.nodes[ee.to]) };
                    match ee.weight { 1 => (graph.gamma - dot).max(0.0), 2 => (graph.gamma - dot).max(0.0) * 2.0, -1 => (dot - graph.epsilon).max(0.0), _ => 0.0 }
                }).sum()
            }
            Action::Repel(a, b) => {
                let u = &graph.nodes[*a];
                let v = &graph.nodes[*b];
                let dot_uv = u.dot(v);
                let grad_a = &(-v + dot_uv * u);
                let na = grad_a.dot(grad_a).sqrt();
                let nu = if na > 1e-12 {
                    let moved = u + &(grad_a / na * REPEL_STRIDE);
                    moved.clone() / moved.dot(&moved).sqrt().max(1e-12)
                } else {
                    -u / u.dot(u).sqrt().max(1e-12)
                };
                let grad_b = &(-u + dot_uv * v);
                let nb = grad_b.dot(grad_b).sqrt();
                let nv = if nb > 1e-12 {
                    let moved = v + &(grad_b / nb * REPEL_STRIDE);
                    moved.clone() / moved.dot(&moved).sqrt().max(1e-12)
                } else {
                    -v / v.dot(v).sqrt().max(1e-12)
                };
                incident.iter().map(|&idx| {
                    let ee = &graph.edges[idx];
                    let dot = if ee.from == *a && ee.to == *b { nu.dot(&nv) }
                              else if ee.from == *a { nu.dot(&graph.nodes[ee.to]) }
                              else if ee.to == *a { graph.nodes[ee.from].dot(&nu) }
                              else if ee.from == *b { nv.dot(&graph.nodes[ee.to]) }
                              else if ee.to == *b { graph.nodes[ee.from].dot(&nv) }
                              else { graph.nodes[ee.from].dot(&graph.nodes[ee.to]) };
                    match ee.weight { 1 => (graph.gamma - dot).max(0.0), 2 => (graph.gamma - dot).max(0.0) * 2.0, -1 => (dot - graph.epsilon).max(0.0), _ => 0.0 }
                }).sum()
            }
        };

        phi_after - phi_before
    }

    pub fn evaluate_all(graph: &Graph, conflict_edge_idx: usize, a: NodeId, b: NodeId) -> ([f64; 3], usize) {
        let actions = [Action::Invert(b), Action::Align(a, b), Action::Repel(a, b)];
        let mut deltas = [0.0; 3];
        let mut best_idx = 0;
        for (i, act) in actions.iter().enumerate() {
            deltas[i] = Critic::evaluate(graph, conflict_edge_idx, act);
            if deltas[i] < deltas[best_idx] {
                best_idx = i;
            }
        }
        (deltas, best_idx)
    }
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------
pub struct Actor {
    q: [[f64; 3]; 2],
    epsilon: f64,
    eta: f64,
}

impl Actor {
    pub fn new(epsilon: f64, eta: f64) -> Self {
        Actor { q: [[0.0; 3]; 2], epsilon, eta }
    }

    pub fn reinforce(&mut self, conflict: ConflictType, action: &Action, reward: f64) {
        self.q[conflict.index()][action.index()] += self.eta * reward;
    }

    pub fn decay_epsilon(&mut self, factor: f64) {
        self.epsilon = (self.epsilon * factor).max(0.05);
    }

    pub fn propose(&self, conflict: ConflictType) -> usize {
        if rand::random::<f64>() < self.epsilon {
            rand::random::<usize>() % 3
        } else {
            let q_values = &self.q[conflict.index()];
            let mut best = 0;
            for i in 1..3 { if q_values[i] > q_values[best] { best = i; } }
            best
        }
    }
}

fn action_from_idx(idx: usize, a: NodeId, b: NodeId) -> Action {
    match idx {
        0 => Action::Invert(b),
        1 => Action::Align(a, b),
        _ => Action::Repel(a, b),
    }
}

// ---------------------------------------------------------------------------
// Resolve
// ---------------------------------------------------------------------------
pub struct ResolveResult {
    pub iterations: usize,
    pub phi_trace: Vec<f64>,
    pub actions_taken: usize,
    pub converged: bool,
    pub oscillation_breaks: usize,
}

fn select_independent_edges(violated: &[(usize, f64)], edges: &[Edge]) -> Vec<usize> {
    let mut busy: HashSet<usize> = HashSet::new();
    let mut batch = Vec::new();
    for &(idx, _) in violated {
        let e = &edges[idx];
        if !busy.contains(&e.from) && !busy.contains(&e.to) {
            batch.push(idx);
            busy.insert(e.from);
            busy.insert(e.to);
        }
    }
    batch
}

const BATCH_LIMIT: usize = 500;

pub fn resolve(graph: &mut Graph, max_iter: usize, tol: f64) -> ResolveResult {
    resolve_with_anneal(graph, max_iter, tol, 0.0)
}

fn boltzmann_select(deltas: &[f64; 3], temperature: f64) -> usize {
    if temperature <= 0.0 { return 0; }
    let min_d = deltas[0].min(deltas[1]).min(deltas[2]);
    let weights: Vec<f64> = deltas.iter()
        .map(|d| (-(d - min_d) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 { return 0; }
    let r = rand::random::<f64>() * total;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if r < acc { return i; }
    }
    2
}

fn select_best_or_actor(deltas: &[f64; 3], best_idx: usize, actor: &Actor, conflict: ConflictType) -> (usize, bool) {
    let action_idx = actor.propose(conflict);
    if deltas[action_idx] < 0.0 {
        (action_idx, true)
    } else {
        (best_idx, false)
    }
}

pub fn resolve_with_anneal(graph: &mut Graph, max_iter: usize, tol: f64, mut temperature: f64) -> ResolveResult {
    let mut actor = Actor::new(0.5, 0.15);
    let mut phi_trace = Vec::new();
    let mut actions_taken = 0;

    let mut best_phi = graph.phi();
    let mut best_nodes = graph.nodes.clone();
    let mut stall_count = 0;
    const STALL_LIMIT: usize = 20;
    let mut osc_count = 0usize;

    for iter in 0..max_iter {
        let phi = graph.phi();
        phi_trace.push(phi);

        if phi < best_phi - 1e-9 {
            best_phi = phi;
            best_nodes = graph.nodes.clone();
            stall_count = 0;
        } else {
            stall_count += 1;
        }

        // Detect oscillation: if phi alternates direction ≥3 times in last 6 iters
        // while stalled, force greedy mode to break the Repel↔Align cycle.
        if temperature > 0.0 && stall_count >= 3 && phi_trace.len() >= 6 {
            let window = &phi_trace[phi_trace.len()-6..];
            let mut sign_flips = 0;
            for i in 2..window.len() {
                let d1 = window[i-1] - window[i-2];
                let d2 = window[i] - window[i-1];
                if d1 * d2 < 0.0 { sign_flips += 1; }
            }
            if sign_flips >= 3 {
                temperature = 0.0;
                osc_count += 1;
            }
        }

        if phi < tol || stall_count >= STALL_LIMIT {
            graph.nodes = best_nodes;
            let final_phi = graph.phi();
            phi_trace.push(final_phi);
            return ResolveResult {
                iterations: iter,
                phi_trace,
                actions_taken,
                converged: true,
                oscillation_breaks: osc_count,
            };
        }

        let mut violated: Vec<(usize, f64)> = graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, graph.edge_phi(e)))
            .filter(|(_, p)| *p > tol)
            .collect();
        violated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        violated.truncate(BATCH_LIMIT);

        let batch = select_independent_edges(&violated, &graph.edges);

        let mut applied: Vec<(usize, Action, f64, ConflictType, bool)> = Vec::new();
        for &edge_idx in &batch {
            let e = &graph.edges[edge_idx];
            let (a, b) = (e.from, e.to);
            let conflict = ConflictType::from_weight(e.weight);

            let (deltas, best_idx) = Critic::evaluate_all(graph, edge_idx, a, b);

            let (select_idx, was_actor) = if temperature > 0.0 {
                (boltzmann_select(&deltas, temperature), false)
            } else {
                select_best_or_actor(&deltas, best_idx, &actor, conflict)
            };

            let selected_action = action_from_idx(select_idx, a, b);
            applied.push((edge_idx, selected_action, deltas[select_idx], conflict, was_actor));
        }

        applied.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let mut any_applied = false;
        for (_edge_idx, action, _delta, conflict, was_actor) in &applied {
            action.apply_to_graph(graph);
            if temperature <= 0.0 {
                if *was_actor {
                    actor.reinforce(*conflict, action, 1.0);
                } else {
                    actor.reinforce(*conflict, action, -0.3);
                }
            }
            actions_taken += 1;
            any_applied = true;
        }

        if !any_applied {
            if violated.is_empty() { break; }
            let (edge_idx, _) = violated[0];
            let e = &graph.edges[edge_idx];
            let (a, b) = (e.from, e.to);
            let (deltas, _) = Critic::evaluate_all(graph, edge_idx, a, b);
            let select_idx = if temperature > 0.0 {
                boltzmann_select(&deltas, temperature)
            } else {
                0
            };
            let best_action = action_from_idx(select_idx, a, b);
            best_action.apply_to_graph(graph);
            actions_taken += 1;
        }

        temperature *= 0.85;
        if temperature <= 0.0 { temperature = 0.0; }
        actor.decay_epsilon(0.997);
    }

    graph.nodes = best_nodes;
    phi_trace.push(graph.phi());
    ResolveResult {
        iterations: max_iter,
        phi_trace,
        actions_taken,
        converged: true,
        oscillation_breaks: osc_count,
    }
}

/// Parallel resolution using scoped threads.
/// Independent edge batches are processed concurrently on node copies,
/// then the best result is merged back. Scales with available cores.
/// Falls back to sequential resolve if num_threads <= 1.
pub fn resolve_parallel(graph: &mut Graph, max_iter: usize, tol: f64, mut temperature: f64, num_threads: usize) -> ResolveResult {
    if num_threads <= 1 {
        return resolve_with_anneal(graph, max_iter, tol, temperature);
    }

    let mut actor = Actor::new(0.5, 0.15);
    let mut phi_trace = Vec::new();
    let mut actions_taken = 0;
    let mut best_phi = graph.phi();
    let mut best_nodes = graph.nodes.clone();
    let mut stall_count = 0;
    let mut osc_count = 0usize;

    for iter in 0..max_iter {
        let phi = graph.phi();
        phi_trace.push(phi);

        if phi < best_phi - 1e-9 {
            best_phi = phi;
            best_nodes = graph.nodes.clone();
            stall_count = 0;
        } else {
            stall_count += 1;
        }

        if temperature > 0.0 && stall_count >= 3 && phi_trace.len() >= 6 {
            let window = &phi_trace[phi_trace.len()-6..];
            let mut sign_flips = 0;
            for i in 2..window.len() {
                let d1 = window[i-1] - window[i-2];
                let d2 = window[i] - window[i-1];
                if d1 * d2 < 0.0 { sign_flips += 1; }
            }
            if sign_flips >= 3 {
                temperature = 0.0;
                osc_count += 1;
            }
        }

        if phi < tol || stall_count >= 20 {
            graph.nodes = best_nodes;
            phi_trace.push(graph.phi());
            return ResolveResult { iterations: iter, phi_trace, actions_taken, converged: true, oscillation_breaks: osc_count };
        }

        let violated: Vec<(usize, f64)> = graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, graph.edge_phi(e)))
            .filter(|(_, p)| *p > tol)
            .collect();
        if violated.is_empty() { break; }

        // Split violated edges into batches for parallel processing
        let batch_size = (violated.len() / num_threads).max(1);
        let chunks: Vec<&[(usize, f64)]> = violated.chunks(batch_size).collect();

        let results = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in &chunks {
                if chunk.is_empty() { continue; }
                let nodes_snapshot = graph.nodes.clone();
                let n_nodes = nodes_snapshot.len();
                let edges_snapshot = graph.edges.clone();
                let gamma = graph.gamma;
                let epsilon = graph.epsilon;
                let temp = temperature;
                let handle = s.spawn(move || {
                    let mut local_graph = Graph { nodes: nodes_snapshot, edges: edges_snapshot, edge_map: std::collections::HashMap::new(), adj: vec![Vec::new(); n_nodes], gamma, epsilon };
                    // Rebuild adj for local copy
                    for (idx, e) in local_graph.edges.iter().enumerate() {
                        local_graph.adj[e.from].push(idx);
                        local_graph.adj[e.to].push(idx);
                    }

                    let mut local_actions = 0usize;
                    let mut local_best_phi = local_graph.phi();
                    let mut changed = false;

                    let batch = select_independent_edges(chunk, &local_graph.edges);
                    let mut applied: Vec<(Action, f64)> = Vec::new();
                    for &edge_idx in &batch {
                        let e = &local_graph.edges[edge_idx];
                        let (a, b) = (e.from, e.to);
                        let (deltas, best_idx) = Critic::evaluate_all(&local_graph, edge_idx, a, b);
                        let select_idx = if temp > 0.0 { boltzmann_select(&deltas, temp) } else { best_idx };
                        let action = action_from_idx(select_idx, a, b);
                        let delta = deltas[select_idx];
                        if delta < 0.0 {
                            applied.push((action, delta));
                        }
                    }
                    applied.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    for (action, _) in &applied {
                        action.apply_to_graph(&mut local_graph);
                        local_actions += 1;
                        changed = true;
                    }
                    let final_local_phi = local_graph.phi();
                    if final_local_phi < local_best_phi { local_best_phi = final_local_phi; }
                    (local_graph.nodes, local_actions, local_best_phi, changed)
                });
                handles.push(handle);
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
        });

        // Merge: apply the best node arrangement from all batches
        let mut any_applied = false;
        for (batch_nodes, batch_actions, batch_best_phi, changed) in &results {
            if *changed {
                any_applied = true;
                actions_taken += batch_actions;
                if *batch_best_phi < best_phi {
                    best_phi = *batch_best_phi;
                    graph.nodes = batch_nodes.clone();
                }
            }
        }

        if !any_applied { break; }

        temperature *= 0.85;
        if temperature <= 0.0 { temperature = 0.0; }
        actor.decay_epsilon(0.997);
    }

    graph.nodes = best_nodes;
    phi_trace.push(graph.phi());
    ResolveResult { iterations: max_iter, phi_trace, actions_taken, converged: true, oscillation_breaks: osc_count }
}

/// Démineur mode: systematically "flag" all violated edges in a graph,
/// removing them all in a single O(|E|) pass and tracking Φ dropped.
/// Guarantees φ < tol on exit (since all edges with φ > tol are removed).
/// Edges are lost permanently — this is an instant-removal strategy.
/// Returns (flags_planted, total_phi_dropped, final_phi).
pub fn demineur_sweep(graph: &mut Graph, tol: f64) -> (usize, f64, f64) {
    let mut flags = 0usize;
    let mut phi_dropped = 0.0;

    if graph.phi() < tol {
        return (0, 0.0, graph.phi());
    }

    // Single O(|E|) pass: drain edges, discard those exceeding tol,
    // rebuild adjacency and edge_map only for survivors.
    // edge_phi depends solely on node states + edge weight, which are
    // invariant during removal, so a single pass yields the correct final state.
    let old_edges = std::mem::take(&mut graph.edges);
    graph.edge_map.clear();
    for adj_list in &mut graph.adj {
        adj_list.clear();
    }

    let mut new_edges: Vec<Edge> = Vec::with_capacity(old_edges.len());
    for e in old_edges {
        let p = graph.edge_phi(&e);
        if p > tol {
            flags += 1;
            phi_dropped += p;
            // edge is dropped permanently
        } else {
            let new_idx = new_edges.len();
            graph.edge_map.insert((e.from, e.to), e.weight);
            graph.edge_map.insert((e.to, e.from), e.weight);
            graph.adj[e.from].push(new_idx);
            graph.adj[e.to].push(new_idx);
            new_edges.push(e);
        }
    }
    graph.edges = new_edges;

    (flags, phi_dropped, graph.phi())
}

/// Démineur avec trace détaillée de chaque drapeau.
/// Affiche Φ avant/après chaque flag (sur le total courant).
/// Retourne (flags, phi_dropped, final_phi, vec![(phi_avant, phi_après, weight)]).
pub fn demineur_sweep_trace(graph: &mut Graph, tol: f64) -> (usize, f64, f64, Vec<(f64, f64, i8)>) {
    let mut flags = 0usize;
    let mut phi_dropped = 0.0;
    let mut trace = Vec::new();

    if graph.phi() < tol {
        return (0, 0.0, graph.phi(), trace);
    }

    // Pre-compute all edge phi values (O(|E|) read-only pass)
    let edge_phi_values: Vec<f64> = graph.edges.iter().map(|e| graph.edge_phi(e)).collect();
    let total_phi: f64 = edge_phi_values.iter().sum();

    // Drain and rebuild
    let old_edges = std::mem::take(&mut graph.edges);
    graph.edge_map.clear();
    for adj_list in &mut graph.adj {
        adj_list.clear();
    }

    let mut new_edges: Vec<Edge> = Vec::with_capacity(old_edges.len());
    let mut remaining_phi = total_phi;
    for (e, p) in old_edges.into_iter().zip(edge_phi_values.into_iter()) {
        if p > tol {
            let phi_before = remaining_phi;
            remaining_phi -= p;
            flags += 1;
            phi_dropped += p;
            trace.push((phi_before, remaining_phi, e.weight));
        } else {
            let new_idx = new_edges.len();
            graph.edge_map.insert((e.from, e.to), e.weight);
            graph.edge_map.insert((e.to, e.from), e.weight);
            graph.adj[e.from].push(new_idx);
            graph.adj[e.to].push(new_idx);
            new_edges.push(e);
            // remaining_phi unchanged — this edge's φ stays in the graph
        }
    }
    graph.edges = new_edges;

    (flags, phi_dropped, graph.phi(), trace)
}

/// Balayage par décroissance exponentielle : à chaque tick, le poids de
/// la pire arête est multiplié par `factor` (ex: 0.95). L'arête est supprimée
/// quand |weight| < 0.5. Aucune suppression brutale.
/// Retourne (flags, phi_dropped, final_phi).
pub fn exponential_decay_sweep(
    graph: &mut Graph,
    tol: f64,
    factor: f64,
) -> (usize, f64, f64) {
    let mut flags = 0usize;
    let mut phi_dropped = 0.0;
    if graph.phi() < tol { return (0, 0.0, graph.phi()); }

    for _ in 0..1000 {
        let violated: Vec<(usize, f64)> = graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, graph.edge_phi(e)))
            .filter(|(_, p)| *p > tol)
            .collect();
        if violated.is_empty() { break; }
        let &(worst_idx, _) = violated.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let (from, to) = { let e = &graph.edges[worst_idx]; (e.from, e.to) };
        let saved = graph.exp_decay_edge_weight(from, to, factor);
        if saved > 0.0 { phi_dropped += saved; flags += 1; }
    }
    (flags, phi_dropped, graph.phi())
}

/// Balayage par inhibition latérale : décroissance progressive du poids
/// des arêtes violées, au lieu de suppression instantanée (flag_edge).
/// Chaque itération réduit le poids de la pire arête de `decay`,
/// jusqu'à `min_weight` où l'arête est supprimée.
/// Retourne (flags, phi_dropped, final_phi).
pub fn lateral_inhibition_sweep(
    graph: &mut Graph,
    tol: f64,
    decay: i8,
) -> (usize, f64, f64) {
    if decay <= 0 { return (0, 0.0, graph.phi()); }
    let mut flags = 0usize;
    let mut phi_dropped = 0.0;
    if graph.phi() < tol { return (0, 0.0, graph.phi()); }

    loop {
        let violated: Vec<(usize, f64)> = graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, graph.edge_phi(e)))
            .filter(|(_, p)| *p > tol)
            .collect();
        if violated.is_empty() { break; }

        let &(worst_idx, _) = violated.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let (from, to) = {
            let e = &graph.edges[worst_idx];
            (e.from, e.to)
        };
        let saved = graph.decay_edge_weight(from, to, decay);
        if saved > 0.0 { phi_dropped += saved; flags += 1; }
    }

    (flags, phi_dropped, graph.phi())
}

/// Version avec trace de lateral_inhibition_sweep.
pub fn lateral_inhibition_trace(
    graph: &mut Graph,
    tol: f64,
    decay: i8,
) -> (usize, f64, f64, Vec<(f64, f64, i8)>) {
    if decay <= 0 { return (0, 0.0, graph.phi(), Vec::new()); }
    let mut flags = 0usize;
    let mut phi_dropped = 0.0;
    let mut trace = Vec::new();

    loop {
        let phi_before = graph.phi();
        if phi_before < tol { break; }

        let violated: Vec<(usize, f64)> = graph.edges.iter().enumerate()
            .map(|(idx, e)| (idx, graph.edge_phi(e)))
            .filter(|(_, p)| *p > tol)
            .collect();
        if violated.is_empty() { break; }

        let &(worst_idx, _) = violated.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let (from, to, weight) = {
            let e = &graph.edges[worst_idx];
            (e.from, e.to, e.weight)
        };
        let saved = graph.decay_edge_weight(from, to, decay);
        let phi_after = graph.phi();
        phi_dropped += saved;
        if saved > 0.0 { flags += 1; }
        trace.push((phi_before, phi_after, weight));
    }

    (flags, phi_dropped, graph.phi(), trace)
}
