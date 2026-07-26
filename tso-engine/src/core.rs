use ndarray::Array1;
use std::collections::{HashMap, HashSet};

pub type NodeId = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Invert(NodeId),
    Align(NodeId, NodeId),
}

impl Action {
    pub fn index(&self) -> usize {
        match self {
            Action::Invert(_) => 0,
            Action::Align(_, _) => 1,
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: i8,
}

#[derive(Clone, Debug)]
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
                    match ee.weight { 1 => (graph.gamma - dot).max(0.0), -1 => (dot - graph.epsilon).max(0.0), _ => 0.0 }
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
                    match ee.weight { 1 => (graph.gamma - dot).max(0.0), -1 => (dot - graph.epsilon).max(0.0), _ => 0.0 }
                }).sum()
            }
        };

        phi_after - phi_before
    }

    pub fn evaluate_all(graph: &Graph, conflict_edge_idx: usize, a: NodeId, b: NodeId) -> ([f64; 2], usize) {
        let actions = [Action::Invert(b), Action::Align(a, b)];
        let mut deltas = [0.0; 2];
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
    q: [[f64; 2]; 2],
    epsilon: f64,
    eta: f64,
}

impl Actor {
    pub fn new(epsilon: f64, eta: f64) -> Self {
        Actor { q: [[0.0; 2]; 2], epsilon, eta }
    }

    pub fn reinforce(&mut self, conflict: ConflictType, action: &Action, reward: f64) {
        self.q[conflict.index()][action.index()] += self.eta * reward;
    }

    pub fn decay_epsilon(&mut self, factor: f64) {
        self.epsilon = (self.epsilon * factor).max(0.05);
    }

    pub fn propose(&self, conflict: ConflictType) -> usize {
        if rand::random::<f64>() < self.epsilon {
            rand::random::<usize>() % 2
        } else {
            let q_values = &self.q[conflict.index()];
            if q_values[0] >= q_values[1] { 0 } else { 1 }
        }
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
    let mut actor = Actor::new(0.5, 0.15);
    let mut phi_trace = Vec::new();
    let mut actions_taken = 0;

    let mut best_phi = graph.phi();
    let mut best_nodes = graph.nodes.clone();
    let mut stall_count = 0;
    const STALL_LIMIT: usize = 20;

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

        if phi < tol || stall_count >= STALL_LIMIT {
            graph.nodes = best_nodes;
            let final_phi = graph.phi();
            phi_trace.push(final_phi);
            return ResolveResult {
                iterations: iter,
                phi_trace,
                actions_taken,
                converged: true,
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

            let action_idx = actor.propose(conflict);
            let proposed = if action_idx == 0 { Action::Invert(b) } else { Action::Align(a, b) };

            let delta = Critic::evaluate(graph, edge_idx, &proposed);

            if delta < 0.0 {
                applied.push((edge_idx, proposed, delta, conflict, true));
            } else {
                let (deltas, best_idx) = Critic::evaluate_all(graph, edge_idx, a, b);
                let best_action = if best_idx == 0 { Action::Invert(b) } else { Action::Align(a, b) };
                applied.push((edge_idx, best_action, deltas[best_idx], conflict, false));
            }
        }

        applied.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let mut any_applied = false;
        for (_edge_idx, action, _delta, conflict, was_actor) in &applied {
            action.apply_to_graph(graph);
            if *was_actor {
                actor.reinforce(*conflict, action, 1.0);
            } else {
                actor.reinforce(*conflict, action, -0.3);
            }
            actions_taken += 1;
            any_applied = true;
        }

        if !any_applied {
            if violated.is_empty() { break; }
            let (edge_idx, _) = violated[0];
            let e = &graph.edges[edge_idx];
            let (a, b) = (e.from, e.to);
            let conflict = ConflictType::from_weight(e.weight);

            let action_idx = actor.propose(conflict);
            let proposed = if action_idx == 0 { Action::Invert(b) } else { Action::Align(a, b) };

            let delta = Critic::evaluate(graph, edge_idx, &proposed);

            let (best_action, was_actor) = if delta < 0.0 {
                (proposed, true)
            } else {
                let (_deltas, best_idx) = Critic::evaluate_all(graph, edge_idx, a, b);
                let best_action = if best_idx == 0 { Action::Invert(b) } else { Action::Align(a, b) };
                (best_action, false)
            };

            best_action.apply_to_graph(graph);
            if was_actor {
                actor.reinforce(conflict, &best_action, 1.0);
            } else {
                actor.reinforce(conflict, &best_action, -0.3);
            }
            actions_taken += 1;
        }

        actor.decay_epsilon(0.997);
    }

    graph.nodes = best_nodes;
    phi_trace.push(graph.phi());
    ResolveResult {
        iterations: max_iter,
        phi_trace,
        actions_taken,
        converged: true,
    }
}
