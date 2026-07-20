use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct FrictionCalculator {
    pub gamma: f64,
    pub epsilon: f64,
}

impl FrictionCalculator {
    pub fn new(gamma: f64, epsilon: f64) -> Self {
        Self { gamma, epsilon }
    }

    /// Phase 1-15 style: sum of violations across implication/exclusion edges.
    pub fn compute_phi(&self, rates: &Array1<f64>, edges: &[(usize, usize, f64, f64)]) -> f64 {
        let mut phi = 0.0;
        for &(i, j, w, strength) in edges {
            let dot = rates[i] * rates[j];
            if w > 0.0 {
                phi += strength * (self.gamma - dot).max(0.0);
            } else if w < 0.0 {
                phi += strength * (dot - self.epsilon).max(0.0);
            }
        }
        phi
    }

    /// Phase 13 style: phi = 1 - p(curr|prev) * N
    pub fn conceptual_phi(
        &self,
        prev_cluster: usize,
        curr_cluster: usize,
        transition_matrix: &Array1<f64>,
        n_concepts: usize,
        n_cols: usize,
    ) -> f64 {
        let total = transition_matrix
            .iter()
            .skip(prev_cluster * n_cols)
            .take(n_cols)
            .sum::<f64>();
        if total < 1e-6 {
            return 1.0;
        }
        let p = transition_matrix[prev_cluster * n_cols + curr_cluster] / total;
        1.0 - p * n_concepts as f64
    }

    /// Phase 13 V3 style: phi = Euclidean distance on SOM grid, normalized.
    pub fn topological_phi(&self, pred_bmu: usize, actual_bmu: usize, rows: usize, cols: usize) -> f64 {
        let ri = pred_bmu / cols;
        let ci = pred_bmu % cols;
        let rj = actual_bmu / cols;
        let cj = actual_bmu % cols;
        ((ri as f64 - rj as f64).powi(2) + (ci as f64 - cj as f64).powi(2)).sqrt() / (rows + cols) as f64
    }
}

impl Default for FrictionCalculator {
    fn default() -> Self {
        Self::new(0.5, 0.3)
    }
}

// ---------------------------------------------------------------------------
// Tri-friction computation (Phase 20+)
// ---------------------------------------------------------------------------

/// Compute the three-component friction vector for a premise–hypothesis pair.
///
/// Returns `[support, conflict, novelty]` each in [0, 1].
pub fn compute_trifriction(
    premise: &str,
    hypothesis: &str,
    graph: &HashMap<String, HashMap<String, f64>>,
    top_k: usize,
) -> [f64; 3] {
    let np_set = neighborhood(premise, graph, top_k);
    let nh_set = neighborhood(hypothesis, graph, top_k);

    let union = np_set.union(&nh_set).count();
    let support = if union > 0 {
        np_set.intersection(&nh_set).count() as f64 / union as f64
    } else {
        0.0
    };

    let conflict = if !np_set.is_empty() {
        let intersection_count = np_set.intersection(&nh_set).count() as f64;
        1.0 - intersection_count / np_set.len() as f64
    } else {
        0.5
    };

    let ht: Vec<String> = tokenize(hypothesis);
    let ht_in_np = ht.iter().filter(|w| np_set.contains(w.as_str())).count();
    let novelty = if !ht.is_empty() {
        1.0 - ht_in_np as f64 / ht.len() as f64
    } else {
        0.0
    };

    [support, conflict, novelty]
}

/// Pre-sort neighbors for fast tri-friction computation.
pub fn prepare_sorted_neighbors(
    graph: &HashMap<String, HashMap<String, f64>>,
    top_k: usize,
) -> HashMap<String, HashSet<String>> {
    let mut sn = HashMap::new();
    for (word, neighbors) in graph {
        let mut sorted: Vec<(&String, &f64)> = neighbors.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let set: HashSet<String> = sorted
            .iter()
            .take(top_k)
            .map(|(w, _)| (*w).clone())
            .collect();
        sn.insert(word.clone(), set);
    }
    sn
}

/// Tri-friction using pre-sorted neighborhoods (faster).
pub fn compute_trifriction_fast(
    premise: &str,
    hypothesis: &str,
    sorted_neighbors: &HashMap<String, HashSet<String>>,
) -> [f64; 3] {
    let np_set = fast_neighborhood(premise, sorted_neighbors);
    let nh_set = fast_neighborhood(hypothesis, sorted_neighbors);

    let union = np_set.union(&nh_set).count();
    let support = if union > 0 {
        np_set.intersection(&nh_set).count() as f64 / union as f64
    } else {
        0.0
    };

    let conflict = if !np_set.is_empty() {
        let intersection_count = np_set.intersection(&nh_set).count() as f64;
        1.0 - intersection_count / np_set.len() as f64
    } else {
        0.5
    };

    let ht: Vec<String> = tokenize(hypothesis);
    let ht_in_np = ht.iter().filter(|w| np_set.contains(w.as_str())).count();
    let novelty = if !ht.is_empty() {
        1.0 - ht_in_np as f64 / ht.len() as f64
    } else {
        0.0
    };

    [support, conflict, novelty]
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

fn neighborhood(
    text: &str,
    graph: &HashMap<String, HashMap<String, f64>>,
    top_k: usize,
) -> HashSet<String> {
    let mut result = HashSet::new();
    for w in tokenize(text) {
        if let Some(neighbors) = graph.get(&w) {
            let mut sorted: Vec<(&String, &f64)> = neighbors.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (n, _) in sorted.iter().take(top_k) {
                result.insert((*n).clone());
            }
        }
    }
    result
}

fn fast_neighborhood(
    text: &str,
    sorted_neighbors: &HashMap<String, HashSet<String>>,
) -> HashSet<String> {
    let mut result = HashSet::new();
    for w in tokenize(text) {
        if let Some(neighbors) = sorted_neighbors.get(&w) {
            result.extend(neighbors.iter().cloned());
        }
    }
    result
}

// ---------------------------------------------------------------------------
// ID-based tri-friction (works with pre-tokenised ID sequences)
// ---------------------------------------------------------------------------

/// ID-keyed sorted neighbourhoods for tri-friction on token IDs.
pub fn prepare_sorted_id_neighbors(
    graph: &HashMap<u32, HashMap<u32, f64>>,
    top_k: usize,
) -> HashMap<u32, HashSet<u32>> {
    let mut sn = HashMap::new();
    for (&id, neighbors) in graph {
        let mut sorted: Vec<(&u32, &f64)> = neighbors.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let set: HashSet<u32> = sorted.iter().take(top_k).map(|(w, _)| **w).collect();
        sn.insert(id, set);
    }
    sn
}

/// Tri-friction computed from pre-tokenised ID sequences and ID-keyed neighbour sets.
pub fn compute_trifriction_ids(
    premise_ids: &[u32],
    hypothesis_ids: &[u32],
    sorted_neighbors: &HashMap<u32, HashSet<u32>>,
) -> [f64; 3] {
    let np_set = fast_id_neighborhood(premise_ids, sorted_neighbors);
    let nh_set = fast_id_neighborhood(hypothesis_ids, sorted_neighbors);

    let union = np_set.union(&nh_set).count();
    let support = if union > 0 {
        np_set.intersection(&nh_set).count() as f64 / union as f64
    } else {
        0.0
    };

    let conflict = if !np_set.is_empty() {
        let intersection_count = np_set.intersection(&nh_set).count() as f64;
        1.0 - intersection_count / np_set.len() as f64
    } else {
        0.5
    };

    let ht_in_np = hypothesis_ids
        .iter()
        .filter(|id| np_set.contains(id))
        .count();
    let novelty = if !hypothesis_ids.is_empty() {
        1.0 - ht_in_np as f64 / hypothesis_ids.len() as f64
    } else {
        0.0
    };

    [support, conflict, novelty]
}

fn fast_id_neighborhood(
    ids: &[u32],
    sorted_neighbors: &HashMap<u32, HashSet<u32>>,
) -> HashSet<u32> {
    let mut result = HashSet::new();
    for id in ids {
        if let Some(neighbors) = sorted_neighbors.get(id) {
            result.extend(neighbors.iter().copied());
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Directional 6D tri-friction (string-based)
// ---------------------------------------------------------------------------

/// Compute a 6‑component friction vector for a premise–hypothesis pair.
///
/// Returns `[support, conflict, novelty, coverage_p, coverage_h, novelty_h]`:
///   * **support**       — Jaccard(support) of premise/hypothesis neighbourhoods
///   * **conflict**      — 1 − |intersection| / |premise neighbourhood|
///   * **novelty**       — fraction of hypothesis words outside premise neighbourhood
///   * **coverage_p**    — |intersection| / |premise neighbourhood| (how much of premise is covered)
///   * **coverage_h**    — |intersection| / |hypothesis neighbourhood| (how much of hypothesis is covered)
///   * **novelty_h**     — |hypothesis neighbourhood \ premise neighbourhood| / |hypothesis neighbourhood|
pub fn compute_directional_6d(
    premise: &str,
    hypothesis: &str,
    sorted_neighbors: &HashMap<String, HashSet<String>>,
) -> [f64; 6] {
    let np = fast_neighborhood(premise, sorted_neighbors);
    let nh = fast_neighborhood(hypothesis, sorted_neighbors);

    let inter = np.intersection(&nh);
    let inter_count = inter.count();

    let union_count = np.union(&nh).count();

    let support = if union_count > 0 { inter_count as f64 / union_count as f64 } else { 0.0 };
    let conflict = if !np.is_empty() { 1.0 - inter_count as f64 / np.len() as f64 } else { 0.5 };
    let ht: Vec<String> = tokenize(hypothesis);
    let ht_in_np = ht.iter().filter(|w| np.contains(w.as_str())).count();
    let novelty = if !ht.is_empty() { 1.0 - ht_in_np as f64 / ht.len() as f64 } else { 0.0 };

    let coverage_p = if !np.is_empty() { inter_count as f64 / np.len() as f64 } else { 0.0 };
    let coverage_h = if !nh.is_empty() { inter_count as f64 / nh.len() as f64 } else { 0.0 };
    let nh_minus_np = nh.difference(&np).count();
    let novelty_h = if !nh.is_empty() { nh_minus_np as f64 / nh.len() as f64 } else { 0.0 };

    [support, conflict, novelty, coverage_p, coverage_h, novelty_h]
}

/// ID‑based version of `compute_directional_6d`.
pub fn compute_directional_6d_ids(
    premise_ids: &[u32],
    hypothesis_ids: &[u32],
    sorted_neighbors: &HashMap<u32, HashSet<u32>>,
) -> [f64; 6] {
    let np = fast_id_neighborhood(premise_ids, sorted_neighbors);
    let nh = fast_id_neighborhood(hypothesis_ids, sorted_neighbors);

    let inter_count = np.intersection(&nh).count();
    let union_count = np.union(&nh).count();

    let support = if union_count > 0 { inter_count as f64 / union_count as f64 } else { 0.0 };
    let conflict = if !np.is_empty() { 1.0 - inter_count as f64 / np.len() as f64 } else { 0.5 };
    let ht_in_np = hypothesis_ids.iter().filter(|id| np.contains(id)).count();
    let novelty = if !hypothesis_ids.is_empty() { 1.0 - ht_in_np as f64 / hypothesis_ids.len() as f64 } else { 0.0 };

    let coverage_p = if !np.is_empty() { inter_count as f64 / np.len() as f64 } else { 0.0 };
    let coverage_h = if !nh.is_empty() { inter_count as f64 / nh.len() as f64 } else { 0.0 };
    let nh_minus_np = nh.difference(&np).count();
    let novelty_h = if !nh.is_empty() { nh_minus_np as f64 / nh.len() as f64 } else { 0.0 };

    [support, conflict, novelty, coverage_p, coverage_h, novelty_h]
}

// ---------------------------------------------------------------------------
// Typed edge and typed tri-friction (Φ) computation
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize)]
pub struct TypedEdge {
    pub weight: f64,
    pub edge_type: i8,
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a * norm_b < 1e-12 { 0.0 } else { dot / (norm_a * norm_b) }
}

/// Build a typed friction graph where each cooccurrence edge gets a type
/// based on SVD embedding cosine similarity.
///
/// - If cos(z_i, z_j) > 0.2 → implication edge (type=+1)
/// - If cos(z_i, z_j) < -0.1 → exclusion edge (type=-1)
/// - Otherwise → edge is dropped (no type → no edge)
pub fn build_typed_graph(
    friction_graph: &HashMap<String, HashMap<String, f64>>,
    embeddings: &HashMap<String, Vec<f64>>,
) -> HashMap<String, HashMap<String, TypedEdge>> {
    let mut typed = HashMap::new();
    for (word, neighbors) in friction_graph {
        let emb_w = match embeddings.get(word) {
            Some(e) => e,
            None => continue,
        };
        let mut typed_neighbors = HashMap::new();
        for (nbr, &weight) in neighbors {
            let cos = match embeddings.get(nbr) {
                Some(e) => cosine_similarity(emb_w, e),
                None => 0.0,
            };
            let edge_type = if cos > 0.2 {
                1
            } else if cos < -0.1 {
                -1
            } else {
                continue;
            };
            typed_neighbors.insert(
                nbr.clone(),
                TypedEdge { weight, edge_type },
            );
        }
        typed.insert(word.clone(), typed_neighbors);
    }
    typed
}

/// Compute tri-friction Φ using typed edges and SVD embeddings.
///
/// For each premise word i, each hypothesis word j:
///   implication: violation = max(0, gamma - cos)
///   exclusion:   violation = max(0, cos + epsilon)
///
/// Returns [support, conflict, novelty]:
///   support   = mean implication violation (0 = perfect alignment)
///   conflict  = mean exclusion violation (0 = perfectly repelling)
///   novelty   = fraction of hypothesis words with 0 typed neighbours in premise
pub fn compute_typed_trifriction(
    premise_words: &[String],
    hypothesis_words: &[String],
    typed_graph: &HashMap<String, HashMap<String, TypedEdge>>,
    embeddings: &HashMap<String, Vec<f64>>,
    gamma: f64,
    epsilon: f64,
) -> [f64; 3] {
    let mut impl_total = 0.0;
    let mut impl_cnt = 0usize;
    let mut excl_total = 0.0;
    let mut excl_cnt = 0usize;
    let mut hyp_with_neighbors = vec![false; hypothesis_words.len()];

    for i_word in premise_words {
        let typed_i = match typed_graph.get(i_word) {
            Some(m) => m,
            None => continue,
        };
        let emb_i = match embeddings.get(i_word) {
            Some(e) => e,
            None => continue,
        };
        for (j_idx, j_word) in hypothesis_words.iter().enumerate() {
            let edge = match typed_i.get(j_word) {
                Some(e) => e,
                None => continue,
            };
            let emb_j = match embeddings.get(j_word) {
                Some(e) => e,
                None => continue,
            };
            let cos = cosine_similarity(emb_i, emb_j);
            hyp_with_neighbors[j_idx] = true;
            if edge.edge_type == 1 {
                impl_total += (gamma - cos).max(0.0);
                impl_cnt += 1;
            } else if edge.edge_type == -1 {
                excl_total += (cos + epsilon).max(0.0);
                excl_cnt += 1;
            }
        }
    }

    let support = if impl_cnt == 0 { 0.0 } else { impl_total / impl_cnt as f64 };
    let conflict = if excl_cnt == 0 { 0.0 } else { excl_total / excl_cnt as f64 };
    let novelty = if hypothesis_words.is_empty() {
        0.0
    } else {
        let no = hyp_with_neighbors.iter().filter(|&&seen| !seen).count();
        no as f64 / hypothesis_words.len() as f64
    };
    [support, conflict, novelty]
}

/// Weighted tri-friction Φ: violation multiplied by edge.weight.
pub fn compute_weighted_trifriction(
    premise_words: &[String],
    hypothesis_words: &[String],
    typed_graph: &HashMap<String, HashMap<String, TypedEdge>>,
    embeddings: &HashMap<String, Vec<f64>>,
    gamma: f64,
    epsilon: f64,
) -> [f64; 3] {
    let mut impl_total = 0.0;
    let mut impl_cnt = 0usize;
    let mut excl_total = 0.0;
    let mut excl_cnt = 0usize;
    let mut hyp_with_neighbors = vec![false; hypothesis_words.len()];

    for i_word in premise_words {
        let typed_i = match typed_graph.get(i_word) {
            Some(m) => m,
            None => continue,
        };
        let emb_i = match embeddings.get(i_word) {
            Some(e) => e,
            None => continue,
        };
        for (j_idx, j_word) in hypothesis_words.iter().enumerate() {
            let edge = match typed_i.get(j_word) {
                Some(e) => e,
                None => continue,
            };
            let emb_j = match embeddings.get(j_word) {
                Some(e) => e,
                None => continue,
            };
            let cos = cosine_similarity(emb_i, emb_j);
            hyp_with_neighbors[j_idx] = true;
            if edge.edge_type == 1 {
                impl_total += edge.weight * (gamma - cos).max(0.0);
                impl_cnt += 1;
            } else if edge.edge_type == -1 {
                excl_total += edge.weight * (cos + epsilon).max(0.0);
                excl_cnt += 1;
            }
        }
    }

    let support = if impl_cnt == 0 { 0.0 } else { impl_total / impl_cnt as f64 };
    let conflict = if excl_cnt == 0 { 0.0 } else { excl_total / excl_cnt as f64 };
    let novelty = if hypothesis_words.is_empty() {
        0.0
    } else {
        let no = hyp_with_neighbors.iter().filter(|&&s| !s).count();
        no as f64 / hypothesis_words.len() as f64
    };
    [support, conflict, novelty]
}

// ---------------------------------------------------------------------------
// R-STDP prediction and edge plasticity
// ---------------------------------------------------------------------------

/// Predict NLI label (0=entail,1=neutral,2=contradict) from Φ values.
///   support < 0.15 && novelty < 0.50 → entailment
///   conflict > 0.25 → contradiction
///   else → neutral
pub fn predict_from_phi(phi: &[f64; 3]) -> usize {
    if phi[0] < 0.15 && phi[2] < 0.50 {
        0
    } else if phi[1] > 0.25 {
        2
    } else {
        1
    }
}

const W_MIN: f64 = 0.01;
const W_MAX: f64 = 10.0;

/// Label-supervised plasticity on typed edge weights.
///
/// Uses the TRUE label as the teaching signal (not prediction correctness).
///
/// Implication edges:
///   label=entailment → Δw = +lr * (1 − violation/γ)   [low violation → strengthen]
///   label≠entailment → Δw = −lr * (1 − violation/γ)   [low violation → weaken]
///
/// Exclusion edges:
///   label=contradiction → Δw = +lr * violation/(1+ε)  [high violation → strengthen]
///   label≠contradiction → Δw = −lr * violation/(1+ε)  [high violation → weaken]
pub fn rstdp_update_edges(
    typed_graph: &mut HashMap<String, HashMap<String, TypedEdge>>,
    premise_words: &[String],
    hypothesis_words: &[String],
    embeddings: &HashMap<String, Vec<f64>>,
    label: usize,
    _prediction: usize,
    gamma: f64,
    epsilon: f64,
    lr: f64,
) {
    for p_word in premise_words {
        let emb_p = match embeddings.get(p_word) {
            Some(e) => e,
            None => continue,
        };
        let Some(neighbors) = typed_graph.get_mut(p_word) else { continue };
        for h_word in hypothesis_words {
            let Some(edge) = neighbors.get_mut(h_word) else { continue };
            let emb_h = match embeddings.get(h_word) {
                Some(e) => e,
                None => continue,
            };
            let cos = cosine_similarity(emb_p, emb_h);
            let (mut delta, bound) = if edge.edge_type == 1 {
                let viol = (gamma - cos).max(0.0);
                let mut d = lr * (1.0 - viol / gamma.max(1e-12));
                if label != 0 { d = -d; }
                let b = (W_MAX - edge.weight) / (W_MAX - W_MIN);
                (d, b)
            } else if edge.edge_type == -1 {
                let viol = (cos + epsilon).max(0.0);
                let mut d = lr * viol / (1.0 + epsilon).max(1e-12);
                if label != 2 { d = -d; }
                let b = (W_MAX - edge.weight) / (W_MAX - W_MIN);
                (d, b)
            } else {
                continue;
            };
            edge.weight = (edge.weight + delta * bound.max(0.0)).clamp(W_MIN, W_MAX);
        }
    }
}
