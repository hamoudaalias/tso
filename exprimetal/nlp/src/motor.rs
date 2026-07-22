use std::collections::HashSet;
use tso_engine::core::{Graph, NodeId};
use crate::corpus::Corpus;
use tso_engine::neurons::{DualLIFState, LIFState};

/// Score(w) = λ·⟨S_t, e(w)⟩ + (1-λ)·topo_bonus − rep_penalty
pub fn generate_next_word(
    lif_state: &ndarray::Array1<f64>,
    last_word: NodeId,
    graph: &Graph,
    used_words: &HashSet<NodeId>,
    lambda: f64,
) -> (NodeId, f64) {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_word = 0;
    let mut best_align = 0.0;

    for (idx, emb) in graph.nodes.iter().enumerate() {
        let alignment = lif_state.dot(emb);

        let topo_bonus = match graph.edge_weight(last_word, idx) {
            Some(1) => 1.0,
            Some(-1) => -1.0,
            _ => 0.0,
        };

        let rep_penalty = if used_words.contains(&idx) { -5.0 } else { 0.0 };

        let score = lambda * alignment + (1.0 - lambda) * topo_bonus + rep_penalty;

        if score > best_score {
            best_score = score;
            best_word = idx;
            best_align = alignment;
        }
    }

    (best_word, best_align)
}

/// Generate a sequence from a prompt, handling `not` as negation operator.
pub fn generate_sequence(
    graph: &Graph,
    corpus: &Corpus,
    prompt_tokens: &[&str],
    max_gen: usize,
    alpha: f64,
    lambda: f64,
) -> Vec<(String, f64)> {
    let mut lif = LIFState::new(graph.nodes[0].len(), alpha);
    let mut used: HashSet<NodeId> = HashSet::new();
    let mut output: Vec<(String, f64)> = Vec::new();
    let mut last_word = 0;
    let mut negate_next = false;

    for &token in prompt_tokens {
        if token == "not" {
            negate_next = true;
            continue;
        }
        let Some(&id) = corpus.vocab.word_to_id.get(token) else { continue; };
        lif.step(&graph.nodes[id], negate_next);
        used.insert(id);
        output.push((token.to_string(), 0.0));
        last_word = id;
        negate_next = false;
    }

    for _ in 0..max_gen {
        let (next_id, align) = generate_next_word(&lif.state, last_word, graph, &used, lambda);
        let word = corpus.vocab.id_to_word[next_id].clone();
        output.push((word, align));
        lif.step(&graph.nodes[next_id], false);
        used.insert(next_id);
        last_word = next_id;
    }

    output
}

/// Dual-LIF version: alignment = β·⟨S_slow, e(w)⟩ + (1-β)·⟨S_fast, e(w)⟩.
pub fn generate_next_word_dual(
    dual: &DualLIFState,
    last_word: NodeId,
    graph: &Graph,
    used_words: &HashSet<NodeId>,
    lambda: f64,
    beta: f64,
) -> (NodeId, f64) {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_word = 0;
    let mut best_align = 0.0;

    for (idx, emb) in graph.nodes.iter().enumerate() {
        let alignment = dual.alignment(emb, beta);

        let topo_bonus = match graph.edge_weight(last_word, idx) {
            Some(1) => 1.0,
            Some(-1) => -1.0,
            _ => 0.0,
        };

        let rep_penalty = if used_words.contains(&idx) { -5.0 } else { 0.0 };

        let score = lambda * alignment + (1.0 - lambda) * topo_bonus + rep_penalty;

        if score > best_score {
            best_score = score;
            best_word = idx;
            best_align = alignment;
        }
    }

    (best_word, best_align)
}

/// Generate a sequence using Dual-LIF reservoirs.
pub fn generate_sequence_dual(
    graph: &Graph,
    corpus: &Corpus,
    prompt_tokens: &[&str],
    max_gen: usize,
    alpha_slow: f64,
    alpha_fast: f64,
    lambda: f64,
    beta: f64,
) -> Vec<(String, f64)> {
    let dim = graph.nodes[0].len();
    let mut dual = DualLIFState::new(dim, alpha_slow, alpha_fast);
    let mut used: HashSet<NodeId> = HashSet::new();
    let mut output: Vec<(String, f64)> = Vec::new();
    let mut last_word = 0;
    let mut negate_next = false;

    for &token in prompt_tokens {
        if token == "not" {
            negate_next = true;
            continue;
        }
        let Some(&id) = corpus.vocab.word_to_id.get(token) else { continue; };
        dual.step(&graph.nodes[id], negate_next);
        used.insert(id);
        output.push((token.to_string(), 0.0));
        last_word = id;
        negate_next = false;
    }

    for _ in 0..max_gen {
        let (next_id, align) = generate_next_word_dual(&dual, last_word, graph, &used, lambda, beta);
        let word = corpus.vocab.id_to_word[next_id].clone();
        output.push((word, align));
        dual.step(&graph.nodes[next_id], false);
        used.insert(next_id);
        last_word = next_id;
    }

    output
}
