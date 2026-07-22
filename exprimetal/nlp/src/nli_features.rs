use ndarray::Array1;
use std::collections::HashSet;
use tso_engine::core::Graph;

/// Extract a 4D topological feature vector for NLI:
///   [global_align, imp_score, excl_score, novelty]
pub fn extract_nli_features(
    premise_ids: &[usize],
    hypothesis_ids: &[usize],
    graph: &Graph,
    lif_premise: &Array1<f64>,
    lif_hypothesis: &Array1<f64>,
) -> Array1<f64> {
    let na = lif_premise.dot(lif_premise).sqrt().max(1e-12);
    let nb = lif_hypothesis.dot(lif_hypothesis).sqrt().max(1e-12);
    let global_align = lif_premise.dot(lif_hypothesis) / (na * nb);

    let mut imp_score = 0.0;
    let mut excl_score = 0.0;
    let mut novelty = 0.0;

    for &wh in hypothesis_ids {
        let mut connected = false;
        for &wp in premise_ids {
            if let Some(w) = graph.edge_weight(wp, wh) {
                connected = true;
                match w {
                    1 => imp_score += 1.0,
                    -1 => excl_score += 1.0,
                    _ => {}
                }
            }
        }
        if !connected {
            novelty += 1.0;
        }
    }

    let n = hypothesis_ids.len().max(1) as f64;
    Array1::from_vec(vec![
        global_align,
        imp_score / n,
        excl_score / n,
        novelty / n,
    ])
}

/// Extract a 28D feature vector preserving Top-K distributions:
///   1. Dual-LIF (6D): cos_slow, cos_fast, dist_slow, dist_fast, ratio_slow, ratio_fast
///   2. Best alignments per hypothesis word, sorted top 5 (5D)
///   3. Aggregate alignment (3D): max_align, coverage, cross_negation
///   4. Implication edge dot-products, sorted top 5 (5D)
///   5. Exclusion edge dot-products, sorted top 5 (5D)
///   6. Novelty (1D)
///   7. Jaccard (3D): words, impl_density, excl_density
pub fn extract_features_28d(
    premise_ids: &[usize],
    hypothesis_ids: &[usize],
    graph: &Graph,
    slow_p: &Array1<f64>,
    fast_p: &Array1<f64>,
    slow_h: &Array1<f64>,
    fast_h: &Array1<f64>,
    premise_has_not: bool,
    hypothesis_has_not: bool,
) -> Array1<f64> {
    const K: usize = 5;

    // 1. Dual-LIF (6D)
    let np_s = slow_p.dot(slow_p).sqrt().max(1e-12);
    let nh_s = slow_h.dot(slow_h).sqrt().max(1e-12);
    let np_f = fast_p.dot(fast_p).sqrt().max(1e-12);
    let nh_f = fast_h.dot(fast_h).sqrt().max(1e-12);
    let cos_slow = slow_p.dot(slow_h) / (np_s * nh_s);
    let cos_fast = fast_p.dot(fast_h) / (np_f * nh_f);
    let dist_slow = (slow_p - slow_h).mapv(|x| x * x).sum().sqrt();
    let dist_fast = (fast_p - fast_h).mapv(|x| x * x).sum().sqrt();

    // 2. Best alignments per hypothesis word → collect, sort, top K
    let mut align_scores = Vec::new();
    for &wh in hypothesis_ids {
        let mut best = 0.0;
        for &wp in premise_ids {
            let n1 = graph.nodes[wp].dot(&graph.nodes[wp]).sqrt().max(1e-12);
            let n2 = graph.nodes[wh].dot(&graph.nodes[wh]).sqrt().max(1e-12);
            let sim = graph.nodes[wp].dot(&graph.nodes[wh]) / (n1 * n2);
            if sim > best {
                best = sim;
            }
        }
        align_scores.push(best);
    }
    align_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // 3. Aggregate alignment
    let n_h = hypothesis_ids.len().max(1) as f64;
    let max_align = *align_scores.first().unwrap_or(&0.0);
    let coverage = align_scores.iter().filter(|&&s| s > 0.5).count() as f64 / n_h;
    let cross_negation = if premise_has_not != hypothesis_has_not { 1.0 } else { 0.0 };

    // 4-5. Edge dot-products for impl/excl
    let mut impl_dots = Vec::new();
    let mut excl_dots = Vec::new();
    for &wh in hypothesis_ids {
        for &wp in premise_ids {
            if let Some(w) = graph.edge_weight(wp, wh) {
                let n1 = graph.nodes[wp].dot(&graph.nodes[wp]).sqrt().max(1e-12);
                let n2 = graph.nodes[wh].dot(&graph.nodes[wh]).sqrt().max(1e-12);
                let dot = graph.nodes[wp].dot(&graph.nodes[wh]) / (n1 * n2);
                match w {
                    1 => impl_dots.push(dot),
                    -1 => excl_dots.push(dot),
                    _ => {}
                }
            }
        }
    }
    impl_dots.sort_by(|a, b| b.partial_cmp(a).unwrap());
    excl_dots.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // 6. Novelty: hypothesis words with no edge to premise
    let novelty = {
        let mut count = 0.0;
        for &wh in hypothesis_ids {
            let connected = premise_ids.iter().any(|&wp| graph.edge_weight(wp, wh).is_some());
            if !connected { count += 1.0; }
        }
        count / n_h
    };

    // 7. Jaccard
    let n_p = premise_ids.len().max(1) as f64;
    let p_set: HashSet<&usize> = premise_ids.iter().collect();
    let h_set: HashSet<&usize> = hypothesis_ids.iter().collect();
    let intersection = p_set.intersection(&h_set).count() as f64;
    let union = p_set.union(&h_set).count().max(1) as f64;
    let jaccard_words = intersection / union;

    let mut imp_edges = 0.0;
    let mut excl_edges = 0.0;
    let total_pairs = (n_p * n_h).max(1.0);
    for &wp in premise_ids {
        for &wh in hypothesis_ids {
            if let Some(w) = graph.edge_weight(wp, wh) {
                match w {
                    1 => imp_edges += 1.0,
                    -1 => excl_edges += 1.0,
                    _ => {}
                }
            }
        }
    }

    // Build vector
    let mut v = Vec::with_capacity(28);

    // 1. Dual-LIF (6)
    v.push(cos_slow.min(1.0).max(-1.0));
    v.push(cos_fast.min(1.0).max(-1.0));
    v.push(dist_slow);
    v.push(dist_fast);
    v.push((np_s / nh_s).min(10.0));
    v.push((np_f / nh_f).min(10.0));

    // 2. Best alignments top K (5)
    for i in 0..K {
        v.push(*align_scores.get(i).unwrap_or(&0.0));
    }

    // 3. Aggregate alignment (3)
    v.push(max_align.min(1.0).max(-1.0));
    v.push(coverage);
    v.push(cross_negation);

    // 4. Implication edge dots top K (5)
    for i in 0..K {
        v.push(*impl_dots.get(i).unwrap_or(&0.0));
    }

    // 5. Exclusion edge dots top K (5)
    for i in 0..K {
        v.push(*excl_dots.get(i).unwrap_or(&0.0));
    }

    // 6. Novelty (1)
    v.push(novelty);

    // 7. Jaccard (3)
    v.push(jaccard_words);
    v.push(imp_edges / total_pairs);
    v.push(excl_edges / total_pairs);

    Array1::from_vec(v)
}
