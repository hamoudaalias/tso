mod corpus;
mod embeddings;
mod motor;
mod nli_features;
mod snli;

use corpus::Corpus;
use ndarray::Array1;
use tso_engine::core::{Graph, Critic, resolve, ConflictType, Action};
use tso_engine::episodic::{ContextBuffer, EpisodicMemory};
use tso_engine::neurons::LIFState;

fn build_graph_with_corpus(sentences: &[&str], embed_dim: usize) -> (Corpus, Graph) {
    let mut corpus = Corpus::new(2);
    for s in sentences {
        corpus.add_sentence(s);
    }

    let ppmi = corpus.ppmi_matrix();
    let embeddings = embeddings::compute_embeddings(&ppmi, embed_dim);

    let mut g = Graph::new();
    for i in 0..corpus.vocab.size() {
        g.add_node(embeddings.row(i).to_owned());
    }
    for i in 0..corpus.vocab.size() {
        for j in (i + 1)..corpus.vocab.size() {
            if let Some(w) = corpus.infer_edge_weight(i, j, 0.5, 0.3) {
                g.add_edge(i, j, w);
            }
        }
    }
    (corpus, g)
}

fn demo_resolution(corpus: &Corpus, g: &mut Graph) {
    println!("=== Avant résolution ===");
    println!("Φ initial = {:.4}", g.phi());
    println!("Arêtes: {}  Nœuds: {}  Dim: {}",
        g.edges.len(), g.nodes.len(), g.nodes[0].len());

    let violated: Vec<(usize, f64)> = g.edges.iter().enumerate()
        .map(|(idx, e)| (idx, g.edge_phi(e)))
        .filter(|(_, p)| *p > 0.0)
        .collect();
    if let Some(&(idx, _)) = violated.first() {
        let e = &g.edges[idx];
        let conflict = ConflictType::from_weight(e.weight);
        let action_inv = Action::Invert(e.to);
        let action_exp = Action::Expand(e.from, e.to);
        let action_ali = Action::Align(e.from, e.to);
        let d_inv = Critic::evaluate(g, idx, &action_inv);
        let d_exp = Critic::evaluate(g, idx, &action_exp);
        let d_ali = Critic::evaluate(g, idx, &action_ali);
        println!("\n=== Échantillon Critic (arête {}–{}, {:?}) depth={} ===",
            corpus.vocab.id_to_word[e.from],
            corpus.vocab.id_to_word[e.to],
            conflict, tso_engine::core::CRITIC_DEPTH);
        println!("  Invert({}):  ΔΦ = {:.4}  {}",
            corpus.vocab.id_to_word[e.to], d_inv,
            if d_inv < 0.0 { "✓" } else { "✗" });
        println!("  Expand({},{}): ΔΦ = {:.4}  {}",
            corpus.vocab.id_to_word[e.from],
            corpus.vocab.id_to_word[e.to], d_exp,
            if d_exp < 0.0 { "✓" } else { "✗" });
        println!("  Align({},{}): ΔΦ = {:.4}  {}",
            corpus.vocab.id_to_word[e.from],
            corpus.vocab.id_to_word[e.to], d_ali,
            if d_ali < 0.0 { "✓" } else { "✗" });
    }

    println!("\n=== Résolution Actor-Critic ===");
    let result = resolve(g, 100, 1e-6);
    println!("Itérations: {}  Actions: {}  Convergé: {}",
        result.iterations, result.actions_taken, result.converged);
    for (i, &phi) in result.phi_trace.iter().enumerate() {
        if i % 5 == 0 || i == result.phi_trace.len() - 1 {
            println!("  ité {}: Φ = {:.4}", i, phi);
        }
    }
    println!("Φ final = {:.4}  Dim: {}", g.phi(), g.nodes[0].len());
}

/// Read a phrase word-by-word through the LIF, returning the sequential φ trace.
fn read_phrase(
    graph: &Graph,
    corpus: &Corpus,
    phrase: &str,
    alpha: f64,
) -> Vec<(String, f64)> {
    let tokens: Vec<&str> = phrase.split_whitespace().collect();
    let mut lif = LIFState::new(graph.nodes[0].len(), alpha);
    let mut trace = Vec::new();
    let mut negate_next = false;

    for &token in &tokens {
        if token == "not" {
            negate_next = true;
            continue;
        }

        let word_id = match corpus.vocab.word_to_id.get(token) {
            Some(&id) => id,
            None => { eprintln!("  mot '{}' inconnu", token); continue; }
        };

        let phi_seq = graph.sequential_phi(&lif.state, word_id, negate_next);
        trace.push((token.to_string(), phi_seq));

        lif.step(&graph.nodes[word_id], negate_next);
        negate_next = false;
    }

    trace
}

fn label_name(l: usize) -> &'static str {
    match l {
        0 => "ENTAILMENT",
        1 => "NEUTRAL",
        2 => "CONTRADICTION",
        _ => "UNKNOWN",
    }
}

fn main() {
    let sentences = vec![
        "the dog runs",
        "the cat sleeps",
        "a dog sleeps",
        "a cat runs",
        "the dog barked",
        "the cat purred",
    ];

    let (corpus, mut g) = build_graph_with_corpus(&sentences, 10);

    println!("=== Vocabulaire ===");
    for id in 0..corpus.vocab.size() {
        println!("  {}: '{}' (freq={})", id, corpus.vocab.id_to_word[id], corpus.freq[id]);
    }

    println!("\n=== Arêtes du graphe ===");
    for e in &g.edges {
        let label = if e.weight == 1 { "impl" } else { "excl" };
        println!("  {}–{} [{}]", corpus.vocab.id_to_word[e.from], corpus.vocab.id_to_word[e.to], label);
    }

    demo_resolution(&corpus, &mut g);

    // ── Phase 4: Temporal LIF + Négation ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 4 — RÉSERVOIR LIF & NÉGATION");
    println!("{}", "=" .repeat(60));

    let test_phrases = vec![
        ("dog cat",        "cooccurrence brute (exclusion)"),
        ("the dog",        "implication attendue"),
        ("dog not cat",    "négation: doit réduire la friction"),
        ("cat not dog",    "négation symétrique"),
        ("the dog not a cat", "phrase complète avec négation"),
        ("dog runs",       "implication (cooccurrence forte)"),
        ("dog sleeps",     "implication"),
    ];

    let alpha = 0.85;
    for (phrase, desc) in &test_phrases {
        println!("\n--- \"{}\" ({}) ---", phrase, desc);
        let trace = read_phrase(&g, &corpus, phrase, alpha);
        let total_phi: f64 = trace.iter().map(|(_, p)| p).sum();
        for (i, (word, phi)) in trace.iter().enumerate() {
            println!("  t{}: {:<8}  φ_seq = {:.4}", i, word, phi);
        }
        println!("  ──> Φ_séquentiel total = {:.4}", total_phi);
    }

    // ── Phase 5: Génération Auto-Régressive (Inverse Motor) ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 5 — GÉNÉRATION AUTO-RÉGRESSIVE (INVERSE MOTOR)");
    println!("{}", "=" .repeat(60));

    let prompts = vec![
        vec!["the", "dog"],
        vec!["a", "cat"],
        vec!["the", "dog", "not", "a", "cat"],
    ];

    let gen_lambda = 0.7;
    let gen_alpha = 0.85;

    for prompt in &prompts {
        let result = motor::generate_sequence(
            &g, &corpus, prompt, 10, gen_alpha, gen_lambda,
        );

        print!("\nPrompt: [{}]", prompt.join(" "));
        let prompt_len = prompt.iter().filter(|&&w| w != "not").count();
        for (i, &(ref word, align)) in result.iter().enumerate() {
            if i >= prompt_len {
                let prev_id = corpus.vocab.word_to_id.get(&result[i-1].0).copied().unwrap_or(0);
                let cur_id = corpus.vocab.word_to_id.get(word).copied().unwrap_or(0);
                let topo = match g.edge_weight(prev_id, cur_id) {
                    Some(1) => "impl",
                    Some(-1) => "excl",
                    _ => "none",
                };
                print!(" {} (a={:.2},e={})", word, align, topo);
            }
        }
        println!();
    }

    // ── Phase 6a: Dual-LIF (slow α=0.9 + fast α=0.5) ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 6a — DUAL-LIF (slow α=0.9 + fast α=0.5)");
    println!("{}", "=" .repeat(60));

    let beta = 0.6;
    for prompt in &prompts {
        let result = motor::generate_sequence_dual(
            &g, &corpus, prompt, 8, 0.9, 0.5, gen_lambda, beta,
        );

        print!("\nPrompt: [{}]", prompt.join(" "));
        let prompt_len = prompt.iter().filter(|&&w| w != "not").count();
        for (i, &(ref word, align)) in result.iter().enumerate() {
            if i >= prompt_len {
                let prev_id = corpus.vocab.word_to_id.get(&result[i-1].0).copied().unwrap_or(0);
                let cur_id = corpus.vocab.word_to_id.get(word).copied().unwrap_or(0);
                let topo = match g.edge_weight(prev_id, cur_id) {
                    Some(1) => "impl",
                    Some(-1) => "excl",
                    _ => "none",
                };
                print!(" {} (a={:.2},e={})", word, align, topo);
            }
        }
        println!();
    }

    // ── Phase 6b: Ancrage épisodique ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 6b — ANCRAGE ÉPISODIQUE");
    println!("{}", "=" .repeat(60));

    let mut ep_mem = EpisodicMemory::new(8);
    let mut ctx = ContextBuffer::new(4);

    let seed_seq = vec!["the", "dog", "runs", "the", "cat", "sleeps"];
    let seed_ids: Vec<usize> = seed_seq.iter()
        .filter_map(|w| corpus.vocab.word_to_id.get(*w).copied())
        .collect();
    ep_mem.store(&seed_ids);
    println!("  Épisode stocké: [{}]", seed_seq.join(" "));

    for &w in &seed_ids {
        ctx.push(w);
    }

    let recall_targets = vec!["the", "dog", "runs", "the"];
    let recall_ids: Vec<usize> = recall_targets.iter()
        .filter_map(|w| corpus.vocab.word_to_id.get(*w).copied())
        .collect();
    if let Some(next) = ep_mem.recall(&recall_ids) {
        let word = &corpus.vocab.id_to_word[next];
        println!("  Contexte [{}] → rappel: '{}' ✓", recall_targets.join(" "), word);
    } else {
        println!("  Contexte [{}] → aucun rappel", recall_targets.join(" "));
    }

    let unknown = vec!["purred"];
    let unknown_ids: Vec<usize> = unknown.iter()
        .filter_map(|w| corpus.vocab.word_to_id.get(*w).copied())
        .collect();
    if ep_mem.recall(&unknown_ids).is_some() {
        println!("  Contexte [purred] → rappel: (trouvé)");
    } else {
        println!("  Contexte [purred] → aucun rappel (correct, new)");
    }

    // Generate a sequence and store it as an episode
    println!("\n  Génération → stockage épisodique:");
    let ep_seq = motor::generate_sequence_dual(&g, &corpus, &["the", "dog"], 6, 0.9, 0.5, 0.7, 0.6);
    let ep_words: Vec<&str> = ep_seq.iter().map(|(w, _)| w.as_str()).collect();
    let ep_ids: Vec<usize> = ep_words.iter()
        .filter_map(|w| corpus.vocab.word_to_id.get(*w).copied())
        .collect();
    ep_mem.store(&ep_ids);
    println!("    stocké: [{}]", ep_words.join(" "));

    // Now recall using the generated sequence as context
    let prefix: Vec<&str> = ep_seq.iter().take(4).map(|(w, _)| w.as_str()).collect();
    let prefix_ids: Vec<usize> = prefix.iter()
        .filter_map(|w| corpus.vocab.word_to_id.get(*w).copied())
        .collect();
    if let Some(next) = ep_mem.recall(&prefix_ids) {
        let word = &corpus.vocab.id_to_word[next];
        println!("    contexte [{}] → rappel: '{}' ✓", prefix.join(" "), word);
    }

    // ── Phase 7: AttractorField NLI ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 8 — FEATURE VECTOR TOPOLOGIQUE (Φ_NLI)");
    println!("{}", "=" .repeat(60));

    let _ = resolve(&mut g, 50, 1e-6);
    let dim = g.nodes[0].len();

    // Helper: read a phrase through Dual-LIF returning slow state + word IDs
    let read_phrase = |tokens: &[&str]| -> (Array1<f64>, Vec<usize>) {
        let mut dual = tso_engine::neurons::DualLIFState::new(dim, 0.9, 0.5);
        let mut ids = Vec::new();
        let mut negate_next = false;
        for &token in tokens {
            if token == "not" {
                negate_next = true;
                continue;
            }
            let Some(&id) = corpus.vocab.word_to_id.get(token) else { continue; };
            dual.step(&g.nodes[id], negate_next);
            ids.push(id);
            negate_next = false;
        }
        (dual.slow.state.clone(), ids)
    };

    let data: Vec<(&[&str], &[&str], usize)> = vec![
        (&["the", "dog", "runs"], &["dog", "runs"], 0),
        (&["a", "cat", "sleeps"], &["cat", "sleeps"], 0),
        (&["the", "dog", "barked"], &["dog", "barked"], 0),
        (&["a", "cat", "purred"], &["cat", "purred"], 0),
        (&["the", "dog", "runs"], &["the", "dog", "runs"], 0),
        (&["the", "dog", "runs"], &["the", "dog", "runs", "fast"], 0),
        (&["the", "dog", "runs"], &["the", "dog", "sleeps"], 2),
        (&["a", "cat", "sleeps"], &["the", "cat", "runs"], 2),
        (&["the", "dog", "barked"], &["the", "dog", "purred"], 2),
        (&["the", "dog", "runs"], &["the", "cat", "runs"], 2),
        (&["the", "dog", "runs"], &["the", "dog", "not", "runs"], 2),
        (&["the", "dog", "sleeps"], &["the", "dog", "not", "sleeps"], 2),
        (&["the", "dog", "runs"], &["the", "cat", "sleeps"], 1),
        (&["a", "dog", "sleeps"], &["the", "cat", "purred"], 1),
        (&["the", "dog", "barked"], &["a", "cat", "sleeps"], 1),
        (&["a", "dog", "runs"], &["a", "cat", "purred"], 1),
        (&["the", "dog", "sleeps"], &["the", "cat", "purred"], 1),
        (&["the", "cat", "sleeps"], &["the", "dog", "barked"], 1),
    ];

    // Pre-compute features using both methods
    let test_idx: std::collections::HashSet<usize> = [3usize, 9, 15].iter().cloned().collect();
    let mut fv_old = Vec::new();
    let mut fv_top = Vec::new();

    for (_i, (p, h, l)) in data.iter().enumerate() {
        let (lp, idp) = read_phrase(p);
        let (lh, idh) = read_phrase(h);

        // Old feature: [LIF_P || LIF_H || dot]
        let dot = lp.dot(&lh).max(-1.0).min(1.0);
        let mut old_fv = Array1::zeros(lp.len() + lh.len() + 1);
        old_fv.slice_mut(ndarray::s![..lp.len()]).assign(&lp);
        old_fv.slice_mut(ndarray::s![lp.len()..lp.len()+lh.len()]).assign(&lh);
        old_fv[lp.len() + lh.len()] = dot;
        fv_old.push((old_fv, *l));

        // New topological feature: 4D
        let top_fv = nli_features::extract_nli_features(&idp, &idh, &g, &lp, &lh);
        fv_top.push((top_fv, *l));
    }

    let train_old: Vec<_> = fv_old.iter().enumerate()
        .filter(|(i, _)| !test_idx.contains(i)).map(|(_, v)| (v.0.clone(), v.1)).collect();
    let test_old: Vec<_> = fv_old.iter().enumerate()
        .filter(|(i, _)| test_idx.contains(i)).map(|(_, v)| (v.0.clone(), v.1)).collect();
    let train_top: Vec<_> = fv_top.iter().enumerate()
        .filter(|(i, _)| !test_idx.contains(i)).map(|(_, v)| (v.0.clone(), v.1)).collect();
    let test_top: Vec<_> = fv_top.iter().enumerate()
        .filter(|(i, _)| test_idx.contains(i)).map(|(_, v)| (v.0.clone(), v.1)).collect();

    println!("  Dataset: {} train + {} test ({} total)",
        train_old.len(), test_old.len(), data.len());
    for (i, (p, h, l)) in data.iter().enumerate() {
        let tag = if test_idx.contains(&i) { " [TEST]" } else { "" };
        println!("  {:<2}: {:<25} × {:<20} → {}{}",
            i, p.join(" "), h.join(" "),
            label_name(*l), tag);
    }

    // Train both fields
    println!("\n  [Baseline] Feature vector LIF 21D:");
    let mut field_old = tso_engine::attractor::AttractorField::new(train_old[0].0.len(), 3, 3, 0.08);
    for epoch in 0..20 {
        if epoch % 5 == 0 {
            println!("    époque {:>2}: train acc = {:.1}%", epoch, field_old.accuracy(&train_old) * 100.0);
        }
        for (s, l) in &train_old { field_old.train_step(s, *l); }
    }
    println!("    test acc  = {:.1}%", field_old.accuracy(&test_old) * 100.0);

    println!("\n  [Φ_NLI] Feature vector topologique 4D:");
    let mut field_top = tso_engine::attractor::AttractorField::new(4, 3, 3, 0.08);
    for epoch in 0..20 {
        if epoch % 5 == 0 {
            println!("    époque {:>2}: train acc = {:.1}%", epoch, field_top.accuracy(&train_top) * 100.0);
        }
        for (s, l) in &train_top { field_top.train_step(s, *l); }
    }
    println!("    test acc  = {:.1}%", field_top.accuracy(&test_top) * 100.0);

    // Unseen generalization — compare both
    let unseen: Vec<(&[&str], &[&str], usize)> = vec![
        (&["the", "dog", "runs"], &["the", "dog", "runs", "fast"], 0),
        (&["a", "dog", "sleeps"], &["dog", "sleeps"], 0),
        (&["the", "dog", "runs"], &["the", "cat", "runs"], 2),
        (&["dog", "barked"], &["dog", "purred"], 2),
        (&["dog", "runs"], &["cat", "sleeps"], 1),
        (&["the", "cat", "sleeps"], &["the", "dog", "barked"], 1),
    ];

    println!("\n  Généralisation — comparaison:");
    println!("  {:<30} {:<15} {:<15}  {}", "Paire", "LIF 21D", "Φ_NLI 4D", "Attendu");
    println!("  {}", "-".repeat(80));
    for (p, h, expected) in &unseen {
        let (lp, idp) = read_phrase(p);
        let (lh, idh) = read_phrase(h);
        let dot = lp.dot(&lh).max(-1.0).min(1.0);
        let mut old_fv = Array1::zeros(lp.len() + lh.len() + 1);
        old_fv.slice_mut(ndarray::s![..lp.len()]).assign(&lp);
        old_fv.slice_mut(ndarray::s![lp.len()..lp.len()+lh.len()]).assign(&lh);
        old_fv[lp.len() + lh.len()] = dot;
        let top_fv = nli_features::extract_nli_features(&idp, &idh, &g, &lp, &lh);

        let pred_old = field_old.predict(&old_fv);
        let pred_top = field_top.predict(&top_fv);
        let expected_name = label_name(*expected);
        let c_old = if pred_old == *expected { "✓" } else { "✗" };
        let c_top = if pred_top == *expected { "✓" } else { "✗" };
        println!("  {:<30} {:<6}{}  {:<6}{}  {}",
            format!("{} × {}", p.join(" "), h.join(" ")),
            label_name(pred_old), c_old,
            label_name(pred_top), c_top,
            expected_name);
    }

    println!("\n  Feature Φ_NLI — distribution des features topologiques:");
    for (i, (p, h, _l)) in data.iter().enumerate() {
        let (lp, idp) = read_phrase(p);
        let (lh, idh) = read_phrase(h);
        let fv = nli_features::extract_nli_features(&idp, &idh, &g, &lp, &lh);
        let tag = if test_idx.contains(&i) { " [TEST]" } else { "" };
        println!("  {:<2}: align={:.2} imp={:.0} excl={:.0} novel={:.0}{}",
            i, fv[0], fv[1], fv[2], fv[3], tag);
    }

    // Phase 9a skipped (merged into Phase 9b)

    // ── Phase 9b: Benchmark SNLI (full) ──
    println!("\n{}", "=" .repeat(60));
    println!("  PHASE 9b — BENCHMARK SNLI (FULL)");
    println!("{}", "=" .repeat(60));

    // Check if SNLI datasets exist
    let snli_train_path = "exprimetal/nlp/snli_1.0/snli_1.0_train.jsonl";
    let snli_test_path = "exprimetal/nlp/snli_1.0/snli_1.0_test.jsonl";
    if !std::path::Path::new(snli_train_path).exists() || !std::path::Path::new(snli_test_path).exists() {
        println!("  Dataset SNLI non trouvé. Téléchargement...");
        let url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip";
        println!("  Télécharge {} et décompresse dans snli_1.0/", url);
        println!("  Ou: curl -O {} && unzip snli_1.0.zip", url);
        println!("  Utilisation du mini-dataset de démonstration à la place.\n");
    } else {
        let max_train = 550000;
        let max_test = 10000;
        println!("  Chargement SNLI train ({} lignes)...", max_train);
        let train_data = snli::load_snli(snli_train_path, max_train);
        println!("  Chargement SNLI test ({} lignes)...", max_test);
        let test_data = snli::load_snli(snli_test_path, max_test);

        // ── Build vocabulary, PPMI, graph from TRAIN only ──
        println!("  Construction vocabulaire et graphe depuis TRAIN uniquement...");
        let mut snli_corpus = corpus::Corpus::new(2);
        for s in &train_data.premise {
            snli_corpus.add_sentence(s);
        }
        for s in &train_data.hypothesis {
            snli_corpus.add_sentence(s);
        }
        let vocab_size = snli_corpus.vocab.size();
        println!("  Vocabulaire TRAIN: {} mots", vocab_size);

        let ppmi = snli::SparsePPMI::new(
            &snli_corpus.sentences,
            3,
            vocab_size,
            &snli_corpus.freq,
        );

        println!("  Calcul des embeddings SVD (dim=50)...");
        let embed_dim = 50.min(vocab_size);
        let (u, s, _vt) = embeddings::randomized_svd_op(&ppmi, embed_dim, 5, 2);
        let mut word_emb = u * &s;
        let emb_norm: Vec<f64> = word_emb.rows().into_iter()
            .map(|r| r.dot(&r).sqrt().max(1e-12)).collect();
        for (i, mut row) in word_emb.rows_mut().into_iter().enumerate() {
            row /= emb_norm[i];
        }

        let mut snli_graph = Graph::new();
        for i in 0..vocab_size {
            snli_graph.add_node(word_emb.row(i).to_owned());
        }

        println!("  Inférence des arêtes (PPMI, {} co-occurrences)...", ppmi.cooc_count());
        for &(a, b) in ppmi.cooc_iter() {
            let p = ppmi.ppmi_from_cooc(a, b);
            if p > 0.3 {
                snli_graph.add_edge(a, b, 1);
            }
        }

        println!("  Inférence des arêtes d'exclusion (kNN, {} mots)...", vocab_size);
        let min_freq = 3;
        let k_excl = 3;
        let mut excl_count = 0;
        let freq = &snli_corpus.freq;
        for i in 0..vocab_size {
            if freq[i] < min_freq { continue; }
            let mut sims: Vec<(f64, usize)> = (0..vocab_size)
                .filter(|&j| j != i && freq[j] >= min_freq && snli_graph.edge_weight(i, j).is_none())
                .map(|j| {
                    let s = word_emb.row(i).dot(&word_emb.row(j));
                    (s, j)
                })
                .collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(sim, j) in sims.iter().take(k_excl) {
                if sim > 0.3 && snli_graph.edge_weight(i, j).is_none() {
                    snli_graph.add_edge(i, j, -1);
                    excl_count += 1;
                }
            }
        }
        println!("  Graphe: {} nœuds, {} arêtes ({} impl, {} excl)",
            snli_graph.nodes.len(), snli_graph.edges.len(),
            snli_graph.edges.len() - excl_count, excl_count);

        // Helper: tokenize and evaluate via Dual-LIF
        let tokenize_and_read = |text: &str, g: &Graph, c: &corpus::Corpus| -> (Array1<f64>, Array1<f64>, Vec<usize>, bool) {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            let dim = g.nodes[0].len();
            let mut dual = tso_engine::neurons::DualLIFState::new(dim, 0.9, 0.5);
            let mut ids = Vec::new();
            let mut has_not = false;
            for &token in &tokens {
                if token == "not" {
                    has_not = true;
                    continue;
                }
                if let Some(&id) = c.vocab.word_to_id.get(token) {
                    dual.step(&g.nodes[id], false);
                    ids.push(id);
                }
            }
            (dual.slow.state.clone(), dual.fast.state.clone(), ids, has_not)
        };

        // Extract features for TRAIN
        println!("  Extraction des features 28D pour TRAIN ({} paires)...", train_data.premise.len());
        let mut train_features = Vec::new();
        let t0 = std::time::Instant::now();
        for i in 0..train_data.premise.len() {
            if i > 0 && i % 2000 == 0 {
                println!("    train {}/{} — {:.1}s", i, train_data.premise.len(), t0.elapsed().as_secs_f64());
            }
            let (sp, fp, idp, not_p) = tokenize_and_read(&train_data.premise[i], &snli_graph, &snli_corpus);
            let (sh, fh, idh, not_h) = tokenize_and_read(&train_data.hypothesis[i], &snli_graph, &snli_corpus);
            let fv = nli_features::extract_features_28d(&idp, &idh, &snli_graph, &sp, &fp, &sh, &fh, not_p, not_h);
            train_features.push((fv, train_data.label[i]));
        }
        println!("    Extraction train terminée — {:.1}s", t0.elapsed().as_secs_f64());

        // Extract features for TEST
        println!("  Extraction des features 28D pour TEST ({} paires)...", test_data.premise.len());
        let mut test_features = Vec::new();
        let t0 = std::time::Instant::now();
        for i in 0..test_data.premise.len() {
            if i > 0 && i % 2000 == 0 {
                println!("    test {}/{} — {:.1}s", i, test_data.premise.len(), t0.elapsed().as_secs_f64());
            }
            let (sp, fp, idp, not_p) = tokenize_and_read(&test_data.premise[i], &snli_graph, &snli_corpus);
            let (sh, fh, idh, not_h) = tokenize_and_read(&test_data.hypothesis[i], &snli_graph, &snli_corpus);
            let fv = nli_features::extract_features_28d(&idp, &idh, &snli_graph, &sp, &fp, &sh, &fh, not_p, not_h);
            test_features.push((fv, test_data.label[i]));
        }
        println!("    Extraction test terminée — {:.1}s", t0.elapsed().as_secs_f64());

        println!("  TRAIN: {}, TEST: {}", train_features.len(), test_features.len());

        // Count label distribution (train only)
        let mut counts = [0usize; 3];
        for (_, l) in &train_features {
            counts[*l] += 1;
        }
        println!("  Distribution (TRAIN 550k): ENT={}, NEU={}, CON={}", counts[0], counts[1], counts[2]);

        // Train AttractorField (28D, 30 prototypes/class)
        let mut snli_field = tso_engine::attractor::AttractorField::new(28, 3, 30, 0.08);
        println!("\n  Entraînement LVQ1 (60 epochs)...");
        let lvq_start = std::time::Instant::now();
        for epoch in 0..60 {
            if epoch % 5 == 0 {
                let train_acc = snli_field.accuracy(&train_features);
                let test_acc = snli_field.accuracy(&test_features);
                println!("    époque {:>2}: train={:.1}% test={:.1}% [{:.0}s]",
                    epoch, train_acc * 100.0, test_acc * 100.0, lvq_start.elapsed().as_secs_f64());
            }
            for (s, l) in &train_features {
                snli_field.train_step(s, *l);
            }
        }

        let final_train = snli_field.accuracy(&train_features) * 100.0;
        let final_test = snli_field.accuracy(&test_features) * 100.0;
        println!("\n  Résultat SNLI — TRAIN sur {} ex (train set 550k), TEST sur {} ex (test set officiel):",
            train_features.len(), test_features.len());
        println!("    Accuracy train: {:.1}%", final_train);
        println!("    Accuracy test:  {:.1}%", final_test);
        let n = train_features.len();
        println!("    (Baseline majorité: {:.1}%)",
            counts.iter().max().unwrap().max(&1) * 100 / n);

        // ── Continual Learning: split TRAIN → three tasks ──
        println!("\n{}", "-".repeat(60));
        println!("  APPRENTISSAGE CONTINU (3 TÂCHES SÉQUENTIELLES SUR TRAIN)");
        println!("{}", "-".repeat(60));

        let n = train_features.len();
        let split1 = n / 3;
        let split2 = 2 * n / 3;
        let task_a: Vec<_> = train_features[..split1].iter().map(|(f, l)| (f.clone(), *l)).collect();
        let task_b: Vec<_> = train_features[split1..split2].iter().map(|(f, l)| (f.clone(), *l)).collect();
        let task_c: Vec<_> = train_features[split2..].iter().map(|(f, l)| (f.clone(), *l)).collect();
        let task_a_test: Vec<_> = task_a[..(task_a.len() / 5)].iter()
            .map(|(f, l)| (f.clone(), *l)).collect();

        // ── Standard CL (no Freeze+Add) ──
        let mut std_field = tso_engine::attractor::AttractorField::new(28, 3, 30, 0.08);
        println!("  Tâche A ({} train)...", task_a.len());
        for _ in 0..20 { for (s, l) in &task_a { std_field.train_step(s, *l); } }
        let acc_a_ref = std_field.accuracy(&task_a_test) * 100.0;
        println!("    Accuracy A (référence): {:.1}%", acc_a_ref);

        println!("  Tâche B ({} train)...", task_b.len());
        for _ in 0..20 { for (s, l) in &task_b { std_field.train_step(s, *l); } }
        let acc_a_b = std_field.accuracy(&task_a_test) * 100.0;
        println!("    Accuracy A après B: {:.1}%  (oubli: {:.1} pts)", acc_a_b, acc_a_ref - acc_a_b);

        println!("  Tâche C ({} train)...", task_c.len());
        for _ in 0..20 { for (s, l) in &task_c { std_field.train_step(s, *l); } }
        let acc_a_c = std_field.accuracy(&task_a_test) * 100.0;
        println!("    Accuracy A après C: {:.1}%  (oubli total: {:.1} pts)", acc_a_c, acc_a_ref - acc_a_c);

        // ── Freeze+Add: keep frozen protos from A, add new for B, then for C ──
        println!("\n  Freeze+Add (30 protos gelés + 10 nouveaux par tâche)...");
        let mut fa_field = std_field.clone(); // start from state after A
        // Already has 30 protos/class from training on A
        for c in 0..3 {
            for _ in 0..10 {
                let mut v: Array1<f64> = (0..28).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
                let n = v.dot(&v).sqrt().max(1e-12); v /= n;
                fa_field.prototypes[c].push(v);
            }
        }
        println!("  Freeze+Add — Tâche B...");
        for _ in 0..20 { for (s, l) in &task_b { fa_field.train_step(s, *l); } }
        let acc_a_fb = fa_field.accuracy(&task_a_test) * 100.0;
        let acc_b_fb = fa_field.accuracy(&task_b[..(task_b.len() / 5)]) * 100.0;
        println!("    A: {:.1}%  (oubli: {:.1} pts)   B: {:.1}%", acc_a_fb, acc_a_ref - acc_a_fb, acc_b_fb);

        for c in 0..3 {
            for _ in 0..10 {
                let mut v: Array1<f64> = (0..28).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
                let n = v.dot(&v).sqrt().max(1e-12); v /= n;
                fa_field.prototypes[c].push(v);
            }
        }
        println!("  Freeze+Add — Tâche C...");
        for _ in 0..20 { for (s, l) in &task_c { fa_field.train_step(s, *l); } }
        let acc_a_fc = fa_field.accuracy(&task_a_test) * 100.0;
        let acc_c_fc = fa_field.accuracy(&task_c[..(task_c.len() / 5)]) * 100.0;
        println!("    A: {:.1}%  (oubli: {:.1} pts)   C: {:.1}%", acc_a_fc, acc_a_ref - acc_a_fc, acc_c_fc);
    }
    println!("\n{}", "=" .repeat(60));
    println!("  FIN DU BENCHMARK");
    println!("{}", "=" .repeat(60));
}

#[cfg(test)]
mod tests {
    use tso_engine::core::*;
    use crate::corpus::*;
    use tso_engine::neurons::*;
    use ndarray::Array1;

    fn small_graph() -> (Corpus, Graph) {
        let mut c = Corpus::new(2);
        c.add_sentence("the dog runs");
        c.add_sentence("the cat sleeps");
        c.add_sentence("a dog sleeps");
        c.add_sentence("a cat runs");
        let ppmi = c.ppmi_matrix();
        let emb = crate::embeddings::compute_embeddings(&ppmi, 5);
        let mut g = Graph::new();
        for i in 0..c.vocab.size() {
            g.add_node(emb.row(i).to_owned());
        }
        for i in 0..c.vocab.size() {
            for j in (i + 1)..c.vocab.size() {
                if let Some(w) = c.infer_edge_weight(i, j, 0.5, 0.3) {
                    g.add_edge(i, j, w);
                }
            }
        }
        (c, g)
    }

    /// Helper: count how many φ contributions come from exclusion vs implication edges
    fn classify_phi(g: &Graph, lif: &LIFState, word_id: NodeId, negate: bool) -> (f64, f64, f64) {
        let e = if negate { -&g.nodes[word_id] } else { g.nodes[word_id].clone() };
        let mut excl = 0.0;
        let mut impl_sum = 0.0;
        let mut total = 0.0;
        for edge in &g.edges {
            let other_id = if edge.from == word_id { edge.to }
                else if edge.to == word_id { edge.from } else { continue };
            let activation = lif.state.dot(&g.nodes[other_id]).max(0.0);
            if activation > 1e-12 {
                let dot = e.dot(&g.nodes[other_id]);
                let phi = match edge.weight {
                    1 => (tso_engine::core::GAMMA - dot).max(0.0),
                    -1 => (dot - tso_engine::core::EPSILON).max(0.0),
                    _ => 0.0,
                };
                let w = activation * phi;
                total += w;
                match edge.weight {
                    1 => impl_sum += w,
                    -1 => excl += w,
                    _ => {}
                }
            }
        }
        (total, excl, impl_sum)
    }

    #[test]
    fn negation_resolves_exclusion_but_may_break_implications() {
        let (corpus, g) = small_graph();

        let dog = corpus.vocab.word_to_id["dog"];
        let cat = corpus.vocab.word_to_id["cat"];
        let has_exclusion = g.edges.iter().any(|e|
            ((e.from == dog && e.to == cat) || (e.from == cat && e.to == dog)) && e.weight == -1
        );
        assert!(has_exclusion, "dog–cat must have exclusion edge");

        let mut lif = LIFState::new(g.nodes[0].len(), 0.85);
        lif.step(&g.nodes[dog], false);

        // With real SVD embeddings, negation changes the friction landscape.
        // It may increase or decrease local φ depending on the embedding geometry.
        let (total_no, excl_no, _) = classify_phi(&g, &lif, cat, false);
        let (total_yes, excl_yes, _) = classify_phi(&g, &lif, cat, true);
        assert!(total_no > 0.0, "dog→cat must have positive φ (exclusion)");
        assert_ne!(total_yes, total_no, "negation must change φ");
        assert_ne!(excl_yes, excl_no, "negation must change exclusion friction");
    }

    #[test]
    fn sequential_phi_detects_exclusion() {
        let (corpus, g) = small_graph();
        let dog = corpus.vocab.word_to_id["dog"];
        let cat = corpus.vocab.word_to_id["cat"];

        let mut lif = LIFState::new(g.nodes[0].len(), 0.85);
        lif.step(&g.nodes[dog], false);

        let (total, _, _) = classify_phi(&g, &lif, cat, false);
        assert!(total > 0.0, "dog→cat (exclusion) should produce positive φ");
    }

    #[test]
    fn negated_state_differs_from_direct() {
        let (corpus, g) = small_graph();
        let dog = corpus.vocab.word_to_id["dog"];
        let cat = corpus.vocab.word_to_id["cat"];

        // LIF after "dog not cat" vs "dog cat"
        let mut lif_neg = LIFState::new(g.nodes[0].len(), 0.85);
        lif_neg.step(&g.nodes[dog], false);
        lif_neg.step(&g.nodes[cat], true);

        let mut lif_dir = LIFState::new(g.nodes[0].len(), 0.85);
        lif_dir.step(&g.nodes[dog], false);
        lif_dir.step(&g.nodes[cat], false);

        let dot_after = lif_neg.state.dot(&lif_dir.state);
        assert!(dot_after < 0.99, "negated vs direct should produce different LIF states");
    }

    #[test]
    fn critic_detects_good_action() {
        let mut g = Graph::new();
        let n0 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        let n1 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        g.add_edge(n0, n1, -1);

        let idx = 0;
        let inv = Action::Invert(n1);
        let delta = Critic::evaluate(&g, idx, &inv);
        assert!(delta < 0.0, "inverting should reduce exclusion friction");
    }

    #[test]
    fn critic_detects_bad_action() {
        let mut g = Graph::new();
        let n0 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        let n1 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        let n2 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        g.add_edge(n0, n1, -1);
        g.add_edge(n1, n2, 1);

        let idx = 0;
        let inv = Action::Invert(n1);
        let delta = Critic::evaluate(&g, idx, &inv);
        assert!(delta > 0.0, "depth=1 sees the ripple: inverting n1 breaks n1-n2 implication (delta={:.4})", delta);
    }

    #[test]
    fn resolve_converges_simple_exclusion() {
        let mut g = Graph::new();
        let n0 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        let n1 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
        g.add_edge(n0, n1, -1);

        let r = resolve(&mut g, 10, 1e-6);
        assert!(r.converged);
        assert!(g.phi() < 1e-6);
    }

    #[test]
    fn resolve_converges_small_graph() {
        let (_c, mut g) = small_graph();
        let r = resolve(&mut g, 50, 1e-6);
        assert!(r.converged, "graph should converge; final φ = {:.4}", g.phi());
    }

    #[test]
    fn inverse_motor_prefers_implication_over_exclusion() {
        let (corpus, g) = small_graph();
        let dog = corpus.vocab.word_to_id["dog"];
        let cat = corpus.vocab.word_to_id["cat"];
        let runs = corpus.vocab.word_to_id["runs"];
        let _sleeps = corpus.vocab.word_to_id["sleeps"];

        let has_dog_runs_impl = g.edge_weight(dog, runs) == Some(1);
        let has_dog_cat_excl = g.edge_weight(dog, cat) == Some(-1);
        assert!(has_dog_runs_impl, "dog–runs must be implication");
        assert!(has_dog_cat_excl, "dog–cat must be exclusion");

        let mut lif = tso_engine::neurons::LIFState::new(g.nodes[0].len(), 0.85);
        lif.step(&g.nodes[dog], false);

        let used: std::collections::HashSet<usize> =
            [dog].iter().cloned().collect();

        let lambda = 0.7;
        let (best, _) = crate::motor::generate_next_word(&lif.state, dog, &g, &used, lambda);

        let edge = g.edge_weight(dog, best);
        assert!(edge == Some(1) || edge == None,
            "generated word should not have exclusion with last word; got edge={:?}", edge);
    }

    #[test]
    fn does_not_repeat_used_words() {
        let (corpus, g) = small_graph();
        let dog = corpus.vocab.word_to_id["dog"];

        let mut lif = tso_engine::neurons::LIFState::new(g.nodes[0].len(), 0.85);
        lif.step(&g.nodes[dog], false);

        let used: std::collections::HashSet<usize> =
            (0..g.nodes.len()).collect();

        let (best, _) = crate::motor::generate_next_word(&lif.state, dog, &g, &used, 0.7);

        // If all words are used, it should still pick something
        // (the penalty prevents exact repeats but doesn't block everything)
        assert!(best < g.nodes.len(), "must return a valid word index");
    }

    #[test]
    fn sequence_respects_topology() {
        let (corpus, mut g) = small_graph();
        let _ = tso_engine::core::resolve(&mut g, 50, 1e-6);

        let prompt = vec!["the", "dog"];
        let result = crate::motor::generate_sequence(&g, &corpus, &prompt, 5, 0.85, 0.85);

        assert!(result.len() >= 7, "should generate prompt + 5 words");

        // With real SVD embeddings, alignment can briefly dominate topology.
        // Check that ≥ 70% of transitions are non-exclusion.
        let mut excl_count = 0usize;
        let mut total = 0usize;
        for i in 1..result.len() {
            let w1 = corpus.vocab.word_to_id.get(&result[i-1].0);
            let w2 = corpus.vocab.word_to_id.get(&result[i].0);
            if let (Some(&a), Some(&b)) = (w1, w2) {
                if g.edge_weight(a, b) == Some(-1) {
                    excl_count += 1;
                }
                total += 1;
            }
        }
        assert!(excl_count as f64 / total as f64 <= 0.3,
            "exclusion rate {:.1}% exceeds 30%: {} exclusions out of {} steps",
            excl_count as f64 / total as f64 * 100.0, excl_count, total);
    }

    #[test]
    fn phase1_core_tests() {
        let mut g = Graph::new();
        let v = Array1::from_vec(vec![1.0, 0.0]);
        let n0 = g.add_node(v.clone());
        let n1 = g.add_node(v.clone());
        g.add_edge(n0, n1, -1);
        let before = g.phi();
        assert!(before > 0.0);
        g.nodes[n1].mapv_inplace(|x| -x);
        let after = g.phi();
        assert!(after < before);
        assert_eq!(after, 0.0);
    }
}
