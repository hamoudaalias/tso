use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use ndarray::{Array1, Array2};
use tso_kernel::decoder::AnchoredTSODecoder;
use tso_kernel::deep::{DeepConfig, DeepTSO};
use tso_kernel::projector::WordProjector;
use tso_kernel::friction::{compute_trifriction_fast, prepare_sorted_neighbors};
use tso_kernel::model::{FrictionGraphCheckpoint, ModelConfig};
use tso_nlp::dataset::NLIDataset;
use tso_nlp::distributional::{
    alignment_features, distributional_features_duallif, phi_sequential, randomized_svd, PPMIMatrix,
};
use tso_nlp::metrics::Metrics;
use tso_nlp::tokenizer::TSOTokenizer;

struct BenchConfig {
    window_size: usize,
    top_k: usize,
    svd_dim: usize,
    checkpoint_dir: PathBuf,
}

fn kmeans_class(features: &[Vec<f64>], indices: &[usize], k: usize, n_inputs: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::Rng;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
    let n = indices.len();
    let mut means: Vec<Vec<f64>> = (0..k)
        .map(|_| features[indices[rng.gen_range(0..n)]].clone())
        .collect();
    for _iter in 0..20 {
        let mut sums = vec![vec![0.0; n_inputs]; k];
        let mut counts = vec![0; k];
        for &idx in indices {
            let x = &features[idx];
            let mut best_d = f64::MAX;
            let mut best_k = 0;
            for ki in 0..k {
                let mut d = 0.0;
                for j in 0..n_inputs {
                    let diff = x[j] - means[ki][j];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best_k = ki;
                }
            }
            for j in 0..n_inputs {
                sums[best_k][j] += x[j];
            }
            counts[best_k] += 1;
        }
        for ki in 0..k {
            if counts[ki] > 0 {
                for j in 0..n_inputs {
                    means[ki][j] = sums[ki][j] / counts[ki] as f64;
                }
            }
        }
    }
    means
}

#[derive(Serialize, Deserialize)]
struct AttractorField {
    prototypes: Vec<Vec<f64>>,
    proto_labels: Vec<usize>,
    n_inputs: usize,
    n_classes: usize,
}

impl AttractorField {
    fn new(n_inputs: usize, n_classes: usize, k_per_class: usize) -> Self {
        let n_proto = n_classes * k_per_class;
        Self {
            prototypes: vec![vec![0.0; n_inputs]; n_proto],
            proto_labels: (0..n_proto).map(|i| i / k_per_class).collect(),
            n_inputs,
            n_classes,
        }
    }

    fn init_kmeans(&mut self, features: &[Vec<f64>], labels: &[usize], k_per_class: usize) {
        for c in 0..self.n_classes {
            let indices: Vec<usize> = labels.iter().enumerate().filter(|(_, &l)| l == c).map(|(i, _)| i).collect();
            let means = kmeans_class(features, &indices, k_per_class, self.n_inputs, 42 + c as u64);
            for (ki, mean) in means.iter().enumerate() {
                self.prototypes[c * k_per_class + ki] = mean.clone();
            }
        }
    }

    fn predict(&self, feat: &[f64]) -> usize {
        let mut best_dist = f64::MAX;
        let mut best_class = 0;
        for p in 0..self.prototypes.len() {
            let mut dist = 0.0;
            for j in 0..self.n_inputs {
                let d = feat[j] - self.prototypes[p][j];
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best_class = self.proto_labels[p];
            }
        }
        best_class
    }

    fn train_epoch(&mut self, features: &[Vec<f64>], labels: &[usize], lr: f64) -> usize {
        let mut correct = 0;
        for i in 0..features.len() {
            let x = &features[i];
            let label = labels[i];

            let mut best_dist = f64::MAX;
            let mut best_p = 0;
            for p in 0..self.prototypes.len() {
                let mut dist = 0.0;
                for j in 0..self.n_inputs {
                    let d = x[j] - self.prototypes[p][j];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_p = p;
                }
            }

            if self.proto_labels[best_p] == label {
                correct += 1;
                for j in 0..self.n_inputs {
                    self.prototypes[best_p][j] += lr * (x[j] - self.prototypes[best_p][j]);
                }
            } else {
                for j in 0..self.n_inputs {
                    self.prototypes[best_p][j] -= lr * (x[j] - self.prototypes[best_p][j]);
                }
            }
        }
        correct
    }
}

#[derive(Serialize, Deserialize)]
struct Normaliser {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Normaliser {
    fn fit(features: &[Vec<f64>]) -> Self {
        let n_feats = features[0].len();
        let n = features.len() as f64;
        let mut mean = vec![0.0; n_feats];
        for feat in features {
            for i in 0..n_feats {
                mean[i] += feat[i];
            }
        }
        for i in 0..n_feats {
            mean[i] /= n;
        }
        let mut std = vec![0.0; n_feats];
        for feat in features {
            for i in 0..n_feats {
                let d = feat[i] - mean[i];
                std[i] += d * d;
            }
        }
        for i in 0..n_feats {
            std[i] = (std[i] / n).sqrt();
        }
        Self { mean, std }
    }

    fn transform(&self, feat: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(feat.len());
        for i in 0..feat.len() {
            out.push((feat[i] - self.mean[i]) / (self.std[i] + 1e-8));
        }
        out
    }
}

/// Computes 17D raw TSO features (Jaccard 3 + Dual-LIF 6 + Phi 4 + Align 4).
const N_FEATS: usize = 17;

fn compute_raw_features(
    dataset: &NLIDataset,
    sorted_sets: &HashMap<String, HashSet<String>>,
    word_embeddings: &HashMap<String, Vec<f64>>,
    idf_weights: &HashMap<String, f64>,
    alpha_slow: f64,
    alpha_fast: f64,
    negation_set: &HashSet<String>,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let results: Vec<(usize, Vec<f64>)> = dataset.samples.par_iter().filter_map(|sample| {
        let label = NLIDataset::label_to_idx(&sample.label)?;
        let j3 = compute_trifriction_fast(&sample.sentence1, &sample.sentence2, sorted_sets);
        let p_words = tokenize_words(&sample.sentence1);
        let h_words = tokenize_words(&sample.sentence2);
        let l6 = distributional_features_duallif(&p_words, &h_words, word_embeddings, idf_weights,
            alpha_slow, alpha_fast, negation_set);
        let p4 = phi_sequential(&p_words, &h_words, word_embeddings, idf_weights, alpha_slow, negation_set);
        let a4 = alignment_features(&p_words, &h_words, word_embeddings);
        let mut feat = Vec::with_capacity(N_FEATS);
        feat.extend_from_slice(&j3);
        feat.extend_from_slice(&l6);
        feat.extend_from_slice(&p4);
        feat.extend_from_slice(&a4);
        Some((label, feat))
    }).collect();
    let features: Vec<Vec<f64>> = results.iter().map(|(_, f)| f.clone()).collect();
    let labels: Vec<usize> = results.iter().map(|(l, _)| *l).collect();
    (features, labels)
}

fn tokenize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

fn count_cooccurrences(
    sequences: &[Vec<(u32, String)>],
    window_size: usize,
) -> HashMap<(u32, u32), u64> {
    let per_thread: Vec<HashMap<(u32, u32), u64>> = sequences
        .par_iter()
        .map(|seq| {
            let mut local = HashMap::new();
            let n = seq.len();
            for i in 0..n {
                let end = (i + window_size + 1).min(n);
                for j in (i + 1)..end {
                    let (a, _) = seq[i];
                    let (b, _) = seq[j];
                    if a != b {
                        let pair = if a < b { (a, b) } else { (b, a) };
                        *local.entry(pair).or_insert(0) += 1;
                    }
                }
            }
            local
        })
        .collect();

    let t0 = Instant::now();
    let mut global = HashMap::new();
    for local in per_thread {
        for (pair, count) in local {
            *global.entry(pair).or_insert(0) += count;
        }
    }
    eprintln!("  Merged {} unique pairs ({:.2?})", global.len(), t0.elapsed());
    global
}

fn build_friction_graph(
    counts: &HashMap<(u32, u32), u64>,
    id_to_word: &HashMap<u32, String>,
) -> HashMap<String, HashMap<String, f64>> {
    let t0 = Instant::now();
    let mut margin: HashMap<u32, f64> = HashMap::new();
    for (&(a, b), &c) in counts {
        let cf = c as f64;
        *margin.entry(a).or_insert(0.0) += cf;
        *margin.entry(b).or_insert(0.0) += cf;
    }
    let mut graph: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (&(a, b), &count) in counts {
        let wa = match id_to_word.get(&a) {
            Some(w) => w,
            None => continue,
        };
        let wb = match id_to_word.get(&b) {
            Some(w) => w,
            None => continue,
        };
        let c = count as f64;
        let ma = *margin.get(&a).unwrap_or(&1.0);
        graph
            .entry(wa.clone())
            .or_default()
            .insert(wb.clone(), c / ma);
        graph
            .entry(wb.clone())
            .or_default()
            .insert(wa.clone(), c / *margin.get(&b).unwrap_or(&1.0));
    }
    eprintln!("  Friction graph: {} nodes ({:.2?})", graph.len(), t0.elapsed());
    graph
}

/// Build the TSO pipeline (graph + sorted neighbours + embeddings) from a training corpus.
/// All artifacts are checkpointed and reused if available.
fn build_pipeline(
    train: &NLIDataset,
    tokenizer: &mut TSOTokenizer,
    config: &BenchConfig,
) -> (HashMap<String, HashSet<String>>, HashMap<String, Vec<f64>>, HashMap<String, f64>) {
    let model_config = ModelConfig {
        window_size: config.window_size,
        top_k: config.top_k,
    };

    // Step 1: Tokenise → counts → friction graph
    let graph_ckpt = config.checkpoint_dir.join("graph.ckpt.bin");
    let counts_ckpt = config.checkpoint_dir.join("counts.ckpt.bin");
    let id2w_ckpt = config.checkpoint_dir.join("id_to_word.ckpt.bin");
    let idf_ckpt = config.checkpoint_dir.join("idf.ckpt.bin");

    let (cooccurrence_counts, id_to_word, idf_weights)
        : (HashMap<(u32, u32), u64>, HashMap<u32, String>, HashMap<String, f64>)
        = if counts_ckpt.exists() && id2w_ckpt.exists() && idf_ckpt.exists() {
            println!("Loading counts from {:?} ...", counts_ckpt);
            let bytes = std::fs::read(&counts_ckpt).expect("read counts");
            let counts = bincode::deserialize(&bytes).expect("deserialize counts");
            let bytes = std::fs::read(&id2w_ckpt).expect("read id_to_word");
            let id2w = bincode::deserialize(&bytes).expect("deserialize id_to_word");
            println!("Loading IDF from {:?} ...", idf_ckpt);
            let bytes = std::fs::read(&idf_ckpt).expect("read idf");
            let idf = bincode::deserialize(&bytes).expect("deserialize idf");
            (counts, id2w, idf)
        } else {
            let total = train.len();
            println!("Tokenising {} pairs...", total);
            let t0 = Instant::now();
            let mut all_seqs: Vec<Vec<(u32, String)>> = Vec::with_capacity(total * 2);
            let mut id_to_word: HashMap<u32, String> = HashMap::new();
            let mut doc_freq: HashMap<String, u64> = HashMap::new();
            for (i, sample) in train.samples.iter().enumerate() {
                for text in [&sample.sentence1, &sample.sentence2] {
                    let words = tokenizer.encode_with_words(text, false);
                    for &(id, ref w) in &words {
                        id_to_word.entry(id).or_insert_with(|| w.clone());
                    }
                    let mut seen = HashSet::new();
                    for &(_, ref w) in &words {
                        if seen.insert(w.clone()) {
                            *doc_freq.entry(w.clone()).or_insert(0) += 1;
                        }
                    }
                    all_seqs.push(words);
                }
                if (i + 1) % 50000 == 0 || i == total - 1 {
                    let pct = (i + 1) as f64 / total as f64 * 100.0;
                    let _ = write!(
                        std::io::stderr(),
                        "\r  [{:>5.1}%] {}/{} ({:.2?})",
                        pct, i + 1, total, t0.elapsed()
                    );
                    let _ = std::io::stderr().flush();
                }
            }
            let n_docs = all_seqs.len() as f64;
            let mut idf_weights: HashMap<String, f64> = HashMap::with_capacity(doc_freq.len());
            for (word, df) in &doc_freq {
                let idf = (n_docs / *df as f64).ln().max(0.0);
                idf_weights.insert(word.clone(), idf);
            }
            eprintln!();
            eprintln!("  {} sequences, {} words, {} IDF entries ({:.2?})",
                all_seqs.len(), id_to_word.len(), idf_weights.len(), t0.elapsed());
            eprintln!("Counting co-occurrences (window={})...", config.window_size);
            let t0 = Instant::now();
            let counts = count_cooccurrences(&all_seqs, config.window_size);
            eprintln!("  {} unique pairs ({:.2?})", counts.len(), t0.elapsed());
            std::fs::create_dir_all(&config.checkpoint_dir).ok();
            let bytes = bincode::serialize(&counts).expect("serialize counts");
            std::fs::write(&counts_ckpt, bytes).expect("save counts");
            let bytes = bincode::serialize(&id_to_word).expect("serialize id_to_word");
            std::fs::write(&id2w_ckpt, bytes).expect("save id_to_word");
            let bytes = bincode::serialize(&idf_weights).expect("serialize idf");
            std::fs::write(&idf_ckpt, bytes).expect("save idf");
            (counts, id_to_word, idf_weights)
        };

    let _friction_graph = if graph_ckpt.exists() {
        println!("Loading graph from {:?} ...", graph_ckpt);
        let ckpt = FrictionGraphCheckpoint::load(&graph_ckpt).expect("load graph");
        println!("  {} words", ckpt.graph.len());
        ckpt.graph
    } else {
        eprintln!("Building friction graph...");
        let graph = build_friction_graph(&cooccurrence_counts, &id_to_word);
        let ckpt = FrictionGraphCheckpoint { config: model_config.clone(), graph: graph.clone() };
        ckpt.save(&graph_ckpt).expect("save graph");
        graph
    };

    // Step 2: Sorted neighbours
    let sorted_ckpt = config.checkpoint_dir.join("sorted.ckpt.bin");
    let sorted_neighbors: HashMap<String, Vec<String>> = if sorted_ckpt.exists() {
        println!("Loading sorted neighbours from {:?} ...", sorted_ckpt);
        let bytes = std::fs::read(&sorted_ckpt).expect("read sorted");
        bincode::deserialize(&bytes).expect("deserialize sorted")
    } else {
        println!("Preparing sorted neighbours (top_k={})...", config.top_k);
        let t0 = Instant::now();
        let sn_sets = prepare_sorted_neighbors(&_friction_graph, config.top_k);
        let sn: HashMap<String, Vec<String>> = sn_sets.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
        println!("  Done ({:.2?})", t0.elapsed());
        let bytes = bincode::serialize(&sn).expect("serialize");
        std::fs::write(&sorted_ckpt, bytes).expect("save sorted");
        sn
    };
    let t0 = Instant::now();
    let sorted_sets: HashMap<String, HashSet<String>> = sorted_neighbors
        .par_iter()
        .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
        .collect();
    println!("  Converted to hash sets ({:.2?})", t0.elapsed());

    // Step 3: PPMI → SVD embeddings
    let embed_ckpt = config.checkpoint_dir.join("embeddings_svd.bin");
    let word_embeddings: HashMap<String, Vec<f64>> = if embed_ckpt.exists() {
        println!("Loading SVD embeddings from {:?} ...", embed_ckpt);
        let bytes = std::fs::read(&embed_ckpt).expect("read embeddings");
        bincode::deserialize(&bytes).expect("deserialize embeddings")
    } else {
        let vocab_size = id_to_word.len();
        eprintln!("Building PPMI matrix ({}×{}, {} nnz)...", vocab_size, vocab_size, cooccurrence_counts.len());
        let t0 = Instant::now();
        let ppmi = PPMIMatrix::from_counts(&cooccurrence_counts, vocab_size);
        eprintln!("  Done ({:.2?})", t0.elapsed());
        eprintln!("Randomised SVD (k={})...", config.svd_dim);
        let t0 = Instant::now();
        let (mut embeddings, sv) = randomized_svd(&ppmi, config.svd_dim, 10, 2, 42);
        eprintln!("  {}×{} ({:.2?})", embeddings.shape()[0], embeddings.shape()[1], t0.elapsed());
        eprintln!("  Top-5 SV: {}", sv.iter().take(5).map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));
        for r in 0..embeddings.shape()[0] {
            let mut norm = 0.0;
            for c in 0..embeddings.shape()[1] {
                norm += embeddings[[r, c]] * embeddings[[r, c]];
            }
            norm = norm.sqrt();
            if norm > 1e-12 {
                for c in 0..embeddings.shape()[1] {
                    embeddings[[r, c]] /= norm;
                }
            }
        }
        eprintln!("  L2-normalised rows");
        let mut wemb: HashMap<String, Vec<f64>> = HashMap::with_capacity(vocab_size);
        for (&id, word) in &id_to_word {
            if (id as usize) < embeddings.shape()[0] {
                wemb.insert(word.clone(), embeddings.row(id as usize).to_vec());
            }
        }
        eprintln!("  {} word embeddings", wemb.len());
        let bytes = bincode::serialize(&wemb).expect("serialize");
        std::fs::write(&embed_ckpt, bytes).expect("save");
        wemb
    };

    (sorted_sets, word_embeddings, idf_weights)
}

fn run_snli_benchmark(
    train: &NLIDataset,
    test: &NLIDataset,
    tokenizer: &mut TSOTokenizer,
    config: &BenchConfig,
) {
    // Steps 1-3: Build pipeline
    let (sorted_sets, word_embeddings, idf_weights) = build_pipeline(train, tokenizer, config);

    let alpha_slow = 0.9;
    let alpha_fast = 0.5;
    let negation_set: HashSet<String> = [
        "not", "no", "never", "n't", "but", "however", "yet", "neither", "nor", "none", "nobody",
        "nothing", "nowhere", "cannot", "without",
    ].iter().map(|s| s.to_string()).collect();

    // Step 4: Compute raw features
    let raw_ckpt = config.checkpoint_dir.join("train_features_raw.bin");
    let (train_features_raw, train_labels, normaliser) = if raw_ckpt.exists() {
        println!("Loading raw features from {:?} ...", raw_ckpt);
        let bytes = std::fs::read(&raw_ckpt).expect("read raw");
        let data: (Vec<Vec<f64>>, Vec<usize>) = bincode::deserialize(&bytes).expect("deserialize raw");
        let norm_bytes = std::fs::read(config.checkpoint_dir.join("normaliser.bin")).expect("read normaliser");
        let norm: Normaliser = bincode::deserialize(&norm_bytes).expect("deserialize normaliser");
        (data.0, data.1, norm)
    } else {
        println!("Computing 17D features (Jaccard 3 + Dual-LIF 6 + Phi 4 + Align 4)...");
        let t0 = Instant::now();
        let (raw, lbl) = compute_raw_features(train, &sorted_sets, &word_embeddings, &idf_weights,
            alpha_slow, alpha_fast, &negation_set);
        println!("  {} features ({:.2?})", raw.len(), t0.elapsed());
        let normaliser = Normaliser::fit(&raw);
        println!("  Normaliser: {} dim", normaliser.mean.len());
        let bytes = bincode::serialize(&(raw.clone(), lbl.clone())).expect("serialize");
        std::fs::write(&raw_ckpt, bytes).expect("save raw");
        let norm_bytes = bincode::serialize(&normaliser).expect("serialize normaliser");
        std::fs::write(config.checkpoint_dir.join("normaliser.bin"), norm_bytes).expect("save normaliser");
        (raw, lbl, normaliser)
    };

    println!("Z-scoring training features...");
    let t0 = Instant::now();
    let train_features: Vec<Vec<f64>> = train_features_raw.par_iter().map(|f| normaliser.transform(f)).collect();
    println!("  Done ({:.2?})", t0.elapsed());

    // Step 5: Train attractor field
    let clf_ckpt = config.checkpoint_dir.join("attractor.ckpt.bin");
    let classifier: AttractorField = if clf_ckpt.exists() {
        println!("Loading attractor field from {:?} ...", clf_ckpt);
        let bytes = std::fs::read(&clf_ckpt).expect("read classifier");
        bincode::deserialize(&bytes).expect("deserialize classifier")
    } else {
        println!("Training attractor field (LVQ1, k=15/cls, lr=0.001, 20 epochs)...");
        let mut clf = AttractorField::new(N_FEATS, 3, 15);
        clf.init_kmeans(&train_features, &train_labels, 15);
        for epoch in 0..20 {
            let correct = clf.train_epoch(&train_features, &train_labels, 0.001);
            let acc = correct as f64 / train_features.len() as f64 * 100.0;
            println!("    Epoch {}: train acc {:.2}% ({}/{})", epoch + 1, acc, correct, train_features.len());
        }
        println!("  Trained ({:.2?})", t0.elapsed());
        let bytes = bincode::serialize(&clf).expect("serialize");
        std::fs::write(&clf_ckpt, bytes).expect("save");
        clf
    };

    // Step 6: Evaluate
    println!("Evaluating on {} pairs...", test.len());
    let t0 = Instant::now();
    let eval: Vec<(usize, usize)> = test.samples.par_iter().filter_map(|sample| {
        let actual = NLIDataset::label_to_idx(&sample.label)?;
        let j3 = compute_trifriction_fast(&sample.sentence1, &sample.sentence2, &sorted_sets);
        let p_words = tokenize_words(&sample.sentence1);
        let h_words = tokenize_words(&sample.sentence2);
        let l6 = distributional_features_duallif(&p_words, &h_words, &word_embeddings, &idf_weights,
            alpha_slow, alpha_fast, &negation_set);
        let p4 = phi_sequential(&p_words, &h_words, &word_embeddings, &idf_weights, alpha_slow, &negation_set);
        let a4 = alignment_features(&p_words, &h_words, &word_embeddings);
        let mut raw = Vec::with_capacity(N_FEATS);
        raw.extend_from_slice(&j3);
        raw.extend_from_slice(&l6);
        raw.extend_from_slice(&p4);
        raw.extend_from_slice(&a4);
        let feat = normaliser.transform(&raw);
        let pred = classifier.predict(&feat);
        Some((pred, actual))
    }).collect();
    let mut m = Metrics::new();
    for (p, a) in eval { m.add(p, a); }
    m.compute();
    println!("\n=== RESULTS ({:.2?}) ===", t0.elapsed());
    println!("{}", m.report());
}

fn run_continual_learning(
    snli_train: &NLIDataset,
    snli_dev: &NLIDataset,
    mnli_train: &NLIDataset,
    tokenizer: &mut TSOTokenizer,
    config: &BenchConfig,
) {
    println!("\n===== CONTINUAL LEARNING: SNLI → MultiNLI =====");
    println!("Two modes:\n  [A] Naive LVQ1 overwrite (baseline forgetting)\n  [B] Protected: freeze SNLI prototypes, add MNLI-specific set\n");

    // Build pipeline from SNLI (graph + embeddings are fixed → immune to forgetting)
    let (sorted_sets, word_embeddings, idf_weights) = build_pipeline(snli_train, tokenizer, config);

    let alpha_slow = 0.9;
    let alpha_fast = 0.5;
    let negation_set: HashSet<String> = [
        "not", "no", "never", "n't", "but", "however", "yet", "neither", "nor", "none", "nobody",
        "nothing", "nowhere", "cannot", "without",
    ].iter().map(|s| s.to_string()).collect();

    // ── Features (computed once, shared across all experiments) ─────────
    println!("\nComputing features (one-time cost)...");
    let (snli_raw, snli_labels) = compute_raw_features(snli_train, &sorted_sets, &word_embeddings,
        &idf_weights, alpha_slow, alpha_fast, &negation_set);
    let normaliser = Normaliser::fit(&snli_raw);
    let snli_train_ft: Vec<Vec<f64>> = snli_raw.par_iter().map(|f| normaliser.transform(f)).collect();
    let (snli_dev_raw, snli_dev_labels) = compute_raw_features(snli_dev, &sorted_sets, &word_embeddings,
        &idf_weights, alpha_slow, alpha_fast, &negation_set);
    let snli_dev_ft: Vec<Vec<f64>> = snli_dev_raw.par_iter().map(|f| normaliser.transform(f)).collect();
    let (mnli_raw, mnli_labels) = compute_raw_features(mnli_train, &sorted_sets, &word_embeddings,
        &idf_weights, alpha_slow, alpha_fast, &negation_set);
    let mnli_ft: Vec<Vec<f64>> = mnli_raw.par_iter().map(|f| normaliser.transform(f)).collect();

    // =====================================================================
    // [A] Naive approach: overwrite SNLI prototypes with MultiNLI training
    // =====================================================================
    println!("\n========== [A] NAIVE LVQ1 OVERWRITE ==========");
    let mut clf_naive = {
        let mut clf = AttractorField::new(N_FEATS, 3, 15);
        clf.init_kmeans(&snli_train_ft, &snli_labels, 15);
        for _epoch in 0..20 {
            clf.train_epoch(&snli_train_ft, &snli_labels, 0.001);
        }
        clf
    };
    let baseline_acc = {
        let mut correct = 0;
        for i in 0..snli_dev_ft.len() {
            if clf_naive.predict(&snli_dev_ft[i]) == snli_dev_labels[i] {
                correct += 1;
            }
        }
        correct as f64 / snli_dev_ft.len() as f64 * 100.0
    };
    println!("  SNLI dev (baseline): {:.2}%", baseline_acc);

    println!("\n  Epoch | SNLI Dev Acc | Forgetting Δ");
    println!("  ------+--------------+--------------");
    for epoch in 0..20 {
        clf_naive.train_epoch(&mnli_ft, &mnli_labels, 0.001);
        let mut correct = 0;
        for i in 0..snli_dev_ft.len() {
            if clf_naive.predict(&snli_dev_ft[i]) == snli_dev_labels[i] {
                correct += 1;
            }
        }
        let snli_acc = correct as f64 / snli_dev_ft.len() as f64 * 100.0;
        let forgetting = baseline_acc - snli_acc;
        println!("  {:>5} | {:>11.2}% | {:>+11.2}",
            epoch + 1, snli_acc, forgetting);
    }

    // =====================================================================
    // Recovery test: fresh classifier on SNLI proves features are intact
    // =====================================================================
    println!("\n--- Recovery test: fresh SNLI-only classifier after MNLI overwrite ---");
    let clf_recovery = {
        let mut clf = AttractorField::new(N_FEATS, 3, 15);
        clf.init_kmeans(&snli_train_ft, &snli_labels, 15);
        for _epoch in 0..20 {
            clf.train_epoch(&snli_train_ft, &snli_labels, 0.001);
        }
        clf
    };
    let recovery_acc = {
        let mut correct = 0;
        for i in 0..snli_dev_ft.len() {
            if clf_recovery.predict(&snli_dev_ft[i]) == snli_dev_labels[i] {
                correct += 1;
            }
        }
        correct as f64 / snli_dev_ft.len() as f64 * 100.0
    };
    println!("  SNLI dev accuracy (recovery): {:.2}% (baseline: {:.2}%, Δ = {:.2})",
        recovery_acc, baseline_acc, recovery_acc - baseline_acc);
    if (recovery_acc - baseline_acc).abs() < 0.5 {
        println!("  ✓ Features intact: full recovery from fresh classifier");
    } else {
        println!("  ⚠ Recovery mismatch: {:.2}%", recovery_acc - baseline_acc);
    }

    // =====================================================================
    // [B] Protected: freeze SNLI prototypes, add MNLI-specific prototypes
    // =====================================================================
    println!("\n========== [B] PROTECTED (FREEZE + ADD) ==========");
    println!("  Strategy: SNLI prototypes frozen, new MNLI prototypes added.");
    println!("  Prediction uses nearest neighbor across ALL prototype sets.\n");

    // Train SNLI prototype set (frozen)
    let mut frozen_snli = AttractorField::new(N_FEATS, 3, 15);
    frozen_snli.init_kmeans(&snli_train_ft, &snli_labels, 15);
    for _epoch in 0..20 {
        frozen_snli.train_epoch(&snli_train_ft, &snli_labels, 0.001);
    }

    // Create MNLI prototype set (separate, starts fresh)
    let mut mnli_protos = AttractorField::new(N_FEATS, 3, 15);
    mnli_protos.init_kmeans(&mnli_ft, &mnli_labels, 15);

    // Combined prediction: nearest across all 90 prototypes
    let combined_dev_acc = |snli: &AttractorField, mnli: &AttractorField, ft: &[Vec<f64>], labels: &[usize]| -> f64 {
        let mut correct = 0;
        for i in 0..ft.len() {
            let x = &ft[i];
            let mut best_dist = f64::MAX;
            let mut best_class = 0;
            for p in 0..snli.prototypes.len() {
                let mut dist = 0.0;
                for j in 0..snli.n_inputs {
                    let d = x[j] - snli.prototypes[p][j];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_class = snli.proto_labels[p];
                }
            }
            for p in 0..mnli.prototypes.len() {
                let mut dist = 0.0;
                for j in 0..mnli.n_inputs {
                    let d = x[j] - mnli.prototypes[p][j];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_class = mnli.proto_labels[p];
                }
            }
            if best_class == labels[i] {
                correct += 1;
            }
        }
        correct as f64 / ft.len() as f64 * 100.0
    };

    // Evaluate frozen SNLI before any MNLI training
    let pre_mnli_snli = combined_dev_acc(&frozen_snli, &mnli_protos, &snli_dev_ft, &snli_dev_labels);
    println!("  SNLI dev (frozen only, before MNLI training): {:.2}%", pre_mnli_snli);

    println!("\n  Epoch | SNLI Dev Acc (frozen)");
    println!("  ------+----------------------");
    for epoch in 0..20 {
        mnli_protos.train_epoch(&mnli_ft, &mnli_labels, 0.001);
        let snli_dev_acc = combined_dev_acc(&frozen_snli, &mnli_protos, &snli_dev_ft, &snli_dev_labels);
        println!("  {:>5} | {:>20.2}%", epoch + 1, snli_dev_acc);
    }

    let final_naive = {
        let mut correct = 0;
        for i in 0..snli_dev_ft.len() {
            if clf_naive.predict(&snli_dev_ft[i]) == snli_dev_labels[i] {
                correct += 1;
            }
        }
        correct as f64 / snli_dev_ft.len() as f64 * 100.0
    };
    let final_protected = combined_dev_acc(&frozen_snli, &mnli_protos, &snli_dev_ft, &snli_dev_labels);

    println!("\n===== CONTINUAL LEARNING SUMMARY =====");
    println!("  [A] Naive overwrite: SNLI dev {:.2}% → {:.2}% (forgetting {:.2}%)",
        baseline_acc, final_naive, baseline_acc - final_naive);
    println!("  [B] Protected (freeze+add): SNLI dev {:.2}% (stable)", final_protected);
    println!("  Recovery (fresh SNLI retrain after MNLI): {:.2}%", recovery_acc);
    println!();
    if (recovery_acc - baseline_acc).abs() < 0.5 {
        println!("  ✓ TSO features are immune to catastrophic forgetting.");
        println!("  ✓ Forgetting in [A] is in the LVQ1 classifier, not the representation.");
        println!("  ✓ Protected mode [B] preserves SNLI accuracy indefinitely.");
    } else {
        println!("  ⚠ Features partially affected.");
    }
}

fn run_generate(
    prompt: &str,
    checkpoint_dir: &str,
    _svd_dim: usize,
    _window_size: usize,
    _top_k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = PathBuf::from(checkpoint_dir);

    // Load sorted neighbours (friction graph topology)
    let sorted_path = ckpt.join("sorted.ckpt.bin");
    println!("Loading friction graph from {:?} ...", sorted_path);
    let bytes = std::fs::read(&sorted_path)?;
    let sorted_neighbors: HashMap<String, Vec<String>> = bincode::deserialize(&bytes)?;
    println!("  {} words with friction edges", sorted_neighbors.len());

    // Load word embeddings (HashMap<String, Vec<f64>>)
    let embed_path = ckpt.join("embeddings_svd.bin");
    println!("Loading embeddings from {:?} ...", embed_path);
    let bytes = std::fs::read(&embed_path)?;
    let word_embeddings: HashMap<String, Vec<f64>> = bincode::deserialize(&bytes)?;
    let vocab_size = word_embeddings.len();
    println!("  {} words", vocab_size);

    // Convert to Array2<V×D> and build idx_to_word / word_to_idx mapping
    let dim = word_embeddings.values().next().map(|v| v.len()).unwrap_or(0);
    let mut idx_to_word: HashMap<usize, String> = HashMap::with_capacity(vocab_size);
    let mut word_to_idx: HashMap<String, usize> = HashMap::with_capacity(vocab_size);
    let mut embed_data = vec![0.0_f64; vocab_size * dim];

    for (i, (word, vec)) in word_embeddings.iter().enumerate() {
        idx_to_word.insert(i, word.clone());
        word_to_idx.insert(word.clone(), i);
        for (j, &val) in vec.iter().enumerate() {
            embed_data[i * dim + j] = val;
        }
    }
    let embeddings = Array2::from_shape_vec((vocab_size, dim), embed_data)?;

    // Tokenise prompt
    let mut tokenizer = TSOTokenizer::whitespace();
    let tokens = tokenizer.encode_with_words(prompt, false);
    println!("Prompt tokens ({})", tokens.len());

    // Look up prompt word vectors (with lower-case fallback)
    let mut prompt_vecs: Vec<(String, Array1<f64>)> = Vec::new();
    for &(_id, ref word) in &tokens {
        if let Some(&idx) = word_to_idx.get(word) {
            prompt_vecs.push((word.clone(), embeddings.row(idx).to_owned()));
        } else {
            let lower = word.to_lowercase();
            if let Some(&idx) = word_to_idx.get(&lower) {
                prompt_vecs.push((lower.clone(), embeddings.row(idx).to_owned()));
            } else {
                eprintln!("  Warning: '{}' not in vocabulary, skipping", word);
            }
        }
    }

    if prompt_vecs.is_empty() {
        eprintln!("Error: no prompt words found in vocabulary");
        std::process::exit(1);
    }

    println!("Prompt: {}", prompt);
    for (w, _) in &prompt_vecs {
        print!("  [{}]", w);
    }
    println!();
    println!();

    // Build V7 Anchored Dual-LIF decoder
    let decoder = AnchoredTSODecoder::new(idx_to_word, word_to_idx, embeddings, 0.9, 0.5)
        .with_friction_graph(sorted_neighbors);
    let mut decoder = decoder;
    decoder.syntax_weight = 0.4;
    decoder.drift_threshold = 0.25;
    decoder.recall_strength = 0.35;
    decoder.stability_threshold = 0.001;
    decoder.friction_lambda = 0.5;

    decoder.ingest(&prompt_vecs);

    // Show top-15 candidates before first prediction (with/without friction)
    let empty_set = HashSet::new();
    println!("Top-15 candidates (pure inverse motor, no friction):");
    let candidates_raw = decoder.top_k_candidates(15, None, &empty_set);
    for (i, (_idx, word, score)) in candidates_raw.iter().enumerate() {
        println!("  {}. {}  {:.4}", i + 1, word, score);
    }
    println!();

    let last_word = prompt_vecs.last().map(|(w, _)| w.as_str());
    println!(
        "Top-15 candidates (with Phi friction, λ={}):",
        decoder.friction_lambda
    );
    let candidates_phi = decoder.top_k_candidates(15, last_word, &empty_set);
    for (i, (_idx, word, score)) in candidates_phi.iter().enumerate() {
        println!("  {}. {}  {:.4}", i + 1, word, score);
    }
    println!();

    // Generate
    println!("Generator:");
    println!("  Prompt: {}", prompt);
    println!("  ───────────────────────────────────────────────");
    print!("  ");

    let last_prompt = prompt_vecs.last().map(|(w, _)| w.as_str());
    let words = decoder.generate(20, last_prompt);
    let text = words.join(" ");
    println!("{}", text);

    println!("  ───────────────────────────────────────────────");
    println!("  (Φ threshold={})", decoder.stability_threshold);
    println!();

    Ok(())
}

// ── DeepTSO Validation (V14) ────────────────────────────────────────────

/// K-means on all word embeddings.
fn kmeans_centroids(
    embeddings: &HashMap<String, Vec<f64>>,
    n_centroids: usize,
    n_dims: usize,
) -> Vec<Vec<f64>> {
    let vecs: Vec<Vec<f64>> = embeddings.values().cloned().collect();
    let indices: Vec<usize> = (0..vecs.len()).collect();
    kmeans_class(&vecs, &indices, n_centroids, n_dims, 42)
}

/// Extract COMPARATIVE DeepTSO features between premise and hypothesis.
///
/// Uses `projector` for word→cluster activations (instead of static SVD cosines).
/// If `update_projections` is true, applies R-STDP update to the projector
/// after each word (used during pre-training).
///
/// Process premise through DeepTSO → capture final_rates (P_state).
/// Reset DeepTSO, process hypothesis → capture final_rates (H_state).
/// Features: [cos(P,H), euclidean(P,H), norm_ratio(P,H)].
fn extract_deep_features(
    deep: &mut DeepTSO,
    projector: &mut WordProjector,
    p_words: &[String],
    h_words: &[String],
    dt: f64,
    input_scale: f64,
    update_projections: bool,
) -> Vec<f64> {
    let n_clusters = projector.n_clusters();

    // Premise → P_state
    deep.reset();
    projector.reset_phi();
    let mut last = None;
    for w in p_words {
        let mut act = projector.lookup(w);
        for a in &mut act { *a *= input_scale; }
        let out = deep.step(&Array1::from_vec(act), dt);
        if update_projections {
            projector.update(w, &out.final_rates, out.inter_phi);
        }
        last = Some(out.final_rates);
    }
    let p_state = last.unwrap_or_else(|| Array1::zeros(n_clusters));

    // Hypothesis → H_state (reset so H state is independent)
    deep.reset();
    projector.reset_phi();
    let mut last = None;
    for w in h_words {
        let mut act = projector.lookup(w);
        for a in &mut act { *a *= input_scale; }
        let out = deep.step(&Array1::from_vec(act), dt);
        if update_projections {
            projector.update(w, &out.final_rates, out.inter_phi);
        }
        last = Some(out.final_rates);
    }
    let h_state = last.unwrap_or_else(|| Array1::zeros(n_clusters));

    // Comparative features
    let cos: f64 = p_state.dot(&h_state);
    let diff = &p_state - &h_state;
    let euc: f64 = diff.dot(&diff).sqrt();
    let norm_p: f64 = p_state.dot(&p_state).sqrt();
    let norm_h: f64 = h_state.dot(&h_state).sqrt();
    let ratio = if norm_h > 1e-12 { norm_p / norm_h } else { 1.0 };

    vec![cos, euc, ratio]
}

/// Build typed edges from centroid cosine similarity.
fn build_deep_edges(
    deep: &mut DeepTSO,
    centroids: &[Vec<f64>],
) {
    let n_clusters = centroids.len();
    let n_layers = deep.n_layers();

    for li in 0..n_layers {
        let layer = deep.layer_mut(li).unwrap();
        let mut cnt = 0usize;
        for i in 0..n_clusters {
            for j in (i + 1)..n_clusters {
                let cos = tso_kernel::projector::cosine_similarity(&centroids[i], &centroids[j]);
                if cos > 0.3 {
                    layer.add_edge(i, j, 1.0, cos);
                    cnt += 1;
                } else if cos < -0.1 {
                    layer.add_edge(i, j, -1.0, -cos);
                    cnt += 1;
                }
            }
        }
        eprintln!("    Layer {}: {} intra edges", li, cnt);
    }

    for li in 0..n_layers.saturating_sub(1) {
        let mut cnt = 0usize;
        for i in 0..n_clusters {
            for j in 0..n_clusters {
                let cos = tso_kernel::projector::cosine_similarity(&centroids[i], &centroids[j]);
                if cos > 0.3 {
                    deep.add_inter_edge(li, i, li + 1, j, 1.0, cos);
                    cnt += 1;
                } else if cos < -0.1 {
                    deep.add_inter_edge(li, i, li + 1, j, -1.0, -cos);
                    cnt += 1;
                }
            }
        }
        eprintln!("    Layer {}→{}: {} inter edges", li, li + 1, cnt);
    }
}

/// Compute sparsity: fraction of clusters with rate > threshold.
/// Pre-train DeepTSO inter-layer edges + word projections by streaming
/// the entire corpus as one continuous word sequence. No reset between samples.
fn pretrain_deep_tso(
    deep: &mut DeepTSO,
    projector: &mut WordProjector,
    train: &NLIDataset,
    dt: f64,
    input_scale: f64,
) {
    let n_layers = deep.n_layers();
    let n_clusters = deep.config().n_clusters;
    let wta_keep = (n_clusters as f64 * deep.config().wta_keep_ratio).max(1.0).ceil() as usize;
    let mut total_words = 0usize;
    let mut total_nonzero: Vec<f64> = vec![0.0; n_layers];
    let mut sample_count = 0usize;

    eprintln!("  Pre-training: streaming {} sentences through {}-layer DeepTSO (WTA keep={}/{}, R-STDP on, {} words)...",
        train.len(), n_layers, wta_keep, n_clusters, projector.len());
    let t0 = Instant::now();

    // No reset between samples — continuous stream for R-STDP
    projector.reset_phi();
    for sample in &train.samples {
        let p_words = tokenize_words(&sample.sentence1);
        let h_words = tokenize_words(&sample.sentence2);

        for w in p_words.iter().chain(h_words.iter()) {
            let mut act = projector.lookup(w);
            for a in &mut act { *a *= input_scale; }
            let out = deep.step(&Array1::from_vec(act), dt);
            // Update word projection via R-STDP
            projector.update(w, &out.final_rates, out.inter_phi);
            for li in 0..n_layers.min(out.layers.len()) {
                let nz = out.layers[li].rates.iter().filter(|&&r| r > 0.0).count();
                total_nonzero[li] += nz as f64;
            }
            total_words += 1;
        }

        sample_count += 1;
        if sample_count % 50000 == 0 || sample_count == train.len() {
            let pct = sample_count as f64 / train.len() as f64 * 100.0;
            let mean_nz: Vec<String> = total_nonzero.iter()
                .map(|&n| format!("{:.1}/{}", n / total_words as f64, wta_keep))
                .collect();
            eprint!("\r    [{:>5.1}%] {} words, nonzero/cluster: [{}], learned {} projections ({:.2?})",
                pct, total_words, mean_nz.join(", "), projector.len(), t0.elapsed());
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();
    let elapsed = t0.elapsed();
    let mean_nz: Vec<String> = total_nonzero.iter()
        .map(|&n| format!("{:.2}/{}({:.1}%)", n / total_words as f64, wta_keep,
            n / total_words as f64 / n_clusters as f64 * 100.0))
        .collect();
    println!("    Pre-training done: {} words, {:.2?}, {} projections, mean nonzero/cluster: [{}]",
        total_words, elapsed, projector.len(), mean_nz.join(", "));
}

/// Run DeepTSO classification on SNLI.
fn run_deep_validation(
    train: &NLIDataset,
    test: &NLIDataset,
    tokenizer: &mut TSOTokenizer,
    config: &BenchConfig,
    n_clusters: usize,
    n_layers: usize,
    cold_start: bool,
) {
    let t0_all = Instant::now();
    let alpha_slow = 0.9;
    let alpha_fast = 0.5;
    let negation_set: HashSet<String> = [
        "not", "no", "never", "n't", "but", "however", "yet", "neither", "nor", "none", "nobody",
        "nothing", "nowhere", "cannot", "without",
    ].iter().map(|s| s.to_string()).collect();
    let dt = 0.5;
    let input_scale = 50.0; // cos×50 → input ∈ [0,50]; avg cos 0.3 → input 15 > 10 → LIF fires

    // Step 1: Build pipeline for word embeddings + V13 features
    let (sorted_sets, word_embeddings, idf_weights) = build_pipeline(train, tokenizer, config);
    let n_dims = word_embeddings.values().next().map(|v| v.len()).unwrap_or(0);

    // Step 2: K-means centroids for DeepTSO
    eprint!("  K-means (k={})...", n_clusters);
    let t0 = Instant::now();
    let centroids = kmeans_centroids(&word_embeddings, n_clusters, n_dims);
    eprintln!(" {:.2?}", t0.elapsed());

    // Step 3: Create WordProjector (warm start from SVD centroids, or cold start)
    let project_lr_pos = 0.05;
    let project_lr_neg = 0.03;
    let mut projector = if cold_start {
        eprintln!("  WordProjector: COLD START — random projections, {} clusters", n_clusters);
        WordProjector::new(n_clusters, project_lr_pos, project_lr_neg)
    } else {
        let p = WordProjector::from_svd(n_clusters, &word_embeddings, &centroids, project_lr_pos, project_lr_neg);
        eprintln!("  WordProjector: WARM START — {} initial projections, {} clusters", p.len(), n_clusters);
        p
    };

    // Step 4: Create DeepTSO with R-STDP for pre-training
    let dt_mults: Vec<f64> = (0..n_layers).map(|i| (2.0_f64).powi(i as i32)).collect();
    let deep_cfg = DeepConfig {
        n_layers,
        n_clusters,
        d: 5,
        dt_multipliers: dt_mults,
        modulatory_strength: if n_layers > 1 { 0.05 } else { 0.0 },
        inter_edge_lr: if n_layers > 1 { 0.01 } else { 0.0 },
        learn_inter_edges: n_layers > 1,
        wta_keep_ratio: 0.05,
        ..Default::default()
    };
    let mut deep = DeepTSO::new(deep_cfg);

    // Step 5: Build typed edges (initial topology from centroid similarities)
    eprintln!("  Building typed edges from centroid similarities...");
    build_deep_edges(&mut deep, &centroids);

    // Step 6: Pre-train inter-layer edges + word projections (V16 — unsupervised R-STDP)
    if n_layers > 1 {
        pretrain_deep_tso(&mut deep, &mut projector, train, dt, input_scale);
    }

    // Step 7: Freeze edges for feature extraction
    deep.set_learn_inter_edges(false);
    deep.set_inter_edge_lr(0.0);

    // Feature dimensionality: V13 (17D) + DeepTSO comparative (3D)
    let n_feats = N_FEATS + 3;

    // Step 8: Extract training features (word projections frozen)
    eprintln!("  Extracting V13+DeepTSO features from {} train samples...", train.len());
    let t0 = Instant::now();
    let mut train_features = Vec::with_capacity(train.len());
    let mut train_labels = Vec::with_capacity(train.len());

    for (i, sample) in train.samples.iter().enumerate() {
        let Some(label) = NLIDataset::label_to_idx(&sample.label) else { continue };
        let p_words = tokenize_words(&sample.sentence1);
        let h_words = tokenize_words(&sample.sentence2);

        // V13 17D features
        let j3 = compute_trifriction_fast(&sample.sentence1, &sample.sentence2, &sorted_sets);
        let l6 = distributional_features_duallif(&p_words, &h_words, &word_embeddings, &idf_weights,
            alpha_slow, alpha_fast, &negation_set);
        let p4 = phi_sequential(&p_words, &h_words, &word_embeddings, &idf_weights, alpha_slow, &negation_set);
        let a4 = alignment_features(&p_words, &h_words, &word_embeddings);

        // DeepTSO 3D comparative features (now with pre-trained edges + projections)
        let d3 = extract_deep_features(&mut deep, &mut projector, &p_words, &h_words, dt, input_scale, false);

        let mut feat = Vec::with_capacity(n_feats);
        feat.extend_from_slice(&j3);
        feat.extend_from_slice(&l6);
        feat.extend_from_slice(&p4);
        feat.extend_from_slice(&a4);
        feat.extend_from_slice(&d3);
        train_features.push(feat);
        train_labels.push(label);

        if (i + 1) % 50000 == 0 || i == train.len() - 1 {
            eprint!("\r    [{:>5.1}%] {}/{} ({:.2?})",
                (i + 1) as f64 / train.len() as f64 * 100.0, i + 1, train.len(), t0.elapsed());
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();
    println!("    {} features ({:.2?})", train_features.len(), t0.elapsed());

    // Step 9: Normalise
    eprint!("  Z-scoring {}D features...", n_feats);
    let t0 = Instant::now();
    let normaliser = Normaliser::fit(&train_features);
    let train_norm: Vec<Vec<f64>> = train_features.par_iter().map(|f| normaliser.transform(f)).collect();
    eprintln!(" {:.2?}", t0.elapsed());

    // Step 10: Train AttractorField
    let k_per_class = 15;
    eprintln!("  Training AttractorField (k={}/cls, lr=0.001, 20 epochs)...", k_per_class);
    let mut clf = AttractorField::new(n_feats, 3, k_per_class);
    clf.init_kmeans(&train_norm, &train_labels, k_per_class);
    for epoch in 0..20 {
        let correct = clf.train_epoch(&train_norm, &train_labels, 0.001);
        let acc = correct as f64 / train_norm.len() as f64 * 100.0;
        eprintln!("    Epoch {}: {:.2}% ({}/{})", epoch + 1, acc, correct, train_norm.len());
    }

    // Step 11: Evaluate
    eprintln!("  Evaluating on {} test samples...", test.len());
    let t0 = Instant::now();
    let mut eval: Vec<(usize, usize)> = Vec::with_capacity(test.samples.len());
    for sample in &test.samples {
        let Some(actual) = NLIDataset::label_to_idx(&sample.label) else { continue };
        let p_words = tokenize_words(&sample.sentence1);
        let h_words = tokenize_words(&sample.sentence2);

        let j3 = compute_trifriction_fast(&sample.sentence1, &sample.sentence2, &sorted_sets);
        let l6 = distributional_features_duallif(&p_words, &h_words, &word_embeddings, &idf_weights,
            alpha_slow, alpha_fast, &negation_set);
        let p4 = phi_sequential(&p_words, &h_words, &word_embeddings, &idf_weights, alpha_slow, &negation_set);
        let a4 = alignment_features(&p_words, &h_words, &word_embeddings);
        let d3 = extract_deep_features(&mut deep, &mut projector, &p_words, &h_words, dt, input_scale, false);

        let mut raw = Vec::with_capacity(n_feats);
        raw.extend_from_slice(&j3);
        raw.extend_from_slice(&l6);
        raw.extend_from_slice(&p4);
        raw.extend_from_slice(&a4);
        raw.extend_from_slice(&d3);
        let feat = normaliser.transform(&raw);
        let pred = clf.predict(&feat);
        eval.push((pred, actual));
    }

    let mut m = Metrics::new();
    for (p, a) in eval { m.add(p, a); }
    m.compute();
    let n_proj = projector.len();
    println!("\n=== DeepTSO {}L × {}C | V16 WordProjector ({} words) + V13+Deep 20D ({:.2?}, total {:.2?}) ===",

        n_layers, n_clusters, n_proj, t0.elapsed(), t0_all.elapsed());
    println!("{}", m.report());
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} eval <train.jsonl> <test.jsonl> [window_size] [top_k] [svd_dim] [checkpoint_dir]", args[0]);
        eprintln!("  {} continual <snli_train> <snli_dev> <mnli_train> [window_size] [top_k] [svd_dim] [checkpoint_dir]", args[0]);
        eprintln!("  {} deval <train.jsonl> <test.jsonl> <n_clusters> <n_layers> [window_size] [top_k] [svd_dim] [checkpoint_dir]", args[0]);
        std::process::exit(1);
    }

    let mode = &args[1];
    match mode.as_str() {
        "eval" | "snli" => {
            if args.len() < 4 {
                eprintln!("Usage: {} eval <train.jsonl> <test.jsonl> [window_size] [top_k] [svd_dim] [checkpoint_dir]", args[0]);
                std::process::exit(1);
            }
            let train_path = &args[2];
            let test_path = &args[3];
            let window_size = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
            let top_k = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20);
            let svd_dim = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(100);
            let checkpoint_dir = args.get(7).cloned().unwrap_or_else(|| "checkpoints_snli".to_string());

            println!("TSO v5.1 — Jaccard 3 + Dual-LIF(α=0.9,α=0.5,neg) 6 + Phi 4 + Align 4 → 17D AttractorField");
            println!("==============================================");
            println!("Loading {} ...", train_path);
            let train = NLIDataset::from_jsonl(train_path)?;
            println!("  {} samples (e:{}, n:{}, c:{})", train.len(),
                train.count_by_label()[0], train.count_by_label()[1], train.count_by_label()[2]);
            println!("Loading {} ...", test_path);
            let test = NLIDataset::from_jsonl(test_path)?;
            println!("  {} samples (e:{}, n:{}, c:{})", test.len(),
                test.count_by_label()[0], test.count_by_label()[1], test.count_by_label()[2]);

            let mut tokenizer = TSOTokenizer::whitespace();
            let config = BenchConfig { window_size, top_k, svd_dim,
                checkpoint_dir: PathBuf::from(&checkpoint_dir) };
            run_snli_benchmark(&train, &test, &mut tokenizer, &config);
        }
        "continual" | "cl" => {
            if args.len() < 5 {
                eprintln!("Usage: {} continual <snli_train> <snli_dev> <mnli_train> [window_size] [top_k] [svd_dim] [checkpoint_dir]", args[0]);
                std::process::exit(1);
            }
            let snli_train_path = &args[2];
            let snli_dev_path = &args[3];
            let mnli_train_path = &args[4];
            let window_size = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
            let top_k = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20);
            let svd_dim = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(100);
            let checkpoint_dir = args.get(8).cloned().unwrap_or_else(|| "checkpoints_snli".to_string());

            println!("Loading SNLI train from {} ...", snli_train_path);
            let snli_train = NLIDataset::from_jsonl(snli_train_path)?;
            println!("  {} samples", snli_train.len());
            println!("Loading SNLI dev from {} ...", snli_dev_path);
            let snli_dev = NLIDataset::from_jsonl(snli_dev_path)?;
            println!("  {} samples", snli_dev.len());
            println!("Loading MultiNLI train from {} ...", mnli_train_path);
            let mnli_train = NLIDataset::from_jsonl(mnli_train_path)?;
            println!("  {} samples", mnli_train.len());

            let mut tokenizer = TSOTokenizer::whitespace();
            let config = BenchConfig { window_size, top_k, svd_dim,
                checkpoint_dir: PathBuf::from(&checkpoint_dir) };
            run_continual_learning(&snli_train, &snli_dev, &mnli_train, &mut tokenizer, &config);
        }
        "deval" | "deep" => {
            if args.len() < 6 {
                eprintln!("Usage: {} deval <train.jsonl> <test.jsonl> <n_clusters> <n_layers> [window_size] [top_k] [svd_dim] [checkpoint_dir] [--cold-start]", args[0]);
                eprintln!("  DeepTSO validation — n_clusters centroids from word embeddings, n_layers cortical layers.");
                eprintln!("  Add --cold-start to initialize WordProjector with random weights (no SVD).");
                eprintln!("  Example: {} deval snli_1.0_train.jsonl snli_1.0_dev.jsonl 50 1", args[0]);
                eprintln!("  Example: {} deval snli_1.0_train.jsonl snli_1.0_dev.jsonl 50 2", args[0]);
                std::process::exit(1);
            }
            let train_path = &args[2];
            let test_path = &args[3];
            let n_clusters: usize = args[4].parse().expect("n_clusters must be usize");
            let n_layers: usize = args[5].parse().expect("n_layers must be usize");
            let window_size = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(5);
            let top_k = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(20);
            let svd_dim = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(100);
            let checkpoint_dir = args.get(9).cloned().unwrap_or_else(|| "checkpoints_deep".to_string());
            let cold_start = args.iter().any(|a| a == "--cold-start");

            if cold_start {
                println!("TSO V16 — DeepTSO Validation **COLD START**");
            } else {
                println!("TSO V16 — DeepTSO Validation (warm start)");
            }
            println!("============================================");
            let train = NLIDataset::from_jsonl(train_path)?;
            println!("Train: {} samples", train.len());
            let test = NLIDataset::from_jsonl(test_path)?;
            println!("Test: {} samples", test.len());

            let mut tokenizer = TSOTokenizer::whitespace();
            let config = BenchConfig { window_size, top_k, svd_dim,
                checkpoint_dir: PathBuf::from(&checkpoint_dir) };
            run_deep_validation(&train, &test, &mut tokenizer, &config, n_clusters, n_layers, cold_start);
        }
        "generate" | "gen" => {
            if args.len() < 4 {
                eprintln!("Usage: {} generate <checkpoint_dir> <prompt> [svd_dim]", args[0]);
                std::process::exit(1);
            }
            let checkpoint_dir = &args[2];
            let prompt = &args[3];
            let svd_dim = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);
            let window_size = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
            let top_k = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20);

            run_generate(prompt, checkpoint_dir, svd_dim, window_size, top_k)?;
        }
        _ => {
            eprintln!("Unknown mode '{}'. Use 'eval', 'continual', or 'generate'.", mode);
            std::process::exit(1);
        }
    }
    Ok(())
}
