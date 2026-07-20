//! TSO Scalability Benchmark
//!
//! Tests the Rust kernel at increasing vocabulary sizes (V).
//! Generates a synthetic corpus of N sequences of length L, each word
//! sampled uniformly from [0, V), then runs:
//!   co-occurrence counting → PPMI matrix → Randomized SVD
//!
//! These are the three heaviest steps. Friction graph and sorted neighbors
//! are O(V) and never bottleneck.
//!
//! Usage: cargo run --release -p tso-bench --bin scalability [V] [N] [L] [window] [svd_dim]

use std::collections::HashMap;
use std::time::Instant;

use rand::Rng;
use rayon::prelude::*;

use tso_nlp::distributional::{randomized_svd, PPMIMatrix};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vocab_size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let n_seqs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let seq_len: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let window: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let svd_dim: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);

    let total_tokens = n_seqs as u64 * seq_len as u64;

    println!("══════════════════════════════════════════════");
    println!("  TSO Scalability Benchmark");
    println!("══════════════════════════════════════════════");
    println!("  Vocabulary size (V):  {:>12}", vocab_size);
    println!("  Sequences (N):        {:>12}", n_seqs);
    println!("  Seq length (L):       {:>12}", seq_len);
    println!("  Total tokens:         {:>12}", total_tokens);
    println!("  Window size:          {:>12}", window);
    println!("  SVD dim:              {:>12}", svd_dim);
    println!("──────────────────────────────────────────────");
    println!("  Step");
    println!("──────────────────────────────────────────────");

    // ── Step 1: Generate synthetic sequences ────────────────────────────
    let t0 = Instant::now();
    let sequences: Vec<Vec<(u32, String)>> = (0..n_seqs)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            (0..seq_len)
                .map(|_| {
                    let id: u32 = rng.gen_range(0..vocab_size as u32);
                    (id, String::new())
                })
                .collect()
        })
        .collect();
    let gen_time = t0.elapsed();
    println!("  Generate {total_tokens} tokens           {gen_time:>8.2?}");

    // ── Step 2: Co-occurrence counting ──────────────────────────────────
    println!("  → Counting co-occurrences (window={window}, parallel)...");
    let t0 = Instant::now();
    let counts = count_cooccurrences_par(&sequences, window);
    let count_time = t0.elapsed();
    let unique_pairs = counts.len();
    let max_possible = vocab_size as u64 * vocab_size as u64;
    let avg_deg = unique_pairs as f64 / vocab_size.max(1) as f64;
    println!("  Co-occurrence counting                {count_time:>8.2?}");
    println!("    unique pairs: {unique_pairs} / {max_possible} ({:.4}%)",
        unique_pairs as f64 / max_possible as f64 * 100.0);
    println!("    avg degree:   {avg_deg:.2}");

    // Estimate memory
    let counts_mb = (unique_pairs * (8 + 8 + 8)) as f64 / 1_048_576.0; // key (8) + val (8) + hash overhead (8)
    println!("    est. memory:  {counts_mb:.1} MB (HashMap)");
    drop(sequences);
    let _ = t0;

    // ── Step 3: PPMI matrix ─────────────────────────────────────────────
    println!("  → Building PPMI CSR matrix ({vocab_size}×{vocab_size}, {unique_pairs} nnz)...");
    let t0 = Instant::now();
    let ppmi = PPMIMatrix::from_counts(&counts, vocab_size);
    let ppmi_time = t0.elapsed();
    let nnz = ppmi.values.len();
    let csr_mb = (ppmi.row_ptr.len() * 8 + ppmi.col_ind.len() * 4 + ppmi.values.len() * 8) as f64 / 1_048_576.0;
    println!("  PPMI CSR matrix                       {ppmi_time:>8.2?}");
    println!("    shape: {vocab_size}×{vocab_size}, nnz: {nnz}");
    println!("    CSR memory: {csr_mb:.1} MB (row_ptr + col_ind + val)");
    println!("    density: {:.6}%", nnz as f64 / (vocab_size as u64 * vocab_size as u64) as f64 * 100.0);
    drop(counts);

    // ── Step 4: Randomized SVD ──────────────────────────────────────────
    let oversampling = 10;
    let power_iter = 2;
    let svd_est = (vocab_size as f64 / 100_000.0) * 85.0; // rough estimate based on V=100K timing
    println!("  → Randomized SVD (k={svd_dim}, oversampling={oversampling}, power_iter={power_iter})...");
    println!("    estimated time: ~{:.0}s (V={vocab_size}, ~{:.1}M nnz)", svd_est, nnz as f64 / 1_000_000.0);
    let t0 = Instant::now();
    let (mut embeddings, sv) = randomized_svd(&ppmi, svd_dim, oversampling, power_iter, 42);
    let svd_time = t0.elapsed();
    let n_rows = embeddings.shape()[0];
    let n_cols = embeddings.shape()[1];
    drop(ppmi);
    println!("  ✓ SVD done in {svd_time:?}");

    // ── Post-processing (from original code) ────────────────────────────
    let t1 = Instant::now();
    for r in 0..n_rows {
        let mut norm = 0.0;
        for c in 0..n_cols {
            norm += embeddings[[r, c]] * embeddings[[r, c]];
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for c in 0..n_cols {
                embeddings[[r, c]] /= norm;
            }
        }
    }
    let post_time = t1.elapsed();

    let embed_mb = (n_rows * n_cols * 8) as f64 / 1_048_576.0;
    println!("  Randomized SVD (k={svd_dim}) + L2 norm      {svd_time:>8.2?}");
    println!("    output: {n_rows}×{n_cols} ({embed_mb:.1} MB f64)");
    println!("    L2 norm: {post_time:?}");
    println!("    Top-5 SV: {}", sv.iter().take(5).map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", "));

    // ── Summary ─────────────────────────────────────────────────────────
    let total_time = gen_time + count_time + ppmi_time + svd_time;
    println!("──────────────────────────────────────────────");
    println!("  Total (steps 1-4):               {total_time:>8.2?}");
    println!("══════════════════════════════════════════════");
}

fn count_cooccurrences_par(
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

    let n_maps = per_thread.len();
    let t0 = Instant::now();
    let mut global = HashMap::new();
    for local in per_thread {
        for (pair, count) in local {
            *global.entry(pair).or_insert(0) += count;
        }
    }
    eprintln!("  (merged {n_maps} local maps in {:.2?})", t0.elapsed());
    global
}
