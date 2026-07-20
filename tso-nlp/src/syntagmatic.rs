use std::collections::{HashMap, HashSet};

/// Directed bigram transition graph (syntagmatic model).
///
/// Captures P(next | current) for ordered token bigrams observed
/// across the training corpus.  This is a simple Markov chain of
/// order 1 built from BPE‑tokenised sequences.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SyntagmaticGraph {
    /// P(next | current) for each ordered pair (current → next).
    pub transitions: HashMap<u32, HashMap<u32, f64>>,
    /// All observed (current, next) transitions for O(1) membership.
    pub transition_set: HashSet<(u32, u32)>,
}

impl SyntagmaticGraph {
    /// Build a syntagmatic graph from a corpus of token‑ID sequences.
    pub fn from_sequences(sequences: &[Vec<u32>]) -> Self {
        let mut bigram_counts: HashMap<(u32, u32), u64> = HashMap::new();
        let mut token_counts: HashMap<u32, u64> = HashMap::new();

        for seq in sequences {
            for window in seq.windows(2) {
                let a = window[0];
                let b = window[1];
                *bigram_counts.entry((a, b)).or_insert(0) += 1;
                *token_counts.entry(a).or_insert(0) += 1;
            }
        }

        let mut transitions: HashMap<u32, HashMap<u32, f64>> = HashMap::new();
        let mut transition_set: HashSet<(u32, u32)> = HashSet::new();

        for ((a, b), count) in bigram_counts {
            let total = token_counts.get(&a).copied().unwrap_or(1);
            let prob = count as f64 / total as f64;
            transitions.entry(a).or_default().insert(b, prob);
            transition_set.insert((a, b));
        }

        Self {
            transitions,
            transition_set,
        }
    }

    /// Compute the 3 syntagmatic features for a premise–hypothesis pair.
    ///
    /// Returns `[support, conflict, novelty]`:
    ///   * **support** – average P(next | current) of hypothesis bigrams whose
    ///     first token appears in the premise (conditionally probable given
    ///     premise context).
    ///   * **conflict** – fraction of hypothesis bigrams whose first token is
    ///     in the premise but the transition has very low probability.
    ///   * **novelty**  – fraction of hypothesis bigrams absent from the
    ///     global transition graph.
    pub fn compute_features(&self, premise_ids: &[u32], hypothesis_ids: &[u32]) -> [f64; 3] {
        let premise_set: HashSet<u32> = premise_ids.iter().copied().collect();
        let total = hypothesis_ids.len().saturating_sub(1);

        if total == 0 {
            return [0.0, 0.0, 0.0];
        }

        let mut prob_sum = 0.0;
        let mut prob_count = 0;
        let mut conflict_count = 0;
        let mut novelty_count = 0;

        for window in hypothesis_ids.windows(2) {
            let a = window[0];
            let b = window[1];

            if let Some(prob) = self
                .transitions
                .get(&a)
                .and_then(|m| m.get(&b))
            {
                if premise_set.contains(&a) {
                    prob_sum += prob;
                    prob_count += 1;
                    if *prob < 0.01 {
                        conflict_count += 1;
                    }
                }
            } else {
                novelty_count += 1;
            }
        }

        let support = if prob_count > 0 {
            prob_sum / prob_count as f64
        } else {
            0.0
        };
        let conflict = conflict_count as f64 / total as f64;
        let novelty = novelty_count as f64 / total as f64;

        [support, conflict, novelty]
    }
}
