/// Résultats d'un run multi-seed.
#[derive(Clone, Debug)]
pub struct SeedResults {
    pub seeds: usize,
    pub mean: f64,
    pub std: f64,
    pub scores: Vec<f64>,
}

impl SeedResults {
    pub fn from_scores(scores: Vec<f64>) -> Self {
        let seeds = scores.len();
        let mean = scores.iter().sum::<f64>() / seeds as f64;
        let var = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / seeds as f64;
        SeedResults { seeds, mean, std: var.sqrt(), scores }
    }

    /// Intervalle de confiance 95% (approximation normale)
    pub fn ci95(&self) -> (f64, f64) {
        let se = self.std / (self.seeds as f64).sqrt();
        (self.mean - 1.96 * se, self.mean + 1.96 * se)
    }

    /// Cohen's d vs baseline (pooled std)
    pub fn cohens_d(&self, baseline: &SeedResults) -> f64 {
        let pooled = ((self.std.powi(2) + baseline.std.powi(2)) / 2.0).sqrt();
        if pooled < 1e-10 { return 0.0; }
        (self.mean - baseline.mean) / pooled
    }

    /// Welch t-test approximation
    pub fn welch_t(&self, baseline: &SeedResults) -> f64 {
        let se = ((self.std.powi(2) / self.seeds as f64) + (baseline.std.powi(2) / baseline.seeds as f64)).sqrt();
        if se < 1e-10 { return 0.0; }
        (self.mean - baseline.mean) / se
    }

    /// Degrés de liberté Welch-Satterthwaite
    pub fn welch_df(&self, baseline: &SeedResults) -> f64 {
        let s1 = self.std.powi(2) / self.seeds as f64;
        let s2 = baseline.std.powi(2) / baseline.seeds as f64;
        let num = (s1 + s2).powi(2);
        let den = s1.powi(2) / (self.seeds as f64 - 1.0) + s2.powi(2) / (baseline.seeds as f64 - 1.0);
        if den < 1e-10 { return 1.0; }
        num / den
    }
}

/// Run une fonction de benchmark sur N seeds, retourne SeedResults.
pub fn run_bench<F>(n_seeds: usize, mut bench_fn: F) -> SeedResults
where
    F: FnMut() -> f64,
{
    let scores: Vec<f64> = (0..n_seeds).map(|_| bench_fn()).collect();
    SeedResults::from_scores(scores)
}
