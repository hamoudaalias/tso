/// Action selection helpers — eliminates inline efe-vs-cerebellum branches.
use ndarray::Array1;
use crate::cerebellum::Cerebellum;
#[cfg(feature = "active-inference")]
use crate::efe;
use crate::model::GenerativeModel;

/// Pick action using either Cerebellum (TD) or EFE scoring.
/// Keeps the if-else in one place so the cognitive cycle reads as a single call.
pub fn select_action(
    decision_state: &Array1<f64>,
    cerebellum: &mut Cerebellum,
    efe_weight: f64,
    use_utility: bool,
    use_info_gain: bool,
    n_actions: usize,
    model: Option<&GenerativeModel>,
) -> usize {
    let use_efe = efe_weight > 0.0;
    if use_efe {
        if let Some(m) = model {
            let qs: Vec<Array1<f64>> = m.D.iter()
                .map(|d| d.clone())
                .collect();
            return efe::select_best_action(&qs, &m.A, &m.B, &m.C,
                &(0..n_actions).collect::<Vec<_>>(),
                use_utility, use_info_gain)
                .0;
        }
    }
    // Default: Cerebellum TD
    let logits = cerebellum.forward_logits(decision_state);
    let mut action_id = 0;
    let mut best_logit = logits[0];
    for (i, l) in logits.iter().enumerate() {
        if *l > best_logit { best_logit = *l; action_id = i; }
    }
    if rand::random::<f64>() < cerebellum.epsilon {
        action_id = rand::random::<usize>() % n_actions;
    }
    cerebellum.mark(decision_state, action_id);
    action_id
}
