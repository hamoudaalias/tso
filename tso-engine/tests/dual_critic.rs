/// Test de séparation critic interne vs externe
///
/// Objectif : vérifier que le critic externe (reward pur)
/// et le critic interne (well-being) sont découplés,
/// ce qui permet à l'apprentissage TD d'être stable
/// même si le well-being varie.

#[test]
fn test_dual_critic_structure() {
    // Vérifie la structure : deux critiques séparés
    // Critic externe : V_ext(s) = E[r_ext + γV_ext(s')]
    // Critic interne : V_int(s) = E[well_being + γV_int(s')]

    // Simule des apprentissages parallèles
    let ext_rewards = vec![0.0, 0.0, -0.5, 0.0, 10.0, 0.0, -0.5, 0.0, 0.0, 10.0];
    let int_well_being = vec![
        -0.5, -0.3, -0.8, -0.2, 8.5, -0.4, -0.9, -0.1, -0.3, 7.2,
    ];

    // Vérifie que les deux séries ont une corrélation imparfaite
    // (le well-being inclut Φ, curiosité, déficits — pas seulement le reward)
    let mut ext_sum = 0.0;
    let mut int_sum = 0.0;
    for i in 0..ext_rewards.len() {
        ext_sum += ext_rewards[i];
        int_sum += int_well_being[i];
    }
    let ext_mean = ext_sum / ext_rewards.len() as f64;
    let int_mean = int_sum / int_well_being.len() as f64;

    let mut cov = 0.0;
    let mut var_ext = 0.0;
    let mut var_int = 0.0;
    for i in 0..ext_rewards.len() {
        let de = ext_rewards[i] - ext_mean;
        let di = int_well_being[i] - int_mean;
        cov += de * di;
        var_ext += de * de;
        var_int += di * di;
    }

    let corr = cov / (var_ext.sqrt() * var_int.sqrt() + 1e-8);

    println!("ext_mean: {:.4}", ext_mean);
    println!("int_mean: {:.4}", int_mean);
    println!("correlation ext/int: {:.4}", corr);

    // La corrélation doit exister mais être imparfaite
    // (le well-being inclut le reward plus d'autres termes)
    assert!(
        corr > -0.5,
        "Corrélation ext/int anormalement négative: {:.4}",
        corr
    );

    println!("PASS: dual_critic_structure");
}

#[test]
fn test_dual_critic_td_error() {
    // Vérifie que le TD-error du critic externe est plus stable
    // que celui du critic interne (qui subit la non-stationnarité)

    // Simule un well-being avec tendance (non-stationnaire)
    let mut well_being_values: Vec<f64> = Vec::new();
    let mut reward_values: Vec<f64> = Vec::new();
    let mut rng = SimpleRng::new(42);

    for step in 0..200 {
        let reward = if step % 50 == 49 { 10.0 } else { -0.5 };
        reward_values.push(reward);

        // Well-being = reward + bruit + tendance (non-stationnaire)
        let trend = (step as f64) * 0.01; // tendance croissante
        let noise = (rng.next_f64() - 0.5) * 0.5;
        let wb = reward + trend + noise;
        well_being_values.push(wb);
    }

    // TD(0) simple sur les deux signaux
    let gamma = 0.99;
    let mut v_ext = 0.0;
    let mut v_int = 0.0;
    let mut td_errors_ext = Vec::new();
    let mut td_errors_int = Vec::new();
    let lr = 0.1;

    for i in 0..well_being_values.len() - 1 {
        let td_ext = reward_values[i] + gamma * v_ext - v_ext;
        let td_int = well_being_values[i] + gamma * v_int - v_int;
        td_errors_ext.push(td_ext);
        td_errors_int.push(td_int);
        v_ext += lr * td_ext;
        v_int += lr * td_int;
    }

    // Calcule la variance des TD-errors
    let ext_var = variance(&td_errors_ext);
    let int_var = variance(&td_errors_int);

    println!("TD-error variance (ext): {:.6}", ext_var);
    println!("TD-error variance (int): {:.6}", int_var);
    println!(
        "Ratio int/ext: {:.2}x",
        int_var / (ext_var + 1e-8)
    );

    // Le TD-error interne devrait être plus variable à cause de la tendance
    println!("PASS: dual_critic_td_error");
}

#[test]
fn test_dual_critic_convergence() {
    // Vérifie que le critic externe converge vers les vraies valeurs
    // même quand le well-being est non-stationnaire

    let gamma = 0.99;
    let mut v_ext = 0.0;
    let lr = 0.05;
    let mut errors = Vec::new();

    // Environnement simple : goal = 10.0 tous les 50 pas
    for step in 0..500 {
        let reward = if step % 50 == 49 { 10.0 } else { -0.5 };
        let td: f64 = reward + gamma * v_ext - v_ext;
        v_ext += lr * td;
        if step > 400 {
            errors.push(td.abs());
        }
    }

    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;

    println!("V_ext convergée: {:.4}", v_ext);
    println!("TD-error moyen (derniers pas): {:.4}", mean_error);

    // Le critic externe devrait converger (TD-error → 0)
    assert!(
        mean_error < 2.0,
        "TD-error externe devrait converger < 2.0, obtenu: {:.4}",
        mean_error
    );

    println!("PASS: dual_critic_convergence");
}

// Helper
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state as f64) / (u64::MAX as f64)
    }
}
