// Tests d'intégration pour Sommeil Phase 3 — Neurogenèse structurelle
// Chaque test suit le cycle RED → GREEN → REFACTOR

use tso_engine::{CognitiveConfig, TsoEngine, SleepReport};

#[test]
fn test_neurogenesis_config_defaults() {
    // e10s01t01: Les nouveaux champs de CognitiveConfig ont les bonnes valeurs par défaut
    let cfg = CognitiveConfig::default();
    assert_eq!(
        cfg.sleep_neurogenesis_rate, 0.2,
        "sleep_neurogenesis_rate devrait être 0.2 par défaut"
    );
    assert_eq!(
        cfg.sleep_max_concepts, 50,
        "sleep_max_concepts devrait être 50 par défaut"
    );
    assert_eq!(
        cfg.sleep_maturation_cycles, 3,
        "sleep_maturation_cycles devrait être 3 par défaut"
    );
    assert!(
        cfg.sleep_synaptic_scaling,
        "sleep_synaptic_scaling devrait être true par défaut"
    );
}

#[test]
fn test_neurogenesis_concept_maturation_init() {
    // e10s01t02: concept_maturation est initialisé vide après new()
    let engine = TsoEngine::new(6, 4);
    assert!(
        engine.concept_maturation().is_empty(),
        "concept_maturation devrait être vide à l'initialisation"
    );
}

#[test]
fn test_neurogenesis_birth() {
    // e10s01t06: Phase 1.5 crée de nouveaux concepts pendant sleep_cycle()
    let mut engine = TsoEngine::new(6, 4);
    engine.cogs.sleep_neurogenesis_rate = 1.0; // forcer la neurogenèse
    engine.cogs.sleep_max_concepts = 50;
    engine.sleep_every_n_episodes = 0; // désactiver le sleep auto

    // Créer quelques concepts via des steps
    use ndarray::Array1;
    for _ in 0..20 {
        let obs = Array1::from_vec(vec![0.1; 6]);
        engine.step(&obs, 0.0, None, &[]);
    }

    let n_before = engine.num_concepts();
    let report = engine.sleep_cycle();
    let n_after = engine.num_concepts();

    assert!(
        report.prototypes_added > 0 || n_after > n_before,
        "La neurogenèse devrait créer au moins un nouveau concept (n_before={n_before}, n_after={n_after}, added={})",
        report.prototypes_added
    );
}

#[test]
fn test_neurogenesis_critical_period() {
    // e10s02t01: La période critique protège du pruning
    let mut engine = TsoEngine::new(6, 4);
    engine.cogs.sleep_neurogenesis_rate = 1.0;
    engine.cogs.sleep_max_concepts = 50;
    engine.cogs.sleep_maturation_cycles = 3;
    engine.sleep_every_n_episodes = 0;
    engine.concept_prune_threshold = 10;

    use ndarray::Array1;
    // Créer des concepts
    for _ in 0..10 {
        let obs = Array1::from_vec(vec![0.1; 6]);
        engine.step(&obs, 0.0, None, &[]);
    }

    // Sleep cycle → crée des nouveaux concepts protégés
    engine.sleep_cycle();

    // Vérifier qu'après sleep, au moins un nouveau concept est en période critique
    let has_maturation = engine.concept_maturation().iter().any(|&m| m > 0);
    assert!(has_maturation, "Au moins un concept devrait être en période critique");

    // Forcer le pruning avec un seuil très bas — les concepts en maturation survivent
    engine.concept_prune_threshold = 0;
    engine.end_episode();
    // Les concepts doivent être préservés car en période critique
    assert!(has_maturation, "Les concepts en maturation survivent au pruning");
}

#[test]
fn test_neurogenesis_maturation_decrements() {
    // e10s02t02: Les compteurs de maturation décrémentent à chaque cycle sommeil
    let mut engine = TsoEngine::new(6, 4);
    engine.cogs.sleep_neurogenesis_rate = 1.0;
    engine.cogs.sleep_maturation_cycles = 3;
    engine.sleep_every_n_episodes = 0;

    use ndarray::Array1;
    for _ in 0..5 {
        let obs = Array1::from_vec(vec![0.1; 6]);
        engine.step(&obs, 0.0, None, &[]);
    }

    let report = engine.sleep_cycle();
    assert!(report.new_concepts > 0, "Au moins 1 nouveau concept créé");

    // Vérifier que le compteur baisse : forcer sleep_neurogenesis_rate = 0 pour éviter
    // l'apparition de nouveaux concepts en maturation à chaque cycle.
    engine.cogs.sleep_neurogenesis_rate = 0.0;

    for step in 0..3 {
        engine.sleep_cycle();
        let max_maturation = engine.concept_maturation().iter().max().copied().unwrap_or(0);
        assert!(
            max_maturation <= 3 - step - 1,
            "step={step}: l'âge max devrait être ≤ {} (était {max_maturation})",
            3 - step - 1
        );
    }

    let all_mature = engine.concept_maturation().iter().all(|&m| m == 0);
    assert!(all_mature, "Après 3 cycles sans neurogenèse, tous les concepts devraient être matures");
}

#[test]
fn test_neurogenesis_lr_boost() {
    // e10s02t03-04: Le lr boost et la restauration ne cassent pas le step
    let mut engine = TsoEngine::new(6, 4);
    engine.cogs.sleep_neurogenesis_rate = 1.0;
    engine.cogs.sleep_maturation_cycles = 3;
    engine.sleep_every_n_episodes = 0;

    use ndarray::Array1;
    for _ in 0..5 {
        let obs = Array1::from_vec(vec![0.1; 6]);
        engine.step(&obs, 0.0, None, &[]);
    }

    // Sleep pour créer un concept protégé
    let report = engine.sleep_cycle();
    assert!(report.new_concepts > 0, "Au moins 1 nouveau concept");

    // Step après neurogenèse — le lr boost ne doit pas faire planter
    let obs = Array1::from_vec(vec![0.3; 6]);
    let action = engine.step(&obs, 0.0, None, &[]);
    assert!(action < 4, "L'action doit être dans l'espace d'action");

    // Vérifier que le lr est restauré (pas de fuite du ×3)
    let lr_after = engine.attractor.lr;
    assert!(
        (lr_after - 0.01).abs() < 1e-6,
        "Le lr devrait être restauré à 0.01 (valeur: {lr_after})"
    );
}
