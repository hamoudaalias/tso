//! Tests unitaires du module d'attention spatiale (e04s01).
//!
//! L'attention spatiale applique un gain multiplicatif aux dimensions
//! des moustaches où l'erreur de prédiction épisodique est maximale.
//! Vérifie le comportement dans les cas nominaux et limites.

use ndarray::Array1;
use tso_engine::attention::Attention;

const T: f64 = 0.5; // température par défaut

/// Helper pour créer une perception 4D (moustaches seules)
fn p4(a: f64, b: f64, c: f64, d: f64) -> Array1<f64> {
    Array1::from_vec(vec![a, b, c, d])
}

/// Helper pour créer un prototype 4D
fn proto4(a: f64, b: f64, c: f64, d: f64) -> Array1<f64> {
    Array1::from_vec(vec![a, b, c, d])
}

// ── Test 1 : Pas de prototype prédit → perception inchangée ──────────────

#[test]
fn test_no_predicted_prototype() {
    let attn = Attention::new(T);
    let p = p4(0.5, 0.3, 0.8, 0.1);
    let gated = attn.attend(&p, None);
    // Sans prototype, tous les poids = 1.0 → perception inchangée
    assert!(
        (gated[0] - 0.5).abs() < 1e-10,
        "gated[0]={}, expected 0.5", gated[0]
    );
    assert!(
        (gated[3] - 0.1).abs() < 1e-10,
        "gated[3]={}, expected 0.1", gated[3]
    );
}

// ── Test 2 : Prototype identique à la perception → tous les gains ≈ 1.0 ──

#[test]
fn test_identical_prototype() {
    let attn = Attention::new(T);
    let p = p4(0.5, 0.3, 0.8, 0.1);
    let proto = proto4(0.5, 0.3, 0.8, 0.1);
    let gated = attn.attend(&p, Some(&proto));
    // Tous les softmax égaux → mean ≈ 0.25 → gain = 1.0
    for i in 0..4 {
        assert!(
            (gated[i] - p[i]).abs() < 1e-6,
            "gated[{}]={}, expected {}", i, gated[i], p[i]
        );
    }
}

// ── Test 3 : Une dimension très différente → gain > 1.0 sur cette dim ────

#[test]
fn test_one_dimension_divergent() {
    let attn = Attention::new(T);
    // Perceptions non-nulles pour que le produit gain × perception soit visible
    let p = p4(0.4, 0.3, 0.3, 0.3);
    let proto = proto4(1.0, 0.3, 0.3, 0.3);
    let gated = attn.attend(&p, Some(&proto));
    // diff[0] = 0.6 (seule dimension différente) → softmax > 0.25 pour dim 0
    // gain = softmax/mean → gain[0] > 1.0 → gated[0] > 0.4
    assert!(
        gated[0] > p[0],
        "gated[0]={} should be > {} (gain > 1)", gated[0], p[0]
    );
    assert!(
        gated[0] > gated[1],
        "gated[0]={} should be > gated[1]={}", gated[0], gated[1]
    );
}

// ── Test 4 : Température très faible → softmax saturé ────────────────────

#[test]
fn test_low_temperature() {
    let attn = Attention::new(0.01); // T très basse → softmax quasi one-hot
    // perception[0] non-nulle pour que le produit gain × perception soit > 0
    let p = p4(0.5, 0.1, 0.2, 0.3);
    let proto = proto4(1.0, 0.2, 0.3, 0.3);
    let gated = attn.attend(&p, Some(&proto));
    // diff[0] = 0.5, diff[1..3] ≈ 0.1/0.0
    // À T=0.01, softmax est quasi one-hot sur dim 0 → gain >> 1
    assert!(
        gated[0] > 0.8,
        "gated[0]={} should be > 0.8 (high gain on divergent dim)", gated[0]
    );
}

// ── Test 5 : Toutes les dimensions identiques → déjà vu dans test_identical
//             mais vérifie aussi le cas où toutes les diffs sont 0

#[test]
fn test_all_zero_diffs() {
    let attn = Attention::new(T);
    let p = p4(0.7, 0.7, 0.7, 0.7);
    let proto = proto4(0.7, 0.7, 0.7, 0.7);
    let gated = attn.attend(&p, Some(&proto));
    for i in 0..4 {
        assert!(
            (gated[i] - 0.7).abs() < 1e-6,
            "gated[{}]={}, expected 0.7", i, gated[i]
        );
    }
}

// ── Test 6 : Perception avec dimensions non-moustache (BFS, food_sense) ──

#[test]
fn test_non_whisker_dims_pass_through() {
    let attn = Attention::new(T);
    let p = Array1::from_vec(vec![0.2, 0.4, 0.6, 0.8, 0.5, 0.3]);
    let proto = proto4(0.5, 0.5, 0.5, 0.5);
    let gated = attn.attend(&p, Some(&proto));
    // Les dims 4 et 5 (non-moustaches) doivent passer inchangées
    assert!(
        (gated[4] - 0.5).abs() < 1e-10,
        "gated[4]={}, expected 0.5", gated[4]
    );
    assert!(
        (gated[5] - 0.3).abs() < 1e-10,
        "gated[5]={}, expected 0.3", gated[5]
    );
}

// ── Test 7 : Valeurs clampées dans [0, 2] ────────────────────────────────

#[test]
fn test_clamp_range() {
    let attn = Attention::new(T);
    // Si le gain est très élevé, le résultat doit être clampé à 2.0
    let p = p4(0.01, 0.5, 0.5, 0.5);
    let proto = proto4(1.0, 0.5, 0.5, 0.5);
    let gated = attn.attend(&p, Some(&proto));
    for i in 0..4 {
        assert!(
            gated[i] >= 0.0 && gated[i] <= 2.0,
            "gated[{}]={} out of [0, 2]", i, gated[i]
        );
    }
}
