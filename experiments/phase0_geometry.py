"""
Phase 0 — Double Mapping Geometry Validation.

Verifies Lemma 1: projecting conflicting concepts into disjoint subspaces
eliminates phi while preserving implication dot products.

This is the exact same computation as double_mapping_test.py but
using the TSOCore kernel.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tso_kernel.operators import TopographicOperator


def run_phase0():
    print("=" * 60)
    print("  Phase 0 — Double Mapping Geometry (Kernel)")
    print("=" * 60)

    d = 5
    np.random.seed(42)

    vectors = {
        'chat': np.random.randn(d).astype(np.float32),
        'animal': np.random.randn(d).astype(np.float32),
        'chien': np.random.randn(d).astype(np.float32),
        'contexte': np.random.randn(d).astype(np.float32),
    }
    for k in vectors:
        vectors[k] /= np.linalg.norm(vectors[k])

    print(f"\n  Vecteurs initiaux (dim={d}):")
    for k, v in vectors.items():
        print(f"    {k:10s} ||v||={np.linalg.norm(v):.4f}")

    dot_chat_animal = float(vectors['chat'] @ vectors['animal'])
    dot_chat_chien = float(vectors['chat'] @ vectors['chien'])
    dot_chat_ctx = float(vectors['chat'] @ vectors['contexte'])
    dot_chien_ctx = float(vectors['chien'] @ vectors['contexte'])
    dot_animal_ctx = float(vectors['animal'] @ vectors['contexte'])

    print(f"\n  Produits scalaires dans R^{d}:")
    print(f"    chat·animal = {dot_chat_animal:.4f}  (implication)")
    print(f"    chat·chien  = {dot_chat_chien:.4f}  (exclusion)")
    print(f"    chat·ctx    = {dot_chat_ctx:.4f}  (contexte)")
    print(f"    chien·ctx   = {dot_chien_ctx:.4f}  (contexte)")
    print(f"    animal·ctx  = {dot_animal_ctx:.4f}  (implication contexte)")

    phi_before = abs(dot_chat_chien)
    print(f"\n  Tension AVANT Double Mapping: Φ = |chat·chien| = {phi_before:.4f}")

    new_vecs, nd = TopographicOperator.double_mapping(
        vectors, ['chat', 'chien'], 'contexte', d
    )

    print(f"\n  Après Double Mapping (dim={nd}):")

    new_dot_chat_chien = float(new_vecs['chat'] @ new_vecs['chien'])
    new_dot_chat_ctx = float(new_vecs['chat'] @ new_vecs['contexte'])
    new_dot_chien_ctx = float(new_vecs['chien'] @ new_vecs['contexte'])

    print(f"    chat·chien  = {new_dot_chat_chien:.6f}  (exclusion → devrait être 0)")
    print(f"    chat·ctx    = {new_dot_chat_ctx:.4f}  (contexte préservé)")
    print(f"    chien·ctx   = {new_dot_chien_ctx:.4f}  (contexte préservé)")

    phi_after = abs(new_dot_chat_chien)
    print(f"\n  Tension APRÈS Double Mapping: Φ = |chat·chien| = {phi_after:.6f}")

    print("\n  VÉRIFICATIONS:")
    checks = 0
    if abs(new_dot_chat_chien) < 1e-6:
        print("    ✓ Exclusion (chat·chien) = 0")
        checks += 1
    if abs(new_dot_chat_ctx - dot_chat_ctx) < 1e-6:
        print("    ✓ Contexte chat·ctx préservé ({:.4f} → {:.4f})".format(
            dot_chat_ctx, new_dot_chat_ctx))
        checks += 1
    if abs(new_dot_chien_ctx - dot_chien_ctx) < 1e-6:
        print("    ✓ Contexte chien·ctx préservé ({:.4f} → {:.4f})".format(
            dot_chien_ctx, new_dot_chien_ctx))
        checks += 1

    print(f"\n  Résultat: {checks}/3 vérifications passées")
    if checks == 3:
        print("  *** LEMME 1 CONFIRMÉ ***")
    else:
        print("  ÉCHEC — revoir l'opérateur")


if __name__ == "__main__":
    run_phase0()
