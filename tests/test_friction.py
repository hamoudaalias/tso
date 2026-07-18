"""
Unit tests for the TSO Kernel — friction computation and operators.
Run with: python -m pytest tests/ -v  (or just python tests/test_friction.py)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tso_kernel.friction import FrictionCalculator
from tso_kernel.operators import TopographicOperator
from tso_kernel.neurons import LIFCluster


def test_friction_zero():
    calc = FrictionCalculator(gamma=0.5, epsilon=0.3)
    rates = np.array([0.8, 0.8])
    edges = [(0, 1, 1, 1.0)]
    phi = calc.compute_phi(rates, edges)
    assert phi == 0.0, f"Expected 0, got {phi}"


def test_friction_positive():
    calc = FrictionCalculator(gamma=0.5, epsilon=0.3)
    rates = np.array([0.1, 0.1])
    edges = [(0, 1, 1, 1.0)]
    phi = calc.compute_phi(rates, edges)
    assert phi > 0.0, f"Expected >0, got {phi}"
    assert abs(phi - 0.49) < 1e-6, f"Expected 0.49, got {phi}"


def test_friction_exclusion():
    calc = FrictionCalculator(gamma=0.5, epsilon=0.3)
    rates = np.array([0.7, 0.7])
    edges = [(0, 1, -1, 1.0)]
    phi = calc.compute_phi(rates, edges)
    assert phi > 0.0, f"Expected >0 for exclusion, got {phi}"
    assert abs(phi - 0.19) < 1e-6, f"Expected 0.19, got {phi}"  # 0.7*0.7 - 0.3 = 0.19


def test_double_mapping_orthogonality():
    d = 5
    np.random.seed(0)
    vectors = {
        'a': np.random.randn(d).astype(np.float32),
        'b': np.random.randn(d).astype(np.float32),
        'ctx': np.random.randn(d).astype(np.float32),
    }
    for k in vectors:
        vectors[k] /= np.linalg.norm(vectors[k])

    new_vecs, _ = TopographicOperator.double_mapping(
        vectors, ['a', 'b'], 'ctx', d
    )

    dot_ab = abs(new_vecs['a'] @ new_vecs['b'])
    assert dot_ab < 1e-6, f"Expected 0, got {dot_ab}"


def test_double_mapping_preserves_context():
    d = 5
    np.random.seed(1)
    vectors = {
        'a': np.random.randn(d).astype(np.float32),
        'b': np.random.randn(d).astype(np.float32),
        'ctx': np.random.randn(d).astype(np.float32),
    }
    for k in vectors:
        vectors[k] /= np.linalg.norm(vectors[k])

    orig_a_ctx = float(vectors['a'] @ vectors['ctx'])
    orig_b_ctx = float(vectors['b'] @ vectors['ctx'])

    new_vecs, _ = TopographicOperator.double_mapping(
        vectors, ['a', 'b'], 'ctx', d
    )

    new_a_ctx = float(new_vecs['a'] @ new_vecs['ctx'])
    new_b_ctx = float(new_vecs['b'] @ new_vecs['ctx'])

    assert abs(new_a_ctx - orig_a_ctx) < 1e-6, "Context dot changed"
    assert abs(new_b_ctx - orig_b_ctx) < 1e-6, "Context dot changed"


def test_conceptual_phi():
    calc = FrictionCalculator()
    W = np.ones((3, 3), dtype=np.float32)
    W[0, 1] = 5.0
    W[0, 2] = 1.0
    phi = calc.conceptual_phi(0, 1, W, 3)
    # p = 5/(5+1+1) = 5/7 ≈ 0.714
    # phi = 1 - 0.714*3 = 1 - 2.143 = -1.143
    expected = 1.0 - (5.0 / 7.0) * 3
    assert abs(phi - expected) < 1e-6, f"Expected {expected}, got {phi}"


def test_lif_step():
    cluster = LIFCluster(5)
    rates_before = cluster.rate
    cluster.step(np.full(5, 20.0))
    assert cluster.rate >= 0
    assert cluster.rate <= 1.0


if __name__ == "__main__":
    test_friction_zero()
    test_friction_positive()
    test_friction_exclusion()
    test_double_mapping_orthogonality()
    test_double_mapping_preserves_context()
    test_conceptual_phi()
    test_lif_step()
    print("  ✓ Tous les tests passent.")
