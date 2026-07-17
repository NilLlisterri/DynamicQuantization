"""Sparse Ternary Compression (STC), Sattler et al., "Robust and Communication-
Efficient Federated Learning From Non-i.i.d. Data", IEEE TNNLS 31(9), 2020.

Implements Algorithm 1 (top-k sparsification + ternarization) and the
communication-cost accounting of Section VI-D (sign bit + Golomb-optimal
position encoding + one shared 32-bit magnitude per compressed tensor).
"""
import math
import numpy as np


def sparsify_ternarize(x: np.ndarray, p: float):
    """STC, Algorithm 1. Always selects exactly k = max(round(p * n), 1) elements of
    x by magnitude via argpartition; k itself is a deterministic function of (n, p)
    alone, matching Algorithm 1's definition where k is computed before the data is
    inspected, but which specific elements are chosen among exact magnitude ties is
    left to argpartition's internal selection algorithm and is not guaranteed to
    follow index order (ties are not expected in practice with real-valued weight
    deltas). Replaces the selected elements with +-mu (their mean magnitude)
    according to sign, and zeroes the rest. Returns (ternary_tensor, k), with k = 0
    for an all-zero input (nothing to sparsify)."""
    n = len(x)
    abs_x = np.abs(x)
    if not np.any(abs_x):
        return np.zeros_like(x), 0
    k = min(max(int(round(p * n)), 1), n)
    top_idx = np.arange(n) if k == n else np.argpartition(abs_x, n - k)[n - k:]
    mu = float(np.mean(abs_x[top_idx]))
    result = np.zeros_like(x)
    result[top_idx] = mu * np.sign(x[top_idx])
    return result, k


def golomb_bits_per_position(p: float) -> float:
    """Optimal average Golomb code length per encoded gap for a geometric
    distribution with sparsity rate p (Sattler et al., Eq. 22). Validated against
    the paper's own worked example: p=0.01 -> 8.38 bits (matches Section VI-D)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    phi = (1 + 5 ** 0.5) / 2
    ratio = math.log(phi - 1) / math.log(1 - p)
    b_star = 1 + math.ceil(math.log2(ratio))
    b_star = max(b_star, 1)
    return b_star + 1 / (1 - (1 - p) ** (2 ** b_star))


def stc_bits(k: int, p: float) -> int:
    """Communication cost, in bits, of one STC-compressed tensor: one sign bit
    plus a Golomb-coded position per nonzero, plus a shared 32-bit magnitude
    (Sattler et al., Section VI-D)."""
    if k == 0:
        return 32
    b_pos = golomb_bits_per_position(p)
    return int(round(k * (1 + b_pos) + 32))
