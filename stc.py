"""Sparse Ternary Compression (STC), Sattler et al., "Robust and Communication-
Efficient Federated Learning From Non-i.i.d. Data", IEEE TNNLS 31(9), 2020.

Implements Algorithm 1 (top-k sparsification + ternarization) and the
communication-cost accounting of Section VI-D (sign bit + Golomb-optimal
position encoding + one shared 32-bit magnitude per compressed tensor).
"""
import math
import numpy as np


def sparsify_ternarize(x: np.ndarray, p: float):
    """STC, Algorithm 1. Keeps the top round(p * n) elements of x by magnitude,
    replaces them with +-mu (their mean magnitude) according to sign, and zeroes
    the rest. Returns (ternary_tensor, k) where k is the number of nonzeros."""
    n = len(x)
    k = max(int(round(p * n)), 1)
    abs_x = np.abs(x)
    if k >= n:
        threshold = 0.0
    else:
        threshold = np.partition(abs_x, n - k)[n - k]
    mask = abs_x >= threshold
    masked = x * mask
    k_actual = int(mask.sum())
    if k_actual == 0:
        return np.zeros_like(x), 0
    mu = np.sum(np.abs(masked)) / k_actual
    return mu * np.sign(masked), k_actual


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
