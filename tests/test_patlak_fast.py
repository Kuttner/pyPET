"""Check that patlak_fast() gives the same Ki and v0 as patlak(), curve by curve.

Run from the repository root:  python tests/test_patlak_fast.py   (or with pytest)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Tissue_compartment_modeling import patlak, patlak_fast  # noqa: E402


def feng_input(t_min):
    # Feng input function (FDG-like), t in minutes, starting at 0.5 min
    t = np.clip(t_min - 0.5, 0, None)
    return (12.0 * t - 1.5 - 0.5) * np.exp(-4.0 * t) + 1.5 * np.exp(-0.2 * t) + 0.5 * np.exp(-0.01 * t)


def two_tcm_irrev(cp, t_min, K1, k2, k3, vB):
    # Irreversible two-tissue model on a fine uniform grid, by convolution
    dt = t_min[1] - t_min[0]
    a = k2 + k3
    h = K1 / a * (k3 + k2 * np.exp(-a * t_min))
    return (1 - vB) * np.convolve(cp, h)[: t_min.size] * dt + vB * cp


def mouse_frames_s():
    # 1 x 30 s, 24 x 5 s, 9 x 20 s, 8 x 300 s (45.5 min); frame mid times in seconds
    d = np.r_[30.0, np.full(24, 5.0), np.full(9, 20.0), np.full(8, 300.0)]
    start = np.r_[0.0, np.cumsum(d)[:-1]]
    return start + d / 2


def synthetic(n_curves=300, seed=0):
    rng = np.random.default_rng(seed)
    t_fine = np.arange(0, 45.5 * 60, 1.0)                    # plasma on a 1 s grid, seconds
    cp_fine = np.clip(feng_input(t_fine / 60), 0, None)
    t_frames = mouse_frames_s()
    curves = []
    for _ in range(n_curves):
        K1, k2, k3, vB = rng.uniform(0.05, 0.6), rng.uniform(0.1, 1.0), rng.uniform(0.01, 0.3), rng.uniform(0, 0.1)
        ct_fine = two_tcm_irrev(cp_fine, t_fine / 60, K1, k2, k3, vB)
        ct = np.interp(t_frames, t_fine, ct_fine)
        curves.append(ct * (1 + 0.05 * rng.standard_normal(ct.size)))
    return cp_fine, t_fine, np.array(curves).T, t_frames


def compare(cp, t_p, ct, t_t, label):
    ki_ref, v0_ref = [], []
    t0 = time.time()
    for i in range(ct.shape[1]):
        ki, v0 = patlak(cp, t_p, ct[:, i], t_t)
        ki_ref.append(ki)
        v0_ref.append(v0)
    t_loop = time.time() - t0
    t0 = time.time()
    ki_b, v0_b = patlak_fast(cp, t_p, ct, t_t)
    t_batch = time.time() - t0
    ki_ref, v0_ref = np.array(ki_ref), np.array(v0_ref)
    rel_ki = np.max(np.abs(ki_b - ki_ref) / np.maximum(np.abs(ki_ref), 1e-12))
    rel_v0 = np.max(np.abs(v0_b - v0_ref) / np.maximum(np.abs(v0_ref), 1e-12))
    print(f"{label}: {ct.shape[1]} curves, max relative difference Ki {rel_ki:.1e}, v0 {rel_v0:.1e}; "
          f"patlak() {t_loop:.2f} s, patlak_fast() {t_batch:.3f} s")
    return rel_ki, rel_v0


def test_patlak_fast_synthetic():
    cp, t_p, ct, t_t = synthetic()
    cp_before = cp.copy()
    rel_ki, rel_v0 = compare(cp, t_p, ct, t_t, "synthetic, separate plasma and tissue time axes")
    assert rel_ki < 1e-9 and rel_v0 < 1e-9
    assert np.array_equal(cp, cp_before), "the input plasma curve was changed"


def test_patlak_fast_single_curve():
    cp, t_p, ct, t_t = synthetic(n_curves=1, seed=1)
    ki, v0 = patlak(cp, t_p, ct[:, 0], t_t)
    ki_b, v0_b = patlak_fast(cp, t_p, ct[:, 0], t_t)
    assert np.isscalar(ki_b) or np.ndim(ki_b) == 0
    assert abs(ki_b / ki - 1) < 1e-9 and abs(v0_b / v0 - 1) < 1e-9


def test_patlak_fast_same_time_axis():
    _, _, ct, t_t = synthetic(n_curves=100, seed=2)
    cp = np.clip(feng_input(t_t / 60), 0, None)
    rel_ki, rel_v0 = compare(cp, t_t, ct, 0, "synthetic, plasma on the tissue frames")
    assert rel_ki < 1e-9 and rel_v0 < 1e-9


if __name__ == "__main__":
    test_patlak_fast_synthetic()
    test_patlak_fast_single_curve()
    test_patlak_fast_same_time_axis()
    print("all tests passed")
