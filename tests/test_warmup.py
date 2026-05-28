"""Warmup must recover the mass matrix it's estimating: M^{-1} = Cov(q) should match the
true Sigma to < 3% Frobenius error, and the dual-averaged step size should land in a sane
range. (The worklog saw < 1% with more chains; 3% is the Standard CI tolerance.)
"""
import mlx.core as mx
import numpy as np

from mlxmc import nuts_warmup, warmup
from mlxmc.targets import GAUSSIAN_SIGMA, gaussian_logp


def test_warmup_recovers_sigma():
    key = mx.random.key(7)
    k_init, k_warm = mx.random.split(key)
    q0 = mx.random.normal(shape=(400, 2), key=k_init) * 5.0
    q_last, eps, Minv = warmup(gaussian_logp, q0, n_warmup=800, n_leap=8, key=k_warm)

    rel = np.linalg.norm(Minv - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert rel < 0.03, f"estimated M^-1 Frobenius rel err {rel:.3f}\n{Minv}\nvs\n{GAUSSIAN_SIGMA}"
    assert 0.05 < eps < 2.0, f"tuned eps {eps:.3f} outside sane range"
    assert q_last.shape == (400, 2)


def test_nuts_warmup_recovers_sigma():
    """NUTS-specific warmup: same Standard tolerance (<3% Frobenius) since the windowed M
    estimator is identical to the fixed-L warmup's. The dual-averaging target differs (NUTS's
    leaf-mean accept vs HMC's whole-trajectory accept), so the tuned eps need not match the
    fixed-L value -- only the M estimate is directly comparable."""
    key = mx.random.key(8)
    k_init, k_warm = mx.random.split(key)
    q0 = mx.random.normal(shape=(400, 2), key=k_init) * 5.0
    q_last, eps, Minv = nuts_warmup(gaussian_logp, q0, n_warmup=800, key=k_warm)

    rel = np.linalg.norm(Minv - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert rel < 0.03, f"estimated M^-1 Frobenius rel err {rel:.3f}\n{Minv}\nvs\n{GAUSSIAN_SIGMA}"
    assert 0.05 < eps < 2.0, f"tuned eps {eps:.3f} outside sane range"
    assert q_last.shape == (400, 2)
