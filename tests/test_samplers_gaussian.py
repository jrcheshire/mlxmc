"""Every sampler must recover the canonical correlated Gaussian within Standard tolerances:
  - mean:  |estimate - true| < 4 * SE     (SE = sample_std / sqrt(ESS))
  - std:   relative error < 5%
  - corr:  |estimate - true| < 0.03
This is the cross-sampler correctness gate; mixing speed is not asserted here.
"""
import mlx.core as mx
import numpy as np
import pytest

from mlxmc import run_chain, run_ensemble, run_hmc, run_nuts, run_phmc, warmup
from mlxmc.targets import GAUSSIAN_MU, GAUSSIAN_SIGMA, gaussian_logp
from util import per_dim_stats, structured

TRUE_STD = np.sqrt(np.diag(GAUSSIAN_SIGMA))                       # [5, 1]
TRUE_CORR = GAUSSIAN_SIGMA[0, 1] / (TRUE_STD[0] * TRUE_STD[1])    # 0.9


def check_gaussian(chain_structured):
    """chain_structured: (T, N, D). Assert mean/std/corr recovery within Standard tolerances."""
    st = per_dim_stats(chain_structured)
    for d in range(2):
        assert abs(st[d]["mean"] - GAUSSIAN_MU[d]) < 4 * st[d]["se"], \
            f"dim {d} mean {st[d]['mean']:.3f} vs {GAUSSIAN_MU[d]} (4*SE={4*st[d]['se']:.3f})"
        assert abs(st[d]["std"] - TRUE_STD[d]) / TRUE_STD[d] < 0.05, \
            f"dim {d} std {st[d]['std']:.3f} vs {TRUE_STD[d]} (rel err {abs(st[d]['std']-TRUE_STD[d])/TRUE_STD[d]:.3f})"
    flat = np.asarray(chain_structured).reshape(-1, 2)
    corr = np.corrcoef(flat.T)[0, 1]
    assert abs(corr - TRUE_CORR) < 0.03, f"corr {corr:.3f} vs {TRUE_CORR}"


def test_ensemble():
    key = mx.random.key(0)
    key, ki = mx.random.split(key)
    e0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    samples, acc = run_ensemble(gaussian_logp, e0, n_steps=2000, burn=500, key=key)
    check_gaussian(structured(samples, 2000 - 500, 200))
    # Liveness band: not a dead sampler (~0) nor degenerate (~1). G&W stretch-move acceptance
    # rises as dimension falls, so 2-D sits high (~0.7); this is a guardrail, not an accuracy gate.
    assert 0.15 < acc < 0.85, f"ensemble acceptance {acc:.3f} outside liveness band"


def test_hmc_identity():
    key = mx.random.key(1)
    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    samples, acc = run_hmc(gaussian_logp, q0, n_steps=1500, burn=500, eps=0.15, n_leap=40, key=key)
    assert acc > 0.6, f"HMC acceptance {acc:.3f} below target"
    check_gaussian(structured(samples, 1500 - 500, 200))


def test_preconditioned_hmc():
    key = mx.random.key(2)
    Minv = GAUSSIAN_SIGMA                                   # M^{-1} = Sigma
    Mhalf = np.linalg.cholesky(np.linalg.inv(GAUSSIAN_SIGMA))
    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    chain = run_phmc(gaussian_logp, q0, 1500, 500, 0.7, 6, key, Minv, Mhalf)
    check_gaussian(chain)


def test_warmup_run_chain():
    key = mx.random.key(3)
    k_init, k_warm, k_sample = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(200, 2), key=k_init) * 5.0
    q_last, eps, Minv = warmup(gaussian_logp, q0, n_warmup=600, n_leap=8, key=k_warm)
    chain = run_chain(gaussian_logp, q_last, n_steps=1500, burn=0, eps=eps,
                      Minv_np=Minv, key=k_sample, n_leap=8)
    check_gaussian(chain)


def test_nuts():
    key = mx.random.key(4)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(200, 2), key=k_init) * 5.0
    q_last, eps, Minv = warmup(gaussian_logp, q0, n_warmup=600, n_leap=8, key=k_warm)
    chain, mean_depth, max_depth = run_nuts(gaussian_logp, q_last, n_samples=800,
                                            eps=eps, Minv_np=Minv, key=k_nuts)
    check_gaussian(chain)
    # NUTS-specific: full covariance within 5% Frobenius (the exactness check from the worklog).
    flat = np.asarray(chain).reshape(-1, 2)
    cov_rel = np.linalg.norm(np.cov(flat.T) - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert cov_rel < 0.05, f"NUTS cov Frobenius rel err {cov_rel:.3f}"
