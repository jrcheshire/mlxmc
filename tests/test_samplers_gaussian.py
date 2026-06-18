"""Every sampler must recover the canonical correlated Gaussian within Standard tolerances:
  - mean:  |estimate - true| < 4 * SE     (SE = sample_std / sqrt(ESS))
  - std:   relative error < 5%
  - corr:  |estimate - true| < 0.03
This is the cross-sampler correctness gate; mixing speed is not asserted here.
"""
import mlx.core as mx
import numpy as np
import pytest

from mlxmc import (
    Result,
    nuts_warmup,
    nuts_warmup_serial,
    run_chain,
    run_ensemble,
    run_hmc,
    run_nuts,
    run_nuts_serial,
    run_phmc,
    warmup,
)
from mlxmc.targets import GAUSSIAN_MU, GAUSSIAN_SIGMA, gaussian_logp
from util import per_dim_stats

TRUE_STD = np.sqrt(np.diag(GAUSSIAN_SIGMA))                       # [5, 1]
TRUE_CORR = GAUSSIAN_SIGMA[0, 1] / (TRUE_STD[0] * TRUE_STD[1])    # 0.9


def check_gaussian(result):
    """result: a Result. Assert mean/std/corr recovery within Standard tolerances."""
    assert isinstance(result, Result)
    chain_tnd = np.transpose(result.samples, (1, 0, 2))          # (chain,draw,dim) -> (T,N,D)
    st = per_dim_stats(chain_tnd)
    for d in range(2):
        assert abs(st[d]["mean"] - GAUSSIAN_MU[d]) < 4 * st[d]["se"], \
            f"dim {d} mean {st[d]['mean']:.3f} vs {GAUSSIAN_MU[d]} (4*SE={4*st[d]['se']:.3f})"
        assert abs(st[d]["std"] - TRUE_STD[d]) / TRUE_STD[d] < 0.05, \
            f"dim {d} std {st[d]['std']:.3f} vs {TRUE_STD[d]} (rel err {abs(st[d]['std']-TRUE_STD[d])/TRUE_STD[d]:.3f})"
    flat = result.flat
    corr = np.corrcoef(flat.T)[0, 1]
    assert abs(corr - TRUE_CORR) < 0.03, f"corr {corr:.3f} vs {TRUE_CORR}"


def test_ensemble():
    key = mx.random.key(0)
    key, ki = mx.random.split(key)
    e0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    result = run_ensemble(gaussian_logp, e0, n_steps=2000, burn=500, key=key)
    check_gaussian(result)
    assert result.independent_chains is False           # walkers are correlated, not chains
    # Liveness band: not a dead sampler (~0) nor degenerate (~1). G&W stretch-move acceptance
    # rises as dimension falls, so 2-D sits high (~0.7); this is a guardrail, not an accuracy gate.
    assert 0.15 < result.accept_frac < 0.85, f"ensemble acceptance {result.accept_frac:.3f} outside band"


def test_hmc_identity():
    key = mx.random.key(1)
    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    result = run_hmc(gaussian_logp, q0, n_steps=1500, burn=500, eps=0.15, n_leap=40, key=key)
    assert result.accept_frac > 0.6, f"HMC acceptance {result.accept_frac:.3f} below target"
    check_gaussian(result)


def test_preconditioned_hmc():
    key = mx.random.key(2)
    Minv = GAUSSIAN_SIGMA                                   # M^{-1} = Sigma
    Mhalf = np.linalg.cholesky(np.linalg.inv(GAUSSIAN_SIGMA))
    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(200, 2), key=ki) * 5.0
    result = run_phmc(gaussian_logp, q0, 1500, 500, 0.7, 6, key, Minv, Mhalf)
    check_gaussian(result)


def test_warmup_run_chain():
    key = mx.random.key(3)
    k_init, k_warm, k_sample = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(200, 2), key=k_init) * 5.0
    q_last, eps, Minv = warmup(gaussian_logp, q0, n_warmup=600, n_leap=8, key=k_warm)
    result = run_chain(gaussian_logp, q_last, n_steps=1500, burn=0, eps=eps,
                       Minv_np=Minv, key=k_sample, n_leap=8)
    check_gaussian(result)


def test_nuts():
    key = mx.random.key(4)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(200, 2), key=k_init) * 5.0
    q_last, eps, Minv = warmup(gaussian_logp, q0, n_warmup=600, n_leap=8, key=k_warm)
    result = run_nuts(gaussian_logp, q_last, n_samples=800, eps=eps, Minv_np=Minv, key=k_nuts)
    check_gaussian(result)
    # The well-conditioned Gaussian should not produce divergences.
    assert result.n_divergent == 0, f"unexpected divergences on the Gaussian: {result.n_divergent}"
    # NUTS-specific: full covariance within 5% Frobenius (the exactness check from the worklog).
    cov_rel = np.linalg.norm(np.cov(result.flat.T) - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert cov_rel < 0.05, f"NUTS cov Frobenius rel err {cov_rel:.3f}"


def test_nuts_with_nuts_warmup():
    """End-to-end with NUTS-specific tuning: nuts_warmup -> run_nuts -> Gaussian moments
    and cov within the same Standard tolerances as the borrowed-warmup test_nuts."""
    key = mx.random.key(5)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(200, 2), key=k_init) * 5.0
    q_last, eps, Minv = nuts_warmup(gaussian_logp, q0, n_warmup=600, key=k_warm)
    result = run_nuts(gaussian_logp, q_last, n_samples=800, eps=eps, Minv_np=Minv, key=k_nuts)
    check_gaussian(result)
    cov_rel = np.linalg.norm(np.cov(result.flat.T) - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert cov_rel < 0.05, f"NUTS (nuts_warmup) cov Frobenius rel err {cov_rel:.3f}"


def test_nuts_serial():
    """Serial (no-vmap) NUTS recovers the Gaussian -- the path for targets whose grad
    MLX can't vmap over chains (e.g. a conv-net Pad). Same Standard tolerances; fewer
    chains since chains run serially in a host loop."""
    key = mx.random.key(6)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(8, 2), key=k_init) * 5.0
    q_last, eps, Minv = nuts_warmup_serial(gaussian_logp, q0, n_warmup=400, key=k_warm)
    result = run_nuts_serial(gaussian_logp, q_last, n_samples=900, eps=eps, Minv_np=Minv, key=k_nuts)
    check_gaussian(result)
    assert result.n_divergent == 0, f"unexpected divergences on the Gaussian: {result.n_divergent}"
    cov_rel = np.linalg.norm(np.cov(result.flat.T) - GAUSSIAN_SIGMA) / np.linalg.norm(GAUSSIAN_SIGMA)
    assert cov_rel < 0.05, f"serial NUTS cov Frobenius rel err {cov_rel:.3f}"


def test_nuts_serial_fixed_metric():
    """estimate_metric=False holds a known-good metric (Sigma) fixed and tunes only eps --
    the mode the bayes-compsep Laplace-preconditioned subspace uses."""
    key = mx.random.key(7)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(8, 2), key=k_init) * 5.0
    q_last, eps, Minv = nuts_warmup_serial(
        gaussian_logp, q0, n_warmup=300, key=k_warm, minv0=GAUSSIAN_SIGMA, estimate_metric=False)
    assert np.allclose(Minv, GAUSSIAN_SIGMA), "fixed metric should be returned unchanged"
    result = run_nuts_serial(gaussian_logp, q_last, n_samples=900, eps=eps, Minv_np=Minv, key=k_nuts)
    check_gaussian(result)
