"""NUTS now surfaces the divergences it always computed but used to discard.

The centered funnel is the project's documented failure case: a global mass matrix can't
follow the position-dependent scale, so NUTS gets stuck at the neck and the leapfrog diverges
in fp32. That bias was previously invisible (the mean looked fine). The non-centered
reparametrization removes the geometry, so it should be divergence-free. This test asserts
the diagnostic fires on the broken model and stays quiet on the fixed one.
"""
import mlx.core as mx
import numpy as np

from mlxmc import run_nuts, warmup
from mlxmc.targets import funnel_logp, funnel_nc_logp

D = 2                       # v + one latent x: enough to make the neck bite, cheap to run
N_CHAINS = 40


def _run(logp, seed):
    key = mx.random.key(seed)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(N_CHAINS, D), key=k_init) * 2.0
    q_last, eps, Minv = warmup(logp, q0, n_warmup=300, n_leap=8, key=k_warm)
    return run_nuts(logp, q_last, n_samples=150, eps=eps, Minv_np=Minv,
                    key=k_nuts, max_tree_depth=6)


def test_centered_funnel_reports_divergences():
    res = _run(funnel_logp, 0)
    assert res.n_divergent is not None
    assert res.n_divergent > 0, "centered funnel should produce divergences (it's biased here)"


def test_noncentered_funnel_is_clean():
    res = _run(funnel_nc_logp, 1)
    total = res.n_chains * res.n_draws
    # The reparametrized model has no funnel geometry -> divergences should be rare/absent.
    assert res.n_divergent < 0.01 * total, f"unexpected divergences after reparam: {res.n_divergent}/{total}"
