"""Validate mlxmc's native convergence diagnostics against ArviZ (the reference impl).

ArviZ + scipy live in the `dev` pixi env (`pixi run -e dev test`); this module skips when
they're absent, so the lean default env still collects the rest of the suite. Tolerances are
set from *measured* cross-implementation agreement (arviz 1.1.0), with ~2.5x margin over the
worst observed relative error -- not relaxed to make a failing test pass:

    inverse-normal CDF  ~5e-9 abs   -> 1e-7 abs
    rank-normalized Rhat ~5e-5 rel  -> 1e-3 rel
    ess bulk/tail/mean   ~2e-3 rel  -> 5e-3 rel   (small-ESS cases dominate the error)
    mcse_mean            ~1e-3 rel  -> 5e-3 rel

The residual ESS difference is the Geyer truncation/endpoint bookkeeping, which differs
slightly between any two faithful implementations; it shrinks at larger ESS.
"""
import numpy as np
import pytest

az = pytest.importorskip("arviz")
ndtri = pytest.importorskip("scipy.special").ndtri

from mlxmc import diagnostics as D

ATOL_PPF = 1e-7
RTOL_RHAT = 1e-3
RTOL_ESS = 5e-3
RTOL_MCSE = 5e-3


def ar1(phi, n, m, seed):
    """m chains of a length-n AR(1) process with unit stationary variance."""
    rng = np.random.default_rng(seed)
    x = np.zeros((m, n))
    e = rng.normal(size=(m, n)) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def offset(n, m, seed, spread=0.5):
    """Chains with staggered means -> Rhat > 1 and low ESS (the hard case for parity)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, n)) + (np.arange(m)[:, None] - (m - 1) / 2) * spread


def _val(ds):
    if hasattr(ds, "data_vars"):
        ds = list(ds.data_vars.values())[0]
    return float(np.asarray(ds))


DATASETS = {
    "white": ar1(0.0, 2000, 4, 1),
    "ar1_0.5": ar1(0.5, 2000, 4, 2),
    "ar1_0.9": ar1(0.9, 4000, 4, 3),
    "offset": offset(1500, 4, 4),
    "short": ar1(0.7, 300, 6, 5),
}


@pytest.fixture(params=list(DATASETS))
def chains(request):
    return DATASETS[request.param]


def test_inverse_normal_cdf_matches_scipy():
    p = np.linspace(1e-6, 1 - 1e-6, 99999)
    assert np.abs(D._inverse_normal_cdf(p) - ndtri(p)).max() < ATOL_PPF


def test_rhat_matches_arviz(chains):
    assert D.rhat(chains) == pytest.approx(_val(az.rhat(chains, method="rank")), rel=RTOL_RHAT)


def test_ess_bulk_matches_arviz(chains):
    assert D.ess_bulk(chains) == pytest.approx(_val(az.ess(chains, method="bulk")), rel=RTOL_ESS)


def test_ess_tail_matches_arviz(chains):
    ref = min(_val(az.ess(chains, method="quantile", prob=0.05)),
              _val(az.ess(chains, method="quantile", prob=0.95)))
    assert D.ess_tail(chains) == pytest.approx(ref, rel=RTOL_ESS)


def test_ess_mean_matches_arviz(chains):
    assert D.ess_mean(chains) == pytest.approx(_val(az.ess(chains, method="mean")), rel=RTOL_ESS)


def test_mcse_mean_matches_arviz(chains):
    assert D.mcse_mean(chains) == pytest.approx(_val(az.mcse(chains, method="mean")), rel=RTOL_MCSE)
