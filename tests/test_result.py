"""The unified Result object: shape canonicalization, diagnostics wiring, constrained-space
mapping, and the ArviZ export. Uses synthetic draws so it's fast and sampler-independent."""
import numpy as np
import pytest

from mlxmc import Exp, Sigmoid, Transform
from mlxmc.result import Result


def _synthetic():
    rng = np.random.default_rng(0)
    chain_tnd = rng.normal(size=(200, 4, 3))                 # (T, N, D) as samplers produce
    stats = {"diverging": rng.random((200, 4)) < 0.02,       # (draw, chain)
             "tree_depth": rng.integers(1, 6, (200, 4))}
    return Result.from_chain(chain_tnd, accept_frac=0.9, sample_stats=stats)


def test_canonical_shapes():
    res = _synthetic()
    assert res.samples.shape == (4, 200, 3)                  # (chain, draw, dim)
    assert (res.n_chains, res.n_draws, res.n_dim) == (4, 200, 3)
    assert res.flat.shape == (800, 3)
    # sample_stats transposed to (chain, draw) to match samples
    assert res.sample_stats["diverging"].shape == (4, 200)


def test_diagnostics_shapes_and_divergences():
    res = _synthetic()
    for fn in (res.rhat, res.ess_bulk, res.ess_tail, res.mcse_mean):
        assert fn().shape == (3,)
    # independent N(0,1) draws -> Rhat ~ 1
    assert np.all(res.rhat() < 1.05)
    assert res.n_divergent == int(res.sample_stats["diverging"].sum())
    assert Result.from_chain(np.zeros((5, 2, 1))).n_divergent is None    # no stats -> None


def test_summary_runs(capsys):
    res = _synthetic()
    rows = res.summary()
    assert len(rows) == 3
    assert {"param", "mean", "sd", "mcse", "ess_bulk", "ess_tail", "r_hat"} <= set(rows[0])
    assert "divergences" in capsys.readouterr().out


def test_constrained_mapping():
    res = _synthetic()
    res.transform = Transform([Exp(), Sigmoid(0.0, 1.0), Exp()])
    con = res.constrained()
    assert con.shape == res.samples.shape
    assert (con[:, :, 0] > 0).all() and (con[:, :, 2] > 0).all()        # Exp -> positive
    assert ((con[:, :, 1] > 0) & (con[:, :, 1] < 1)).all()             # Sigmoid -> (0,1)
    # no transform -> identity
    assert np.array_equal(Result.from_chain(np.zeros((5, 2, 1))).constrained(),
                          np.zeros((2, 5, 1)))


def test_to_arviz():
    pytest.importorskip("arviz")
    res = _synthetic()
    idata = res.to_arviz()
    post = idata["posterior"]["x"]
    assert post.sizes["chain"] == 4 and post.sizes["draw"] == 200 and post.sizes["x_dim_0"] == 3
    assert int(np.asarray(idata["sample_stats"]["diverging"]).sum()) == res.n_divergent
