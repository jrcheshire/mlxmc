"""Constrained-parameter transforms: round-trips, Jacobian correctness, differentiability,
and end-to-end recovery of known constrained targets.

The bounded/uniform recovery test is the real Jacobian gate: a flat density on (low, high)
only pushes forward to a *uniform* distribution if the log-Jacobian term is correct -- drop
it and the draws pile up in the middle.
"""
import mlx.core as mx
import numpy as np
import pytest

from mlxmc import Exp, Identity, Sigmoid, Transform, constrain, run_ensemble

TRANSFORMS = {
    "identity": Transform([Identity()]),
    "exp": Transform([Exp()]),
    "sigmoid": Transform([Sigmoid(-2.0, 3.0)]),
    "mixed": Transform([Identity(), Exp(), Sigmoid(0.0, 10.0)]),
}


@pytest.fixture(params=list(TRANSFORMS))
def transform(request):
    return TRANSFORMS[request.param]


def test_roundtrip(transform):
    rng = np.random.default_rng(0)
    u = rng.normal(size=transform.n_dim).astype(np.float64)
    x = np.asarray(transform.forward(mx.array(u)))
    u_back = np.asarray(transform.inverse(mx.array(x)))
    assert np.allclose(u_back, u, atol=1e-4), f"inverse(forward(u)) != u: {u_back} vs {u}"


def test_log_det_jacobian_matches_finite_difference(transform):
    """Analytic log|det J| vs a central-difference numerical Jacobian (diagonal, per coord)."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=transform.n_dim).astype(np.float64)
    eps = 1e-5
    num = 0.0
    for d in range(transform.n_dim):
        up, um = u.copy(), u.copy()
        up[d] += eps
        um[d] -= eps
        deriv = (transform.forward_np(up)[d] - transform.forward_np(um)[d]) / (2 * eps)
        num += np.log(np.abs(deriv))
    ana = float(transform.log_det_jacobian(mx.array(u)))
    assert ana == pytest.approx(num, abs=1e-3), f"log|det J| analytic {ana} vs numerical {num}"


def test_grad_through_constrain():
    """mx.grad of the wrapped (unconstrained) log-density must match finite differences --
    confirms the gradient path HMC/NUTS rely on is correct through the transform + Jacobian."""
    tr = Transform([Exp(), Sigmoid(0.0, 1.0)])

    def logp_constrained(x):                       # arbitrary smooth target on (0,inf)x(0,1)
        return -0.5 * (x[0] - 2.0) ** 2 - 3.0 * (x[1] - 0.25) ** 2

    logp_u, _ = constrain(logp_constrained, tr)
    u = mx.array([0.3, -0.4])
    g = np.asarray(mx.grad(logp_u)(u))
    eps = 1e-3
    num = np.zeros(2)
    for d in range(2):
        up, um = np.array(u), np.array(u)
        up[d] += eps
        um[d] -= eps
        num[d] = (float(logp_u(mx.array(up))) - float(logp_u(mx.array(um)))) / (2 * eps)
    assert np.allclose(g, num, atol=1e-2), f"grad {g} vs finite-diff {num}"


def test_recover_exponential():
    """Target x ~ Exponential(1) on x>0 (logp = -x), sampled in unconstrained space via Exp.
    Recover mean=1, var=1."""
    tr = Transform([Exp()])
    logp_u, _ = constrain(lambda x: -x[0], tr)

    key = mx.random.key(0)
    key, ki = mx.random.split(key)
    e0 = mx.random.normal(shape=(120, 1), key=ki)
    res = run_ensemble(logp_u, e0, n_steps=5000, burn=1000, key=key)
    res.transform = tr
    x = res.constrained().reshape(-1)
    assert x.mean() == pytest.approx(1.0, rel=0.05), f"Exponential mean {x.mean():.3f}"
    assert x.var() == pytest.approx(1.0, rel=0.10), f"Exponential var {x.var():.3f}"
    assert (x > 0).all(), "Exp transform must keep samples positive"


def test_recover_uniform_bounded():
    """Flat target on (2, 5) sampled via Sigmoid(2, 5). Pushes forward to Uniform(2, 5)
    ONLY because the log-Jacobian is included -- recover mean=3.5, var=(3^2)/12=0.75."""
    low, high = 2.0, 5.0
    tr = Transform([Sigmoid(low, high)])
    logp_u, _ = constrain(lambda x: x[0] * 0.0, tr)        # flat in constrained space

    key = mx.random.key(1)
    key, ki = mx.random.split(key)
    e0 = mx.random.normal(shape=(120, 1), key=ki)
    res = run_ensemble(logp_u, e0, n_steps=5000, burn=1000, key=key)
    res.transform = tr
    x = res.constrained().reshape(-1)
    assert x.min() >= low and x.max() <= high, "Sigmoid must keep samples in (low, high)"
    assert x.mean() == pytest.approx(0.5 * (low + high), abs=0.1), f"Uniform mean {x.mean():.3f}"
    assert x.var() == pytest.approx((high - low) ** 2 / 12.0, rel=0.10), f"Uniform var {x.var():.3f}"
