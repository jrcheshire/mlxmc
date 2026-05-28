"""Affine invariance of the G&W ensemble sampler.

Run the sampler on a base target and on its affine image q(y)=p(A^-1(y-b)) with the SAME
random stream. In exact arithmetic the two runs are related by y_t = A x_t + b at every
step, so acceptance is bit-identical and a 256x-worse-conditioned target costs nothing.

Tolerances (Standard): acceptance fraction EXACTLY equal (no Metropolis decision flips);
trajectory image |y - (A x + b)| < 0.1 over all samples. A flipped borderline accept (an
fp32 artifact on the ill-conditioned image) would break BOTH at once, so this is a real
invariance check, not a loose one.
"""
import mlx.core as mx
import numpy as np

from mlxmc import run_ensemble

D = 3


def test_affine_invariance():
    rng = np.random.default_rng(0)

    def logp_base(x):
        return -0.5 * (x @ x)

    # Ill-conditioned affine map: random rotation times scales [8, 2, 0.5] -> cond(Sigma)=256.
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    A_np = Q @ np.diag([8.0, 2.0, 0.5])
    b_np = np.array([3.0, -5.0, 1.0])
    A_T = mx.transpose(mx.array(A_np))
    b = mx.array(b_np)
    Ainv = mx.array(np.linalg.inv(A_np))

    def logq(y):
        r = Ainv @ (y - b)
        return -0.5 * (r @ r)

    key = mx.random.key(42)
    key, k_init = mx.random.split(key)
    E0 = mx.random.normal(shape=(200, D), key=k_init)         # matched to base N(0, I)
    E0_mapped = E0 @ A_T + b                                  # matched to q = N(b, A A^T)

    # SAME key for both runs -> identical random stream.
    xs, acc_base = run_ensemble(logp_base, E0, 100, 0, key)
    ys, acc_tr = run_ensemble(logq, E0_mapped, 100, 0, key)

    assert acc_base == acc_tr, f"acceptance not identical: {acc_base} vs {acc_tr} (a decision flipped)"

    mapped = np.array(xs) @ A_np.T + b_np
    max_dev = float(np.abs(np.array(ys) - mapped).max())
    assert max_dev < 0.1, f"affine image deviation {max_dev:.2e} exceeds 0.1"

    # And the transformed run recovers N(b, A A^T).
    y = np.array(ys)
    assert np.allclose(y.mean(0), b_np, atol=0.5)
    cov_true = A_np @ A_np.T
    assert np.allclose(np.cov(y.T).diagonal(), cov_true.diagonal(), rtol=0.15)
