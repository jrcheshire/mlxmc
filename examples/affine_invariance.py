"""Empirical proof of affine invariance for the G&W ensemble sampler.

Map the base target p(x) through y = A x + b to q(y) = p(A^{-1}(y-b)). Running
the sampler on q from the affine-mapped initial ensemble, with the SAME random
stream, must reproduce the base run exactly mapped: y_t = A x_t + b for every
walker and step. So acceptance and mixing are identical -- a 256x-worse-
conditioned target costs nothing extra. (Exact to float32; a borderline accept
can rarely flip, which would show up as a large deviation.)
"""
import mlx.core as mx
import numpy as np

from mlxmc.ensemble import run_ensemble

rng = np.random.default_rng(0)
D = 3

# Base target: isotropic standard normal.
def logp_base(x):
    return -0.5 * (x @ x)

# Ill-conditioned affine map: random rotation times scales [8, 2, 0.5].
Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
A_np = Q @ np.diag([8.0, 2.0, 0.5])          # cond(A)=16 -> cond(Sigma)=256
b_np = np.array([3.0, -5.0, 1.0])
A = mx.array(A_np)
A_T = mx.transpose(A)
b = mx.array(b_np)
Ainv = mx.array(np.linalg.inv(A_np))

# Transformed target q(y) = N(b, A A^T): logq(y) = -0.5 |A^{-1}(y-b)|^2.
def logq(y):
    r = Ainv @ (y - b)
    return -0.5 * (r @ r)

n_walkers, n_steps, burn = 200, 100, 0       # init is in equilibrium, no burn needed
key = mx.random.key(42)
key, k_init = mx.random.split(key)
E0 = mx.random.normal(shape=(n_walkers, D), key=k_init)   # matched to base N(0, I)
E0_mapped = E0 @ A_T + b                                   # matched to q = N(b, A A^T)

# SAME key for both runs -> identical random stream.
xs, acc_base = run_ensemble(logp_base, E0, n_steps, burn, key)
ys, acc_tr = run_ensemble(logq, E0_mapped, n_steps, burn, key)

mapped = xs @ A_T + b
max_dev = float(mx.max(mx.abs(ys - mapped)))

print(f"condition number:   base target 1   |   transformed target {np.linalg.cond(A_np @ A_np.T):.0f}")
print(f"acceptance:         base {acc_base:.6f}   transformed {acc_tr:.6f}   (identical => invariant)")
print(f"max |y - (A x + b)| over {ys.shape[0]:,} samples:  {max_dev:.2e}   (=> exact affine image, to float32)")

y = np.array(ys)
print("\ntransformed run recovers N(b, A A^T):")
print(f"  mean recovered {np.round(y.mean(0), 2)}   vs true {b_np}")
print(f"  cov diag recovered {np.round(np.cov(y.T).diagonal(), 1)}   vs true {np.round((A_np @ A_np.T).diagonal(), 1)}")
