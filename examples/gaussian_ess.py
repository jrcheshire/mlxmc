"""The Gaussian ESS story: affine-invariant ensemble vs identity-mass HMC vs
preconditioned HMC (M = Sigma^{-1}), compared by ESS/sec on the canonical
correlated 2-D Gaussian (corr 0.9, 25:1 variance ratio).

The point: the ensemble handles the ill-conditioning for free (no tuning, no
gradients); identity-mass HMC pays for the bad conditioning in mixing; supplying
the right mass matrix (here the true Sigma) is HMC's affine invariance and recovers
the gap with far fewer, cheaper leapfrog steps. `examples/warmup_validation.py`
shows the same M *estimated* during warmup rather than supplied.

ESS needs the per-chain structure, so these local runners retain the (T, N, D)
chain -- unlike the library's run_ensemble/run_hmc, which flatten for moment recovery.

Run:  python examples/gaussian_ess.py
"""
import time

import mlx.core as mx
import numpy as np

from mlxmc.diagnostics import report
from mlxmc.ensemble import make_sampler
from mlxmc.hmc import make_hmc
from mlxmc.preconditioned import run_phmc
from mlxmc.targets import GAUSSIAN_SIGMA, gaussian_logp


def run_ensemble_chain(e0, n_steps, burn, key, a=2.0):
    n_walkers, n_dim = e0.shape
    half = n_walkers // 2
    update = make_sampler(gaussian_logp, n_dim, a)
    chain, e = [], e0
    for t in range(n_steps):
        key, k0, k1 = mx.random.split(key, 3)
        h0, h1 = e[:half], e[half:]
        h0, _ = update(h0, h1, k0)
        h1, _ = update(h1, h0, k1)
        e = mx.concatenate([h0, h1], axis=0)
        mx.eval(e)
        if t >= burn:
            chain.append(e)
    return mx.stack(chain, axis=0)


def run_hmc_chain(q0, n_steps, burn, eps, n_leap, key):
    step = make_hmc(gaussian_logp, eps, n_leap)
    chain, q = [], q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, _ = step(q, k)
        mx.eval(q)
        if t >= burn:
            chain.append(q)
    return mx.stack(chain, axis=0)


if __name__ == "__main__":
    Sigma = GAUSSIAN_SIGMA
    Minv = Sigma                                       # M^{-1} = Sigma
    Mhalf = np.linalg.cholesky(np.linalg.inv(Sigma))   # chol(M), M = Sigma^{-1}
    key = mx.random.key(0)

    key, ki = mx.random.split(key)
    ens0 = mx.random.normal(shape=(2000, 2), key=ki) * 5.0
    t0 = time.time()
    ec = run_ensemble_chain(ens0, 2000, 500, key)
    mx.eval(ec)
    e_ess, e_dt = report(ec, "ensemble (no grad, no tuning)", time.time() - t0)

    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(1000, 2), key=ki) * 5.0
    t0 = time.time()
    hc = run_hmc_chain(q0, 1500, 500, 0.15, 40, key)
    mx.eval(hc)
    h_ess, h_dt = report(hc, "HMC identity mass (eps=0.15, L=40)", time.time() - t0)

    key, ki = mx.random.split(key)
    q0p = mx.random.normal(shape=(1000, 2), key=ki) * 5.0
    t0 = time.time()
    pc = run_phmc(gaussian_logp, q0p, 1500, 500, 0.7, 6, key, Minv, Mhalf)
    mx.eval(pc)
    p_ess, p_dt = report(pc, "HMC preconditioned M=Sigma^-1 (eps=0.7, L=6)", time.time() - t0)

    print("\n=== ESS/sec ===")
    print(f"  ensemble:       {e_ess / e_dt:>10,.0f}")
    print(f"  HMC identity:   {h_ess / h_dt:>10,.0f}")
    print(f"  HMC precond:    {p_ess / p_dt:>10,.0f}   "
          f"({(p_ess / p_dt) / (h_ess / h_dt):.1f}x identity HMC, L=6 vs 40 -> 7/41 the gradients/step)")
