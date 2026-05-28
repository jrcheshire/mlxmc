"""Validate Stan-style warmup: estimate M during warmup, then check (a) the estimate
matches the true Sigma and (b) the adapted sampler recovers the oracle (true-Sigma)
ESS/sec. The same eps_bar is used for both sampling runs so the comparison isolates
the mass-matrix estimate.

Run:  python examples/warmup_validation.py [L]
  L (default 8) is the leapfrog length. Short L pushes tau > 1, where adapted-vs-oracle
  ESS/sec is a reliable discriminator; long L mixes into the antithetic floor (tau < 1)
  where it isn't. eps is tuned at the chosen L either way.
"""
import sys
import time

import mlx.core as mx
import numpy as np

from mlxmc.diagnostics import report
from mlxmc.targets import GAUSSIAN_SIGMA, gaussian_logp
from mlxmc.warmup import run_chain, warmup

if __name__ == "__main__":
    Sigma = GAUSSIAN_SIGMA
    n_leap = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_chains, n_warmup, n_sample = 1000, 600, 1500
    # Distinct streams for init / warmup / each sampling run (no RNG reuse across phases).
    key = mx.random.key(0)
    k_init, k_warm, k_ad, k_or = mx.random.split(key, 4)

    q0 = mx.random.normal(shape=(n_chains, 2), key=k_init) * 5.0

    q_last, eps_bar, Minv_est = warmup(gaussian_logp, q0, n_warmup, n_leap, k_warm)
    rel_err = np.linalg.norm(Minv_est - Sigma) / np.linalg.norm(Sigma)
    print(f"warmup: {n_warmup} steps, {n_chains} chains, L={n_leap}")
    print(f"  tuned eps (eps_bar): {eps_bar:.3f}   (oracle precond used 0.7 at L=6)")
    print(f"  estimated M^-1 vs true Sigma  (rel. Frobenius err {rel_err:.3f}):")
    print(f"    estimated: {Minv_est.ravel().round(3)}")
    print(f"    true:      {Sigma.ravel().round(3)}")

    # Sampling carries the warmed-up positions forward (q_last is near-stationary), so
    # burn=0. Both runs start from the same q_last; only the metric differs.
    t0 = time.time()
    adapted = run_chain(gaussian_logp, q_last, n_sample, 0, eps_bar, Minv_est, k_ad, n_leap)
    mx.eval(adapted)
    a_ess, a_dt = report(adapted, "adapted (estimated M, tuned eps)", time.time() - t0)

    t0 = time.time()
    oracle = run_chain(gaussian_logp, q_last, n_sample, 0, eps_bar, Sigma, k_or, n_leap)
    mx.eval(oracle)
    o_ess, o_dt = report(oracle, "oracle (true Sigma, same eps)", time.time() - t0)

    print("\n=== ESS/sec (adapted vs oracle; close => warmup recovered the metric) ===")
    print(f"  adapted:  {a_ess / a_dt:>10,.0f}")
    print(f"  oracle:   {o_ess / o_dt:>10,.0f}   "
          f"(adapted = {(a_ess / a_dt) / (o_ess / o_dt):.2f}x oracle)")
