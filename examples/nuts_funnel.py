"""NUTS exercised two ways:

  default       -- correctness on the canonical Gaussian (recovered cov vs true Sigma).
  funnel        -- NUTS vs jittered fixed-L HMC vs ensemble on the centered/non-centered
                   funnel. NUTS uses a GLOBAL static M (from warmup), so the centered
                   funnel's position-dependent scale should still bite (improved but not
                   cured); the non-centered funnel should be a clean NUTS win, matching
                   jittered fixed-L HMC with no manual L. Watch mean tree depth << max:
                   that's the vectorized-NUTS masking overhead (the batch pays the deepest
                   chain's cost).

Run:  python examples/nuts_funnel.py [funnel]
"""
import sys
import time

import mlx.core as mx
import numpy as np

from mlxmc.diagnostics import integrated_time, report
from mlxmc.nuts import run_nuts
from mlxmc.targets import (GAUSSIAN_MU, GAUSSIAN_SIGMA, funnel_logp,
                           funnel_nc_logp, gaussian_logp)
from mlxmc.warmup import warmup

# Sampling-phase-timed HMC/ensemble runners live in the sibling benchmark example.
from hard_targets import sample_ensemble, sample_hmc


def funnel_compare(key):
    def _summary(label, chain_mx, dt, extra=""):
        c = np.array(chain_mx)
        f = c.reshape(-1, c.shape[-1])
        n_bad = int((~np.isfinite(f).all(1)).sum())
        v = f[np.isfinite(f).all(1)][:, 0]
        tau = integrated_time(c[:, :, 0])               # v dimension
        ess_s = c.shape[0] * c.shape[1] / max(tau, 1.0) / dt
        print(f"  [{label:24s}] v mean {v.mean():+.2f} std {v.std():.2f}  v<-3 {100*(v<-3).mean():4.1f}%"
              f"  min v {v.min():+6.1f}  tau_v {tau:5.1f}  ESS/s {ess_s:10,.0f}"
              + (f"  non-finite {n_bad}" if n_bad else "") + (f"  {extra}" if extra else ""))

    for name, logp in [("centered", funnel_logp), ("non-centered", funnel_nc_logp)]:
        print(f"\n==== FUNNEL {name}: NUTS vs fixed-L HMC vs ensemble  (true v: mean 0.00, std 3.00) ====")
        key, kqh, kqe, kw, kn, kh, ke = mx.random.split(key, 7)
        q0 = mx.random.normal(shape=(1000, 2), key=kqh)
        e0 = mx.random.normal(shape=(2000, 2), key=kqe)
        q_last, eps_bar, Minv = warmup(logp, q0, 600, 8, kw)

        t0 = time.time()
        ch, mdepth, maxdepth = run_nuts(logp, q_last, 1500, eps_bar, Minv, kn)
        mx.eval(ch)
        ndt = time.time() - t0
        _summary("NUTS", ch, ndt, extra=f"wall {ndt:5.1f}s  depth mean {mdepth:.1f}/max {maxdepth}")
        hc, hdt = sample_hmc(logp, q_last, eps_bar, Minv, kh, 8, 1500)
        _summary("fixed-L HMC (L=8, jit)", hc, hdt)
        ec, edt = sample_ensemble(logp, e0, ke, 3000, 1000)
        _summary("ensemble", ec, edt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "funnel":
        funnel_compare(mx.random.key(0))
        sys.exit(0)

    Sigma = GAUSSIAN_SIGMA
    n_chains, n_warmup, n_sample = 1000, 600, 1500
    key = mx.random.key(0)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)

    q0 = mx.random.normal(shape=(n_chains, 2), key=k_init) * 5.0
    q_last, eps_bar, Minv = warmup(gaussian_logp, q0, n_warmup, 8, k_warm)
    print(f"warmup: eps {eps_bar:.3f},  estimated M^-1 diag {np.diag(Minv).round(2)}")

    t0 = time.time()
    chain, mean_depth, max_depth = run_nuts(gaussian_logp, q_last, n_sample, eps_bar, Minv, k_nuts)
    mx.eval(chain)
    dt = time.time() - t0
    report(chain, "NUTS (multinomial, warmup eps+M)", dt)

    s = np.array(chain).reshape(-1, 2)
    print(f"  tree depth mean {mean_depth:.2f} / max {max_depth}  (low mean vs max => leapfrog masked)")
    print(f"  recovered mean {s.mean(0).round(3)}  (true {GAUSSIAN_MU})")
    print(f"  recovered cov\n{np.cov(s.T).round(2)}\n  true\n{Sigma}")
