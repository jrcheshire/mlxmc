"""Effective sample size: how many *independent* draws the chains are worth.

ESS = N / tau, with tau the integrated autocorrelation time (emcee-style:
FFT autocorrelation averaged over walkers, Sokal automatic windowing). The
fair cross-sampler metric is ESS/sec, since it folds in per-step cost.

Self-contained: reuses the compiled step-builders from the two samplers but
runs its own loops so it can retain the structured (T, N, D) chain.
"""
import time

import mlx.core as mx
import numpy as np

from ensemble_sampler import make_sampler
from hmc import make_hmc

# Shared target: corr 0.9, 25:1 variance ratio (same as both samplers).
mu_np = np.array([1.0, -2.0])
Sigma_np = np.array([[25.0, 4.5], [4.5, 1.0]])
Sig_inv = mx.array(np.linalg.inv(Sigma_np))
mu = mx.array(mu_np)


def logp_single(x):
    d = x - mu
    return -0.5 * (d @ Sig_inv @ d)


def run_ensemble(ensemble, n_steps, burn, key, a=2.0):
    n_walkers, n_dim = ensemble.shape
    half = n_walkers // 2
    update = make_sampler(logp_single, n_dim, a)
    chain, e = [], ensemble
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
    step = make_hmc(logp_single, eps, n_leap)
    chain, q = [], q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, _ = step(q, k)
        mx.eval(q)
        if t >= burn:
            chain.append(q)
    return mx.stack(chain, axis=0)


def autocorr_1d(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] == 0:               # constant (stuck) walker: autocorrelation undefined
        return None
    return acf / acf[0]


def auto_window(taus, c=5.0):
    m = np.arange(len(taus)) < c * taus
    return np.argmin(m) if np.any(~m) else len(taus) - 1


def integrated_time(y):                       # y: (T, N) for one dimension
    # Skip stuck (zero-variance) walkers, which would make acf/acf[0] a 0/0 = nan and
    # poison the walker average. A large skipped fraction is itself a mixing-failure tell.
    acfs = [a for a in (autocorr_1d(y[:, w]) for w in range(y.shape[1])) if a is not None]
    if not acfs:
        return np.nan
    f = np.mean(acfs, axis=0)
    taus = 2.0 * np.cumsum(f) - 1.0
    return taus[auto_window(taus)]


def report(chain_mx, label, dt):
    c = np.array(chain_mx)                     # (T, N, D)
    T, N, D = c.shape
    total = T * N
    tau = max(integrated_time(c[:, :, d]) for d in range(D))
    ess = total / tau
    print(f"\n[{label}]")
    print(f"  raw samples {total:,}  ({T} steps x {N})   wall {dt:.2f}s")
    print(f"  tau (worst dim): {tau:.1f} steps  ->  ESS {ess:,.0f}  ({100 * ess / total:.1f}% independent)")
    print(f"  ESS/sec: {ess / dt:,.0f}")
    return ess, dt


if __name__ == "__main__":
    key = mx.random.key(0)

    key, ki = mx.random.split(key)
    ens0 = mx.random.normal(shape=(2000, 2), key=ki) * 5.0
    t0 = time.time()
    ens_chain = run_ensemble(ens0, 2000, 500, key)
    mx.eval(ens_chain)
    e_ess, e_dt = report(ens_chain, "affine-invariant ensemble", time.time() - t0)

    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(1000, 2), key=ki) * 5.0
    t0 = time.time()
    hmc_chain = run_hmc_chain(q0, 1500, 500, 0.15, 40, key)
    mx.eval(hmc_chain)
    h_ess, h_dt = report(hmc_chain, "HMC (identity mass, eps=0.15 L=40)", time.time() - t0)

    print(f"\nESS/sec ratio (ensemble / HMC): {(e_ess / e_dt) / (h_ess / h_dt):.1f}x")
