"""Effective sample size: how many *independent* draws the chains are worth.

ESS = N / tau, with tau the integrated autocorrelation time (emcee-style:
FFT autocorrelation averaged over walkers, Sokal automatic windowing). The
fair cross-sampler metric is ESS/sec, since it folds in per-step cost.

Pure numpy on a structured (T, N, D) chain -- no sampler or MLX dependency, so
it's a leaf module the samplers don't pull in.
"""
import numpy as np


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
