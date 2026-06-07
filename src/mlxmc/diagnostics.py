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
    # Accept either a raw (T, N, D) chain or a Result (duck-typed via `.samples`, whose
    # canonical layout is (chain, draw, dim) -> transpose back to time-major for tau).
    if hasattr(chain_mx, "samples"):
        c = np.transpose(np.asarray(chain_mx.samples), (1, 0, 2))
    else:
        c = np.array(chain_mx)                 # (T, N, D)
    T, N, D = c.shape
    total = T * N
    tau = max(integrated_time(c[:, :, d]) for d in range(D))
    ess = total / tau
    print(f"\n[{label}]")
    print(f"  raw samples {total:,}  ({T} steps x {N})   wall {dt:.2f}s")
    if tau <= 0 or ess <= 0:
        # The FFT+Sokal window collapses to ~lag-1 in the strong-antithetic regime (tau < 1)
        # and can return a negative tau/ESS. That's an honest "below the estimator's floor"
        # signal, not a real ESS -- warn rather than print a meaningless number.
        print(f"  tau (worst dim): {tau:.1f} steps  ->  ESS undefined (antithetic floor, tau<=0)")
        print("  ESS/sec: n/a (compare via the recovered metric, not ESS/sec, when tau<=1)")
    else:
        print(f"  tau (worst dim): {tau:.1f} steps  ->  ESS {ess:,.0f}  ({100 * ess / total:.1f}% independent)")
        print(f"  ESS/sec: {ess / dt:,.0f}")
    return ess, dt


# ----------------------------------------------------------------- convergence diagnostics
# Rank-normalized split-R-hat and bulk/tail-ESS (Vehtari, Gelman, Simpson, Carpenter,
# Burkner 2021, Bayesian Analysis 16(2):667-718). These answer "did independent chains
# *agree*?" (convergence), which is distinct from integrated_time/report above, which
# measure *efficiency* (ESS/sec) of a single run. The estimators below mirror the Stan /
# ArviZ implementations so they validate against ArviZ to numerical tolerance; they take a
# single parameter's draws shaped (chains, draws) -- the per-dimension loop lives in Result.
#
# Pure numpy throughout (Acklam's inverse-normal, no scipy at runtime) to keep this a
# dependency-light leaf module, consistent with the rest of the file.


def _inverse_normal_cdf(p):
    """Phi^{-1}(p) via Acklam's rational approximation (|abs err| < 1.15e-9 in the central
    region). Avoids a scipy runtime dependency; validated against scipy.special.ndtri."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = np.asarray(p, dtype=float)
    x = np.empty_like(p)
    lo, hi = p < 0.02425, p > (1 - 0.02425)
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.where(lo, p, 1.0)))
    x = np.where(lo, (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                 / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), x)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 1.0)))
    x = np.where(hi, -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                 / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), x)
    q = np.where(mid, p, 0.5) - 0.5
    r = q * q
    x = np.where(mid, (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                 / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1), x)
    return x


def _rankdata_average(flat):
    """Average ranks (1-based, ties averaged) -- scipy.stats.rankdata(method='average')."""
    sorter = np.argsort(flat, kind="mergesort")
    inv = np.empty(flat.size, dtype=np.intp)
    inv[sorter] = np.arange(flat.size)
    s = flat[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.flatnonzero(obs), flat.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _z_scale(x):
    """Rank-normalize (chains, draws): pool, average-rank, map to normal scores
    z = Phi^{-1}((r - 3/8)/(S - 1/4)) -- the rank normalization of Vehtari 2021."""
    r = _rankdata_average(x.ravel())
    z = _inverse_normal_cdf((r - 0.375) / (r.size - 0.25))
    return z.reshape(x.shape)


def _split_chains(x):
    """(chains, draws) -> (2*chains, draws//2). Splitting halves catches within-chain
    non-stationarity (a chain that drifts looks like two disagreeing half-chains)."""
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, n:2 * n]], axis=0)


def _rhat(x):
    """Classic Gelman-Rubin R-hat on (chains, draws) of already-transformed values."""
    m, n = x.shape
    chain_means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()
    B = n * chain_means.var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def rhat(samples):
    """Rank-normalized split-R-hat (Vehtari 2021): max of bulk and folded-tail R-hat.
    `samples`: (chains, draws) for one parameter. Convergence target < 1.01."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2:
        raise ValueError("rhat expects a 2-D (chains, draws) array")
    bulk = _rhat(_z_scale(_split_chains(x)))
    folded = _rhat(_z_scale(_split_chains(np.abs(x - np.median(x)))))
    return max(bulk, folded)


def _autocov(x):
    """Autocovariance at every lag for a 1-D series (biased /n estimator, via FFT) --
    matches Stan/ArviZ. Reuses the FFT idiom of autocorr_1d but keeps it unnormalized."""
    x = x - x.mean()
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()          # next power of two >= 2n
    f = np.fft.rfft(x, n=nfft)
    return np.fft.irfft(f * np.conjugate(f), n=nfft)[:n].real / n


def _ess(x):
    """Stan/Geyer multi-chain ESS on (chains, draws). Combines within- and between-chain
    variance, then sums the autocorrelation via Geyer's initial positive + monotone
    sequence (the truncation that makes the sum stable)."""
    m, n = x.shape
    if n < 4:
        return np.nan
    acov = np.stack([_autocov(x[j]) for j in range(m)], axis=0)     # (m, n), biased
    mean_var = acov[:, 0].mean() * n / (n - 1.0)                    # W, within-chain (ddof=1)
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus += x.mean(axis=1).var(ddof=1)                     # + between-chain
    a = acov.mean(axis=0)                                           # mean autocov over chains
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even, rho_odd = 1.0, 1.0 - (mean_var - a[1]) / var_plus
    rho[1] = rho_odd
    t = 1                                                           # Geyer initial positive seq
    while t < (n - 3) and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - a[t + 1]) / var_plus
        rho_odd = 1.0 - (mean_var - a[t + 2]) / var_plus
        if (rho_even + rho_odd) >= 0.0:
            rho[t + 1], rho[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t
    k = 1                                                           # initial monotone seq
    while k <= max_t - 2:
        if (rho[k + 1] + rho[k + 2]) > (rho[k - 1] + rho[k]):
            rho[k + 1] = (rho[k - 1] + rho[k]) / 2.0
            rho[k + 2] = rho[k + 1]
        k += 2
    tau = -1.0 + 2.0 * rho[:max_t + 1].sum()
    tau = max(tau, 1.0 / np.log10(m * n))                          # ArviZ's small-sample floor
    return (m * n) / tau


def ess_bulk(samples):
    """Bulk effective sample size: ESS of the rank-normalized, split draws (Vehtari 2021).
    `samples`: (chains, draws). Recommended minimum ~100 per chain for the bulk to be usable."""
    return _ess(_z_scale(_split_chains(np.asarray(samples, dtype=float))))


def ess_tail(samples):
    """Tail ESS: min of the 5%/95% quantile-indicator ESS -- the tails can be badly
    undersampled even when the bulk mixes, so credible-interval reliability needs this."""
    x = np.asarray(samples, dtype=float)
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return min(_ess(_split_chains((x <= q05).astype(float))),
               _ess(_split_chains((x <= q95).astype(float))))


def ess_mean(samples):
    """ESS for the posterior mean (split draws, no rank normalization) -- the ESS that
    pairs with the mean's Monte Carlo standard error."""
    return _ess(_split_chains(np.asarray(samples, dtype=float)))


def mcse_mean(samples):
    """Monte Carlo standard error of the posterior mean: sd / sqrt(ess_mean). Report this
    alongside the mean so readers can judge numerical reliability of the estimate."""
    x = np.asarray(samples, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(ess_mean(x)))
