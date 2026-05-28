"""Warmup adaptation for HMC: dual-averaging step size + windowed dense mass-matrix
estimation (Stan-style). This makes the preconditioned-HMC result *honest* -- rather
than being handed the true Sigma, we estimate M^{-1} = Cov(q) during warmup and adapt
eps to a target acceptance, then sample. The mass matrix is HMC's affine invariance, and
this is how you earn it instead of supplying it.

MLX notes:
  - eps, M^{-1}, and chol(M)^T are passed as *array arguments* to the compiled step, not
    closed-over constants (as make_hmc/make_phmc do). mx.compile recompiles on shape/dtype
    or baked-constant changes, NOT on array-value changes -- so we can vary eps/M every
    warmup iteration and reuse the one compiled graph.
  - n_leap stays a Python int: it's the unrolled leapfrog length, structural to the graph.
    Phase 1 adapts eps + M with L fixed; dynamic trajectory length is NUTS (Phase 2), which
    this MLX has no control-flow primitive (while_loop/scan) for.
  - Covariance + Cholesky run host-side in numpy fp64, so the ill-conditioned linear algebra
    (the real fp32 limit) never touches the GPU; only the leapfrog runs fp32.
  - Covariance is pooled across chains AND steps within a window (many chains => fast
    estimate). Early overdispersion is discarded by the windowed schedule; the final, longest
    slow window estimates M near stationarity.
"""
import math

import mlx.core as mx
import numpy as np


class DualAveraging:
    """Nesterov dual averaging for step-size adaptation (Hoffman & Gelman 2014, Alg. 5).

    The chain runs on the raw `exp(log_eps)` (keeps exploring); the *averaged* `eps_bar`
    is what we freeze for sampling. `restart` re-anchors at a window boundary, since the
    stable step size changes when the mass matrix does.
    """

    def __init__(self, eps0, target_accept=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.target, self.gamma, self.t0, self.kappa = target_accept, gamma, t0, kappa
        self.restart(eps0)

    def restart(self, eps0):
        self.mu = math.log(10.0 * eps0)          # shrink toward 10x the anchor
        self.Hbar = 0.0
        self.log_eps = math.log(eps0)
        self.log_eps_bar = math.log(eps0)
        self.m = 0

    def update(self, accept_prob):
        self.m += 1
        m = self.m
        w = 1.0 / (m + self.t0)
        self.Hbar = (1.0 - w) * self.Hbar + w * (self.target - accept_prob)
        self.log_eps = self.mu - math.sqrt(m) / self.gamma * self.Hbar
        eta = m ** (-self.kappa)
        self.log_eps_bar = eta * self.log_eps + (1.0 - eta) * self.log_eps_bar
        return math.exp(self.log_eps)

    @property
    def eps_bar(self):
        return math.exp(self.log_eps_bar)


def stan_windows(n_warmup, init_buffer=75, term_buffer=50, base_window=25):
    """Stan-style warmup schedule. Returns (init_buffer, term_buffer, window_ends), where
    window_ends are iteration counts (1-based, inclusive) at which to re-estimate M and
    restart step-size adaptation. M is held fixed during the init buffer (find the mode)
    and term buffer (final eps polish); the slow windows between double in length, and the
    last absorbs any remainder. Not bit-identical to Stan, but the same structure."""
    if init_buffer + term_buffer + base_window > n_warmup:
        # Too short for the default buffers: fall back to Stan's 15/75/10 proportions.
        init_buffer = max(1, int(round(0.15 * n_warmup)))
        term_buffer = max(1, int(round(0.10 * n_warmup)))
        base_window = max(1, n_warmup - init_buffer - term_buffer)
    last = n_warmup - term_buffer
    ends, start, window = [], init_buffer, base_window
    while start + window < last:
        nxt = start + window
        if nxt + 2 * window > last:     # next doubled window would overrun -> absorb now
            nxt = last
        ends.append(nxt)
        start, window = nxt, window * 2
    if not ends or ends[-1] != last:
        ends.append(last)
    return init_buffer, term_buffer, ends


def regularize_cov(cov, n):
    """Stan's dense-metric shrinkage toward a small diagonal (stabilizes small-n windows)."""
    d = cov.shape[0]
    return (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)


def make_warmup_step(logp_single, n_leap):
    """Leapfrog + Metropolis step with eps, M^{-1}, chol(M)^T as array arguments.
    Returns (q_new, mean_accept_prob); the continuous accept prob feeds dual averaging."""
    grad_logp = mx.vmap(mx.grad(logp_single))
    logp = mx.vmap(logp_single)

    @mx.compile
    def step(q, key, eps, Minv, Mhalf_T):
        n, _ = q.shape
        k_p, k_acc = mx.random.split(key, 2)
        z = mx.random.normal(shape=q.shape, key=k_p)
        p0 = z @ Mhalf_T                          # ~ N(0, M)
        K0 = 0.5 * ((p0 @ Minv) * p0).sum(1)      # 0.5 p^T M^{-1} p
        logp_q = logp(q)

        qq = q
        p = p0 + 0.5 * eps * grad_logp(qq)
        for i in range(n_leap):
            qq = qq + eps * (p @ Minv)            # drift uses M^{-1}
            if i != n_leap - 1:
                p = p + eps * grad_logp(qq)
        p = p + 0.5 * eps * grad_logp(qq)

        K = 0.5 * ((p @ Minv) * p).sum(1)
        log_accept = (logp(qq) - logp_q) + (K0 - K)
        accept_prob = mx.minimum(1.0, mx.exp(log_accept))     # continuous, for dual avg
        # A divergent leapfrog (NaN energy, e.g. the funnel neck in fp32) must REJECT, not
        # poison the dual-averaging mean with NaN. Treat NaN accept prob as 0.
        accept_prob = mx.where(mx.isnan(accept_prob), mx.zeros_like(accept_prob), accept_prob)
        accept = mx.random.uniform(shape=(n,), key=k_acc) < accept_prob
        q_new = mx.where(accept[:, None], qq, q)
        return q_new, accept_prob.mean()

    return step


def warmup(logp_single, q0, n_warmup, n_leap, key, eps0=0.25, target_accept=0.8,
           init_buffer=75, term_buffer=50, base_window=25):
    """Run Stan-style warmup. Returns (q_last, eps_bar, Minv_np) -- the tuned step size and
    estimated M^{-1} = Cov(q), ready to hand to run_chain."""
    n_chains, d = q0.shape
    init_buffer, term_buffer, window_ends = stan_windows(
        n_warmup, init_buffer, term_buffer, base_window)
    boundaries = set(window_ends)

    da = DualAveraging(eps0, target_accept)
    step = make_warmup_step(logp_single, n_leap)

    Minv_np = np.eye(d)                                   # start from the identity metric
    eps = mx.array(eps0, dtype=mx.float32)
    Minv = mx.array(Minv_np.astype(np.float32))
    Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))

    q, window_samples = q0, []
    for t in range(n_warmup):
        key, k = mx.random.split(key, 2)
        q, acc_prob = step(q, k, eps, Minv, Mhalf_T)
        mx.eval(q, acc_prob)
        eps = mx.array(da.update(float(acc_prob)), dtype=mx.float32)   # adapt eps every step

        if init_buffer <= t < (n_warmup - term_buffer):
            window_samples.append(np.array(q))            # collect cov samples in slow windows
        if (t + 1) in boundaries and window_samples:
            X = np.concatenate(window_samples, axis=0)    # (steps * n_chains, d)
            Minv_np = regularize_cov(np.cov(X, rowvar=False), X.shape[0])
            Minv = mx.array(Minv_np.astype(np.float32))
            Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))
            window_samples = []
            da.restart(da.eps_bar)                        # re-anchor eps to the new metric
            eps = mx.array(da.eps_bar, dtype=mx.float32)

    return q, da.eps_bar, Minv_np


def run_chain(logp_single, q0, n_steps, burn, eps, Minv_np, key, n_leap):
    """Sample with fixed tuned (eps, M). Returns the structured (T, N, D) chain for ESS."""
    step = make_warmup_step(logp_single, n_leap)
    eps_a = mx.array(eps, dtype=mx.float32)
    Minv_a = mx.array(Minv_np.astype(np.float32))
    Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))
    chain, q = [], q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, _ = step(q, k, eps_a, Minv_a, Mhalf_T)
        mx.eval(q)
        if t >= burn:
            chain.append(q)
    return mx.stack(chain, axis=0)
