"""NUTS (No-U-Turn Sampler), multinomial variant (Hoffman & Gelman 2014 + Betancourt 2017),
vectorized over chains in MLX.

The MLX story (Phase 2): MLX has no `while_loop`/`scan` (see the README's "Why MLX"), so the tree-doubling
recursion runs in **host Python** while each leapfrog leaf is `vmap`'d over all chains and
`mx.compile`'d. Chains U-turn at different depths; a finished chain still rides along in the
batched leapfrog but is **masked out** (`mx.where` on a per-chain `cont` flag), so it's frozen
correctly. The doubling loop stops at `max_tree_depth` or when no chain is still going. The gap
vs JAX's traced `while_loop` is the wasted leapfrogs on already-stopped chains (mean tree depth
<< max is the tell). Trajectory length is adaptive, which removes the fixed-L resonance that
eps-jitter papers over for fixed-L HMC.

`eps` is a per-step argument to the returned step (not closed over) so dual-averaging in
`nuts_warmup` can vary it every iteration without rebuilding or recompiling the leapfrog leaf.
`Minv` is closed over for the compile, so changing M (at window boundaries during warmup)
means rebuilding the closure -- a few recompile events over a warmup, not per step.
"""
import mlx.core as mx
import numpy as np

from mlxmc.result import Result
from mlxmc.warmup import DualAveraging, regularize_cov, stan_windows

DMAX = 1000.0      # divergence threshold on the Hamiltonian error
NEG = -1e30        # stand-in for log-weight 0 (divergent leaf); finite to keep logaddexp NaN-free


def expit(x):                                  # logistic; finite NEG keeps this NaN-free
    return 1.0 / (1.0 + mx.exp(-x))


def logaddexp(a, b):
    m = mx.maximum(a, b)
    return m + mx.log(mx.exp(a - m) + mx.exp(b - m))


def wsel(mask, a, b):                          # per-chain select; broadcasts (N,) over (N,D)
    return mx.where(mask[:, None] if a.ndim == 2 else mask, a, b)


def make_nuts(logp_single, Minv_np, max_tree_depth=10):
    """Build a NUTS step `step(theta, key, eps)` that returns (sample, depths, mean_accept).

    `mean_accept` is the H&G Alg 6 dual-averaging statistic -- the per-leaf Metropolis
    acceptance min(1, exp(-Delta_H)) averaged over all leapfrog leaves in the iteration's
    tree (per chain), then averaged over chains. Zeroed on divergent leaves. Returned for
    use by `nuts_warmup`; `run_nuts` discards it.
    """
    grad_logp = mx.vmap(mx.grad(logp_single))
    logp = mx.vmap(logp_single)
    Minv = mx.array(Minv_np.astype(np.float32))
    Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))

    @mx.compile
    def leap(theta, r, se):                    # one leapfrog; se = signed step, (N,1)
        r = r + 0.5 * se * grad_logp(theta)
        theta = theta + se * (r @ Minv)
        r = r + 0.5 * se * grad_logp(theta)
        return theta, r

    def joint(theta, r):                        # -H = logp - 0.5 r^T M^-1 r  (the log-weight)
        return logp(theta) - 0.5 * ((r @ Minv) * r).sum(1)

    def no_uturn(tm, rm, tp, rp):               # True = keep going (no U-turn), generalized metric
        d = tp - tm
        return ((d * (rm @ Minv)).sum(1) >= 0) & ((d * (rp @ Minv)).sum(1) >= 0)

    def build(theta, r, lw0, depth, dirn, eps, key):
        """Recursive subtree builder. Returns the two endpoints, the multinomial proposal from
        this subtree, its total log-weight, a per-chain continue flag, and the leaf-accept
        (sum, count) used by the dual-averaging statistic."""
        if depth == 0:                          # base: a single leapfrog in `dirn`
            th1, r1 = leap(theta, r, (dirn * eps)[:, None])
            lw1 = joint(th1, r1)
            div = (lw0 - lw1 > DMAX) | mx.isnan(lw1)
            # H&G Alg 6 leaf statistic: min(1, exp(lw1 - lw0)) = min(1, exp(-Delta_H)); zeroed
            # on divergence/NaN. exp(min(0, ...)) keeps the value <= 1 without an explicit clamp.
            leaf_a = mx.where(div, mx.zeros_like(lw1),
                              mx.exp(mx.minimum(mx.zeros_like(lw1), lw1 - lw0)))
            ones = mx.ones_like(lw1)
            return th1, r1, th1, r1, th1, mx.where(div, NEG, lw1), ~div, leaf_a, ones, div

        kL, kR, ks = mx.random.split(key, 3)
        tm, rm, tp, rp, p1, lw1, s1, a1, c1, d1 = build(theta, r, lw0, depth - 1, dirn, eps, kL)
        lt, lr = wsel(dirn < 0, tm, tp), wsel(dirn < 0, rm, rp)      # extend the leading edge
        tm2, rm2, tp2, rp2, p2, lw2, s2, a2, c2, d2 = build(lt, lr, lw0, depth - 1, dirn, eps, kR)

        ftm, frm = wsel(dirn < 0, tm2, tm), wsel(dirn < 0, rm2, rm)  # stitched full endpoints
        ftp, frp = wsel(dirn < 0, tp, tp2), wsel(dirn < 0, rp, rp2)
        pick2 = mx.random.uniform(shape=(theta.shape[0],), key=ks) < expit(lw2 - lw1)
        prop = wsel(pick2, p2, p1)
        lw = logaddexp(lw1, lw2)
        s = s1 & s2 & no_uturn(ftm, frm, ftp, frp)
        # Accept stat sums all leaves traversed (H&G: n_alpha counts every leapfrog leaf,
        # not just the ones in valid subtrees). State propagation still gates on s1 as before.
        # Divergence is OR'd over the whole subtree -- a divergence anywhere taints the draw.
        return (wsel(s1, ftm, tm), wsel(s1, frm, rm), wsel(s1, ftp, tp), wsel(s1, frp, rp),
                wsel(s1, prop, p1), mx.where(s1, lw, lw1), s1 & s, a1 + a2, c1 + c2, d1 | d2)

    def step(theta, key, eps):
        N = theta.shape[0]
        km, k = mx.random.split(key, 2)
        r0 = mx.random.normal(shape=theta.shape, key=km) @ Mhalf_T   # ~ N(0, M)
        lw0 = joint(theta, r0)
        tm = tp = theta
        rm = rp = r0
        sample, lw_tree = theta, lw0
        cont = mx.array(np.ones(N, dtype=bool))
        depths = mx.zeros((N,), dtype=mx.int32)
        accept_sum = mx.zeros((N,))
        accept_cnt = mx.zeros((N,))
        diverged = mx.array(np.zeros(N, dtype=bool))

        for depth in range(max_tree_depth):
            cont_was = cont                     # gate accept-stat accumulation on entering state
            k, kdir, ksub, ksel = mx.random.split(k, 4)
            dirn = mx.where(mx.random.uniform(shape=(N,), key=kdir) < 0.5, -1.0, 1.0)
            depths = depths + cont.astype(mx.int32)
            lt, lr = wsel(dirn < 0, tm, tp), wsel(dirn < 0, rm, rp)
            ntm, nrm, ntp, nrp, prop, lw_sub, s_sub, a_sub, c_sub, div_sub = build(
                lt, lr, lw0, depth, dirn, eps, ksub)

            new_tm, new_rm = wsel(dirn < 0, ntm, tm), wsel(dirn < 0, nrm, rm)
            new_tp, new_rp = wsel(dirn < 0, tp, ntp), wsel(dirn < 0, rp, nrp)
            # multinomial: adopt the new subtree's proposal with prob W_sub / (W_tree + W_sub),
            # but ONLY if the subtree is valid (H&G Alg 3: gate on s'). Adopting proposals from
            # a subtree that internally U-turned/diverged over-samples its returned far points.
            pick = (mx.random.uniform(shape=(N,), key=ksel) < expit(lw_sub - lw_tree)) & cont & s_sub
            sample = wsel(pick, prop, sample)
            lw_tree = mx.where(cont, logaddexp(lw_tree, lw_sub), lw_tree)
            tm, rm = wsel(cont, new_tm, tm), wsel(cont, new_rm, rm)
            tp, rp = wsel(cont, new_tp, tp), wsel(cont, new_rp, rp)
            # Chains that had already stopped contribute no new leaves this iteration.
            accept_sum = accept_sum + mx.where(cont_was, a_sub, mx.zeros_like(a_sub))
            accept_cnt = accept_cnt + mx.where(cont_was, c_sub, mx.zeros_like(c_sub))
            diverged = diverged | mx.where(cont_was, div_sub, mx.zeros_like(div_sub))
            cont = cont & s_sub & no_uturn(new_tm, new_rm, new_tp, new_rp)
            mx.eval(cont, sample, tm, tp, rm, rp, lw_tree, depths, accept_sum, accept_cnt, diverged)
            if cont.sum().item() == 0:          # host-side early stop once all chains U-turned
                break

        # Per-chain leaf-mean accept, then mean over chains. Empty count -> 1 to avoid 0/0.
        per_chain = accept_sum / mx.maximum(accept_cnt, mx.ones_like(accept_cnt))
        return sample, depths, per_chain.mean(), diverged

    return step


def run_nuts(logp_single, theta0, n_samples, eps, Minv_np, key, max_tree_depth=10):
    """Sample with NUTS. Returns a `Result` (theta0 already warmed); `sample_stats` carries
    per-draw `diverging` (the divergence info NUTS computes but used to throw away) and
    `tree_depth` -- max-vs-mean depth is the masking-overhead tell, the batch pays the
    deepest chain's cost."""
    step = make_nuts(logp_single, Minv_np, max_tree_depth)
    chain, depths_all, divs, theta = [], [], [], theta0
    for _ in range(n_samples):
        key, k = mx.random.split(key, 2)
        theta, depths, _, diverged = step(theta, k, eps)    # discard the leaf-accept stat
        mx.eval(theta, depths, diverged)
        chain.append(theta)
        depths_all.append(depths)
        divs.append(diverged)
    return Result.from_chain(
        mx.stack(chain, axis=0),
        sample_stats={"diverging": np.array(mx.stack(divs, axis=0)),
                      "tree_depth": np.array(mx.stack(depths_all, axis=0))},
    )


def nuts_warmup(logp_single, q0, n_warmup, key, eps0=0.25, target_accept=0.8,
                init_buffer=75, term_buffer=50, base_window=25, max_tree_depth=10):
    """NUTS-specific warmup: dual-averaging on NUTS's tree-averaged leaf-acceptance statistic,
    Stan-style windowed dense-M estimation. Returns (q_last, eps_bar, Minv_np) ready for
    `run_nuts` -- the same interface as `mlxmc.warmup.warmup`, but the tuned eps reflects
    NUTS's adaptive trajectory length rather than borrowing fixed-L HMC's optimum.

    The leaf-accept stat (H&G 2014 Alg 6): per leapfrog leaf, alpha_leaf = min(1, exp(-Delta_H)),
    zeroed on divergence; alpha for the iteration is the mean over leaves per chain, then over
    chains. Stan defaults target this at 0.8 (the same number used here -- the statistic differs,
    so the resulting eps does too).

    Implementation notes:
      - `eps` is a per-step argument to the NUTS step, so dual-averaging changes it every
        iteration without recompiling the leapfrog leaf.
      - `Minv` is closed over the compiled leaf, so when the windowed estimator produces a
        new M, the NUTS step is rebuilt (one recompile per window boundary, not per step).
      - Covariance + Cholesky are done host-side in fp64; only the leapfrog runs fp32.
      - A NaN accept stat (e.g. all leaves diverged in a single iteration) is treated as 0
        so dual-averaging stays finite.
    """
    n_chains, d = q0.shape
    init_buffer, term_buffer, window_ends = stan_windows(
        n_warmup, init_buffer, term_buffer, base_window)
    boundaries = set(window_ends)

    da = DualAveraging(eps0, target_accept)
    Minv_np = np.eye(d)
    step = make_nuts(logp_single, Minv_np, max_tree_depth)
    eps = eps0

    q, window_samples = q0, []
    for t in range(n_warmup):
        key, k = mx.random.split(key, 2)
        q, _depths, accept_prob, _div = step(q, k, eps)
        mx.eval(q, accept_prob)
        a = float(accept_prob)
        if not np.isfinite(a):
            a = 0.0                                       # treat all-divergent batch as reject
        eps = da.update(a)

        if init_buffer <= t < (n_warmup - term_buffer):
            window_samples.append(np.array(q))            # collect cov samples in slow windows
        if (t + 1) in boundaries and window_samples:
            X = np.concatenate(window_samples, axis=0)    # (steps * n_chains, d)
            Minv_np = regularize_cov(np.cov(X, rowvar=False), X.shape[0])
            step = make_nuts(logp_single, Minv_np, max_tree_depth)   # rebuild for the new M
            window_samples = []
            da.restart(da.eps_bar)                        # re-anchor eps to the new metric
            eps = da.eps_bar

    return q, da.eps_bar, Minv_np
