"""NUTS (No-U-Turn Sampler), multinomial variant (Hoffman & Gelman 2014 + Betancourt 2017),
vectorized over chains in MLX.

The MLX story (Phase 2): MLX has no `while_loop`/`scan` (see CLAUDE.md), so the tree-doubling
recursion runs in **host Python** while each leapfrog leaf is `vmap`'d over all chains and
`mx.compile`'d. Chains U-turn at different depths; a finished chain still rides along in the
batched leapfrog but is **masked out** (`mx.where` on a per-chain `cont` flag), so it's frozen
correctly. The doubling loop stops at `max_tree_depth` or when no chain is still going. The gap
vs JAX's traced `while_loop` is the wasted leapfrogs on already-stopped chains (mean tree depth
<< max is the tell). Trajectory length is adaptive, which removes the fixed-L resonance that
eps-jitter papers over for fixed-L HMC.

Pair with `warmup` for (eps, dense M); NUTS only replaces the trajectory length.
"""
import mlx.core as mx
import numpy as np

DMAX = 1000.0      # divergence threshold on the Hamiltonian error
NEG = -1e30        # stand-in for log-weight 0 (divergent leaf); finite to keep logaddexp NaN-free


def expit(x):                                  # logistic; finite NEG keeps this NaN-free
    return 1.0 / (1.0 + mx.exp(-x))


def logaddexp(a, b):
    m = mx.maximum(a, b)
    return m + mx.log(mx.exp(a - m) + mx.exp(b - m))


def wsel(mask, a, b):                          # per-chain select; broadcasts (N,) over (N,D)
    return mx.where(mask[:, None] if a.ndim == 2 else mask, a, b)


def make_nuts(logp_single, Minv_np, eps, max_tree_depth=10):
    grad_logp = mx.vmap(mx.grad(logp_single))
    logp = mx.vmap(logp_single)
    Minv = mx.array(Minv_np.astype(np.float32))
    Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))
    eps = float(eps)

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

    def build(theta, r, lw0, depth, dirn, key):
        """Recursive subtree builder. Returns the two endpoints, the multinomial proposal from
        this subtree, its total log-weight, and a per-chain continue flag."""
        if depth == 0:                          # base: a single leapfrog in `dirn`
            th1, r1 = leap(theta, r, (dirn * eps)[:, None])
            lw1 = joint(th1, r1)
            div = (lw0 - lw1 > DMAX) | mx.isnan(lw1)
            return th1, r1, th1, r1, th1, mx.where(div, NEG, lw1), ~div

        kL, kR, ks = mx.random.split(key, 3)
        tm, rm, tp, rp, p1, lw1, s1 = build(theta, r, lw0, depth - 1, dirn, kL)
        lt, lr = wsel(dirn < 0, tm, tp), wsel(dirn < 0, rm, rp)      # extend the leading edge
        tm2, rm2, tp2, rp2, p2, lw2, s2 = build(lt, lr, lw0, depth - 1, dirn, kR)

        ftm, frm = wsel(dirn < 0, tm2, tm), wsel(dirn < 0, rm2, rm)  # stitched full endpoints
        ftp, frp = wsel(dirn < 0, tp, tp2), wsel(dirn < 0, rp, rp2)
        pick2 = mx.random.uniform(shape=(theta.shape[0],), key=ks) < expit(lw2 - lw1)
        prop = wsel(pick2, p2, p1)
        lw = logaddexp(lw1, lw2)
        s = s1 & s2 & no_uturn(ftm, frm, ftp, frp)
        # If the first half already stopped, keep first-half-only state (mask the second half).
        return (wsel(s1, ftm, tm), wsel(s1, frm, rm), wsel(s1, ftp, tp), wsel(s1, frp, rp),
                wsel(s1, prop, p1), mx.where(s1, lw, lw1), s1 & s)

    def step(theta, key):
        N = theta.shape[0]
        km, k = mx.random.split(key, 2)
        r0 = mx.random.normal(shape=theta.shape, key=km) @ Mhalf_T   # ~ N(0, M)
        lw0 = joint(theta, r0)
        tm = tp = theta
        rm = rp = r0
        sample, lw_tree = theta, lw0
        cont = mx.array(np.ones(N, dtype=bool))
        depths = mx.zeros((N,), dtype=mx.int32)

        for depth in range(max_tree_depth):
            k, kdir, ksub, ksel = mx.random.split(k, 4)
            dirn = mx.where(mx.random.uniform(shape=(N,), key=kdir) < 0.5, -1.0, 1.0)
            depths = depths + cont.astype(mx.int32)
            lt, lr = wsel(dirn < 0, tm, tp), wsel(dirn < 0, rm, rp)
            ntm, nrm, ntp, nrp, prop, lw_sub, s_sub = build(lt, lr, lw0, depth, dirn, ksub)

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
            cont = cont & s_sub & no_uturn(new_tm, new_rm, new_tp, new_rp)
            mx.eval(cont, sample, tm, tp, rm, rp, lw_tree, depths)
            if cont.sum().item() == 0:          # host-side early stop once all chains U-turned
                break

        return sample, depths

    return step


def run_nuts(logp_single, theta0, n_samples, eps, Minv_np, key, max_tree_depth=10):
    """Returns (chain (T,N,D), mean_tree_depth, max_tree_depth_seen). theta0 already warmed.
    max-vs-mean depth is the masking-overhead tell: the batch pays the deepest chain's cost."""
    step = make_nuts(logp_single, Minv_np, eps, max_tree_depth)
    chain, depth_sum, depth_max, theta = [], 0.0, 0, theta0
    for _ in range(n_samples):
        key, k = mx.random.split(key, 2)
        theta, depths = step(theta, k)
        mx.eval(theta, depths)
        chain.append(theta)
        depth_sum += float(depths.mean())
        depth_max = max(depth_max, int(depths.max()))
    return mx.stack(chain, axis=0), depth_sum / n_samples, depth_max
