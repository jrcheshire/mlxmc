"""Banana and funnel: targets where affine invariance and a *global* mass matrix start to
fail, so the easy-Gaussian story breaks down.

  - Banana (Haario twisted Gaussian): a curved ridge. The local covariance rotates along
    the ridge, so no single linear preconditioner (constant M) or affine map is right
    everywhere. HMC's gradients follow the curve; the ensemble's straight-line stretch
    move and a static M only approximate it.
  - Funnel (Neal): v ~ N(0, 3^2), x_i | v ~ N(0, exp(v)). The scale of x spans orders of
    magnitude with v. A constant M is wrong everywhere at once, and the neck (v << 0) has
    stiff gradients that diverge in fp32. The canonical "everything struggles" target;
    the honest fix is position-dependent curvature / reparametrization (beyond Phase 1).

For each target we compare the tuning-free ensemble against warmup-adapted dense-M HMC,
and report BOTH mixing (ESS, only trustworthy at tau>1) AND accuracy (recovered moments
vs known truth) -- a sampler can mix fast yet be biased (e.g. never reach the funnel neck).

Run:  python examples/hard_targets.py [lscan|dscan]
"""
import sys
import time

import mlx.core as mx
import numpy as np

from mlxmc.diagnostics import integrated_time, report
from mlxmc.ensemble import make_sampler
from mlxmc.targets import (B_BANANA, BANANA_TRUTH, FUNNEL_TRUTH, banana_logp,
                           funnel_logp, funnel_nc_logp)
from mlxmc.warmup import make_warmup_step, warmup


# ------------------------------------------------------------------------ runners + report
def sample_hmc(logp, q0, eps, Minv_np, key, n_leap, n_collect, n_warm=50, jitter=(0.9, 1.1)):
    """Untimed warm steps (trigger compile + settle), then a TIMED collection of n_collect
    structured (T, N, D) steps. Returns (chain, sampling_dt). q0 should already be warmed.

    `jitter` multiplies eps by U[lo, hi] each trajectory (state-independent, so detailed
    balance holds), perturbing trajectory length to break the fixed-L resonance that tanks
    HMC on near-Gaussian targets (e.g. eps*L near a multiple of 2pi). Pass None to disable."""
    step = make_warmup_step(logp, n_leap)
    eps_a = mx.array(eps, dtype=mx.float32)
    Minv_a = mx.array(Minv_np.astype(np.float32))
    Mhalf_T = mx.array(np.linalg.cholesky(np.linalg.inv(Minv_np)).T.astype(np.float32))

    def one(q, key):
        key, k, kj = mx.random.split(key, 3)
        e = eps_a if jitter is None else eps_a * mx.random.uniform(
            low=jitter[0], high=jitter[1], shape=(), key=kj)
        q, _ = step(q, k, e, Minv_a, Mhalf_T)
        return q, key

    q = q0
    for _ in range(n_warm):                                # untimed: compile + settle
        q, key = one(q, key)
    mx.eval(q)
    chain, t0 = [], time.time()
    for _ in range(n_collect):
        q, key = one(q, key)
        mx.eval(q)
        chain.append(q)
    return mx.stack(chain, axis=0), time.time() - t0


def sample_ensemble(logp, e0, key, n_collect, n_burn, a=2.0):
    """Untimed burn (compile + mix-in), then a TIMED collection. Returns (chain, sampling_dt).
    Mirrors sample_hmc so ESS/sec excludes setup for both methods alike."""
    n_walkers, n_dim = e0.shape
    half = n_walkers // 2
    update = make_sampler(logp, n_dim, a)

    def sweep(e, key):
        key, k0, k1 = mx.random.split(key, 3)
        h0, h1 = e[:half], e[half:]
        h0, _ = update(h0, h1, k0)
        h1, _ = update(h1, h0, k1)
        return mx.concatenate([h0, h1], axis=0), key

    e = e0
    for _ in range(n_burn):                                # untimed
        e, key = sweep(e, key)
    mx.eval(e)
    chain, t0 = [], time.time()
    for _ in range(n_collect):
        e, key = sweep(e, key)
        mx.eval(e)
        chain.append(e)
    return mx.stack(chain, axis=0), time.time() - t0


def n_stuck(chain_mx, tol=1e-10):
    """Count chains that never moved (zero variance over time) -- HMC chains that reject
    everything, e.g. stranded in the funnel neck. A mixing-failure tell, not just noise."""
    c = np.array(chain_mx)                                 # (T, N, D)
    var_per_chain = c.var(axis=0).sum(axis=1)              # (N,)
    return int((var_per_chain < tol).sum()), c.shape[1]


def accuracy(chain_mx, label, truth):
    """Recovered moments vs known truth, with a non-finite-row count (divergence tell)."""
    c = np.array(chain_mx).reshape(-1, np.array(chain_mx).shape[-1])
    finite_mask = np.isfinite(c).all(axis=1)
    n_bad = int((~finite_mask).sum())
    f = c[finite_mask]
    print(f"  [{label}] accuracy  ({f.shape[0]:,} finite samples"
          + (f", {n_bad:,} non-finite" if n_bad else "") + "):")
    for nm, (idx, tmean, tsd) in truth.items():
        print(f"     {nm}: mean {f[:, idx].mean():+6.2f} (true {tmean:+.2f})   "
              f"std {f[:, idx].std():6.2f} (true {tsd:.2f})")
    if "v" in truth:                                       # funnel: neck penetration
        print(f"     v range: [{f[:, 0].min():+.2f}, {f[:, 0].max():+.2f}]   "
              f"(deep neck v<-3: {100 * (f[:, 0] < -3).mean():.1f}% of samples)")
    return f


def evaluate(name, logp, q0_hmc, e0_ens, n_leap, truth, key,
             n_warmup=600, n_sample=1500, ens_collect=3000, ens_burn=1000):
    # ESS/sec times the productive sampling phase ONLY for both methods -- HMC's warmup and
    # the ensemble's burn (plus the one-time compile) are excluded as amortizable setup.
    print(f"\n================ {name} ================")
    k_warm, k_hmc, k_ens = mx.random.split(key, 3)

    q_last, eps_bar, Minv = warmup(logp, q0_hmc, n_warmup, n_leap, k_warm)
    hc, h_dt = sample_hmc(logp, q_last, eps_bar, Minv, k_hmc, n_leap, n_sample)
    mx.eval(hc)
    h_ess, _ = report(hc, f"{name}: adapted dense-M HMC (eps={eps_bar:.3f}, L={n_leap})", h_dt)
    accuracy(hc, "HMC", truth)
    stuck, total = n_stuck(hc)
    print(f"     stuck chains (never moved): {stuck:,}/{total:,}")
    print(f"     estimated M^-1 diag: {np.diag(Minv).round(2)}  (global metric)")

    ec, e_dt = sample_ensemble(logp, e0_ens, k_ens, ens_collect, ens_burn)
    mx.eval(ec)
    e_ess, _ = report(ec, f"{name}: affine-invariant ensemble (tuning-free)", e_dt)
    accuracy(ec, "ensemble", truth)

    print(f"\n  ESS/sec (sampling phase only)  ->  HMC {h_ess / h_dt:,.0f}   "
          f"ensemble {e_ess / e_dt:,.0f}   (HMC = {(h_ess / h_dt) / (e_ess / e_dt):.2f}x ensemble)")


def funnel_L_scan(key, logp=funnel_logp, label="centered", Ls=(6, 8, 12, 16, 24, 48)):
    """Sweep L on a funnel. Centered: tests whether the static-M v-bias is structural (it
    persists across L) vs bad tuning. Non-centered: tests whether the low ESS/sec is an
    eps*L trajectory-length resonance (tau_v swings with L / trajectory length) vs a real
    deficit. Reports both the v-bias and tau_v so one scan answers both."""
    print(f"\n==== funnel ({label}) HMC across L (true v: mean 0.00, std 3.00) ====")
    for L in Ls:
        key, kq, kw, kh = mx.random.split(key, 4)
        q0 = mx.random.normal(shape=(1000, 2), key=kq) * 1.0
        q_last, eps_bar, Minv = warmup(logp, q0, 600, L, kw)
        hc, _ = sample_hmc(logp, q_last, eps_bar, Minv, kh, L, 1500, jitter=None)  # expose resonance
        mx.eval(hc)
        f = np.array(hc).reshape(-1, 2)
        tau_v = integrated_time(np.array(hc)[:, :, 0])     # v dimension only
        stuck, _ = n_stuck(hc)
        print(f"  L={L:3d} eps={eps_bar:.3f} traj_len={eps_bar * L:5.2f}: "
              f"v std {f[:, 0].std():.2f}  tau_v {tau_v:6.1f}  "
              f"v<-3 {100 * (f[:, 0] < -3).mean():4.1f}%  stuck {stuck}")


def mixing(chain_mx, dt):
    """(worst-dim tau, ESS/sec). tau floored at 1 for the rate, since antithetic tau<1
    would otherwise inflate ESS/sec without bound."""
    c = np.array(chain_mx)
    T, N, D = c.shape
    tau = max(integrated_time(c[:, :, d]) for d in range(D))
    return tau, (T * N / max(tau, 1.0)) / dt


def dim_scan(key, logp, label, init, dim0_std, Ds=(2, 5, 10, 25, 50), n_leap=10):
    """Push D up: where does the (tuning-free) ensemble degrade vs adapted HMC? Reports
    worst-dim tau, ESS/sec, and dim-0 recovered std for both, per dimension."""
    print(f"\n==== {label}: ensemble vs adapted HMC across D  (dim0 true std {dim0_std:.1f}) ====")
    for D in Ds:
        key, kqh, kqe, kw, kh, ke = mx.random.split(key, 6)
        q0, e0 = init((1000, D), kqh), init((1000, D), kqe)
        q_last, eps_bar, Minv = warmup(logp, q0, 600, n_leap, kw)
        hc, h_dt = sample_hmc(logp, q_last, eps_bar, Minv, kh, n_leap, 1000)
        ec, e_dt = sample_ensemble(logp, e0, ke, 2000, 1000)
        mx.eval(hc, ec)
        h_tau, h_es = mixing(hc, h_dt)
        e_tau, e_es = mixing(ec, e_dt)
        h0 = np.array(hc).reshape(-1, D)[:, 0].std()
        e0s = np.array(ec).reshape(-1, D)[:, 0].std()
        print(f"  D={D:3d}   HMC: tau {h_tau:6.1f}  ESS/s {h_es:11,.0f}  std0 {h0:5.2f}   |   "
              f"ENS: tau {e_tau:6.1f}  ESS/s {e_es:11,.0f}  std0 {e0s:5.2f}")


def _banana_init(shape, key):
    D = shape[1]
    scale = mx.array([8.0, 6.0] + [1.0] * (D - 2))         # dims 0,1 banana; rest N(0,1)
    return mx.random.normal(shape=shape, key=key) * scale


if __name__ == "__main__":
    key = mx.random.key(0)
    n_chains, n_walkers = 1000, 2000

    if len(sys.argv) > 1 and sys.argv[1] == "lscan":
        k1, k2 = mx.random.split(key)
        funnel_L_scan(k1, funnel_logp, "centered")
        funnel_L_scan(k2, funnel_nc_logp, "non-centered")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "dscan":
        k1, k2 = mx.random.split(key)
        dim_scan(k1, funnel_nc_logp, "FUNNEL non-centered (v + (D-1) x)",
                 lambda s, k: mx.random.normal(shape=s, key=k) * 1.0, dim0_std=3.0)
        dim_scan(k2, banana_logp, "BANANA (+ nuisance dims)", _banana_init, dim0_std=10.0)
        sys.exit(0)

    # --- banana (2-D) ---
    key, kqh, kqe = mx.random.split(key, 3)
    scale = mx.array([8.0, 6.0])
    q0_b = mx.random.normal(shape=(n_chains, 2), key=kqh) * scale
    e0_b = mx.random.normal(shape=(n_walkers, 2), key=kqe) * scale
    key, ke = mx.random.split(key)
    evaluate("BANANA (B=%.2f)" % B_BANANA, banana_logp, q0_b, e0_b, n_leap=12,
             truth=BANANA_TRUTH, key=ke)

    # --- funnel (2-D: v + 1 x) --- start mild (v ~ N(0,1)) to avoid an immediate blow-up
    key, kqh, kqe = mx.random.split(key, 3)
    q0_f = mx.random.normal(shape=(n_chains, 2), key=kqh) * 1.0
    e0_f = mx.random.normal(shape=(n_walkers, 2), key=kqe) * 1.0
    key, ke = mx.random.split(key)
    evaluate("FUNNEL centered (2-D)", funnel_logp, q0_f, e0_f, n_leap=12,
             truth=FUNNEL_TRUTH, key=ke)

    # --- non-centered funnel (2-D) --- same v marginal, but the geometry is reparametrized
    # away; HMC's global metric should now be correct and the centered-funnel bias vanish.
    key, kqh, kqe = mx.random.split(key, 3)
    q0_n = mx.random.normal(shape=(n_chains, 2), key=kqh) * 1.0
    e0_n = mx.random.normal(shape=(n_walkers, 2), key=kqe) * 1.0
    key, ke = mx.random.split(key)
    evaluate("FUNNEL non-centered (2-D)", funnel_nc_logp, q0_n, e0_n, n_leap=12,
             truth=FUNNEL_TRUTH, key=ke)
