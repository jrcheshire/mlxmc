"""Render the banana/funnel story to a PNG: the target shapes, the funnel-neck contrast,
the v-marginal bias and its non-centered fix, the ensemble's high-D degradation, and the
NUTS masking overhead (the price vectorized NUTS pays for heterogeneous trajectory lengths).

Headless (Agg) -> writes hard_targets_figure.png. Regenerates samples via the same
hard_targets machinery, so the figure always matches the current code.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from mlxmc.nuts import run_nuts
from mlxmc.targets import banana_logp, funnel_logp, funnel_nc_logp
from mlxmc.warmup import warmup

# Benchmark helpers (sampling-phase-timed runners) live in the sibling example.
from hard_targets import mixing, sample_ensemble, sample_hmc

N_LEAP = 12


def _nuts_chain(*args, **kw):
    """run_nuts (now returns a Result) -> (T,N,D numpy chain, mean depth, max depth)."""
    res = run_nuts(*args, **kw)
    td = res.sample_stats["tree_depth"]
    return np.transpose(res.samples, (1, 0, 2)), float(td.mean()), int(td.max())


def get_samples(logp, q0, e0, key, n_leap=N_LEAP):
    """(HMC samples, ensemble samples) as flat (n, D) arrays for a 2-D target."""
    kw, kh, ke = mx.random.split(key, 3)
    q_last, eps_bar, Minv = warmup(logp, q0, 600, n_leap, kw)
    hc, _ = sample_hmc(logp, q_last, eps_bar, Minv, kh, n_leap, 1500)
    ec, _ = sample_ensemble(logp, e0, ke, 3000, 1000)
    return np.array(hc).reshape(-1, 2), np.array(ec).reshape(-1, 2)


def nuts_depth(logp, key, n_chains=300, n_sample=400, n_leap=10):
    """NUTS tree-depth summary (mean, max, wall-time) for a 2-D target -- the masking-overhead
    inputs. Modest config so the centered funnel (which hits max_tree_depth) stays affordable."""
    ki, kw, kn = mx.random.split(key, 3)
    q0 = mx.random.normal(shape=(n_chains, 2), key=ki) * 1.0
    q_last, eps_bar, Minv = warmup(logp, q0, 600, n_leap, kw)
    t0 = time.time()
    _ch, mdepth, maxdepth = _nuts_chain(logp, q_last, n_sample, eps_bar, Minv, kn)
    return mdepth, maxdepth, time.time() - t0


def thin(a, n=12000, seed=0):
    idx = np.random.default_rng(seed).choice(a.shape[0], min(n, a.shape[0]), replace=False)
    return a[idx]


def scatter(ax, pts, color, title, xlim, ylim):
    ax.scatter(pts[:, 1], pts[:, 0], s=2, alpha=0.08, c=color, edgecolors="none")
    ax.set(title=title, xlim=xlim, ylim=ylim)


if __name__ == "__main__":
    key = mx.random.key(1)
    nc, nw = 1000, 2000

    # --- 2-D samples for the three targets ---
    key, ka, kb = mx.random.split(key, 3)
    b_hmc, b_ens = get_samples(banana_logp,
                               mx.random.normal(shape=(nc, 2), key=ka) * mx.array([8., 6.]),
                               mx.random.normal(shape=(nw, 2), key=kb) * mx.array([8., 6.]), key)
    key, ka, kb = mx.random.split(key, 3)
    f_hmc, f_ens = get_samples(funnel_logp, mx.random.normal(shape=(nc, 2), key=ka),
                               mx.random.normal(shape=(nw, 2), key=kb), key)
    key, ka, kb = mx.random.split(key, 3)
    n_hmc, n_ens = get_samples(funnel_nc_logp, mx.random.normal(shape=(nc, 2), key=ka),
                               mx.random.normal(shape=(nw, 2), key=kb), key)
    # Non-centered samples are (v, x_tilde); back-transform to original (v, x) for display.
    n_hmc_orig = np.column_stack([n_hmc[:, 0], n_hmc[:, 1] * np.exp(n_hmc[:, 0] / 2)])

    # --- NUTS masking overhead: tree depth + wall time, centered vs non-centered funnel ---
    key, kmc, kmn = mx.random.split(key, 3)
    c_mean, c_max, c_wall = nuts_depth(funnel_logp, kmc)
    n_mean, n_max, n_wall = nuts_depth(funnel_nc_logp, kmn)

    # --- dimension scan (NC funnel): ESS/sec + NUTS tree depth vs D ---
    Ds = [2, 5, 10, 25, 50]
    hmc_eps, ens_eps, nuts_eps, nuts_dmean, nuts_dmax = [], [], [], [], []
    for D in Ds:
        key, kqh, kqe, kw, kh, ke, kn = mx.random.split(key, 7)
        q0 = mx.random.normal(shape=(1000, D), key=kqh)
        e0 = mx.random.normal(shape=(1000, D), key=kqe)
        q_last, eps_bar, Minv = warmup(funnel_nc_logp, q0, 600, 10, kw)
        hc, hdt = sample_hmc(funnel_nc_logp, q_last, eps_bar, Minv, kh, 10, 800)
        ec, edt = sample_ensemble(funnel_nc_logp, e0, ke, 1500, 800)
        t0 = time.time()
        nch, ndm, ndx = _nuts_chain(funnel_nc_logp, q_last, 800, eps_bar, Minv, kn)
        ndt = time.time() - t0
        mx.eval(hc, ec)
        hmc_eps.append(mixing(hc, hdt)[1])
        ens_eps.append(mixing(ec, edt)[1])
        nuts_eps.append(mixing(nch, ndt)[1])
        nuts_dmean.append(ndm)
        nuts_dmax.append(ndx)

    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 4, figsize=(20.5, 9.2))
    fig.suptitle("Where affine invariance and a global mass matrix break down "
                 "(adapted dense-M HMC vs tuning-free ensemble; + NUTS masking overhead)", fontsize=13)

    BLUE, ORANGE, GREEN, GREY = "#2166ac", "#d6604d", "#1a9850", "#888888"
    scatter(ax[0, 0], thin(b_hmc), BLUE, "Banana — HMC follows the ridge", (-35, 35), (-50, 18))
    scatter(ax[0, 1], thin(f_hmc), BLUE, "Centered funnel — HMC: neck not reached", (-25, 25), (-15, 16))
    scatter(ax[0, 2], thin(n_hmc_orig), BLUE, "Non-centered funnel — HMC: neck reached", (-25, 25), (-15, 16))
    for a in (ax[0, 1], ax[0, 2]):
        a.set_xlabel("x"); a.set_ylabel("v")
    ax[0, 0].set_xlabel("x1"); ax[0, 0].set_ylabel("x2")

    # NUTS masking overhead: mean vs max tree depth (the batch pays the deepest chain).
    xpos, w = np.arange(2), 0.35
    ax[0, 3].bar(xpos - w / 2, [c_mean, n_mean], w, color=BLUE, label="mean depth")
    ax[0, 3].bar(xpos + w / 2, [c_max, n_max], w, color=GREY, label="max depth")
    ax[0, 3].axhline(10, ls=":", color="k", lw=1.2, label="max_tree_depth cap")
    for i, (mx_d, wl) in enumerate(zip([c_max, n_max], [c_wall, n_wall])):
        ax[0, 3].text(i, mx_d + 0.25, f"{wl:.0f}s wall", ha="center", fontsize=9, fontweight="bold")
    ax[0, 3].set(title="NUTS masking overhead — mean ≪ max ⇒ wasted leapfrogs",
                 xticks=xpos, ylabel="tree depth", ylim=(0, 11))
    ax[0, 3].set_xticklabels(["centered\nfunnel", "non-centered\nfunnel"])
    ax[0, 3].legend(fontsize=8, loc="upper right")

    # ensemble funnel for contrast in the neck-penetration panel context
    scatter(ax[1, 0], thin(f_ens), ORANGE, "Centered funnel — ensemble: neck reached", (-25, 25), (-15, 16))
    ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("v")

    # v-marginal: the bias and its fix
    vgrid = np.linspace(-14, 16, 400)
    ax[1, 1].plot(vgrid, np.exp(-vgrid**2 / 18) / np.sqrt(2 * np.pi * 9), "k--", lw=1.5, label="truth N(0,9)")
    for v, c, lbl in [(f_hmc[:, 0], BLUE, "centered HMC (biased)"),
                      (f_ens[:, 0], ORANGE, "centered ensemble"),
                      (n_hmc[:, 0], GREEN, "non-centered HMC")]:
        ax[1, 1].hist(v, bins=80, range=(-14, 16), density=True, histtype="step", lw=1.8, color=c, label=lbl)
    ax[1, 1].set(title="v marginal — reparam removes the bias", xlabel="v", ylabel="density")
    ax[1, 1].legend(fontsize=8)

    # dimension scan: ESS/sec vs D (now incl. NUTS)
    ax[1, 2].semilogy(Ds, hmc_eps, "o-", color=BLUE, label="adapted HMC")
    ax[1, 2].semilogy(Ds, ens_eps, "s-", color=ORANGE, label="ensemble")
    ax[1, 2].semilogy(Ds, nuts_eps, "^-", color=GREEN, label="NUTS")
    ax[1, 2].set(title="Non-centered funnel — ESS/sec vs dimension",
                 xlabel="dimension D", ylabel="ESS/sec (sampling phase)")
    ax[1, 2].legend(fontsize=9); ax[1, 2].grid(alpha=0.3, which="both")

    # NUTS tree depth vs D (NC funnel): adaptive-L grows gently, stays shallow.
    ax[1, 3].plot(Ds, nuts_dmean, "o-", color=GREEN, label="mean depth")
    ax[1, 3].plot(Ds, nuts_dmax, "s--", color=GREY, label="max depth")
    ax[1, 3].set(title="NUTS tree depth vs dimension (NC funnel)",
                 xlabel="dimension D", ylabel="tree depth", ylim=(0, 11))
    ax[1, 3].legend(fontsize=9); ax[1, 3].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "hard_targets_figure.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  | masking: centered depth {c_mean:.1f}/{c_max} {c_wall:.0f}s, "
          f"NC depth {n_mean:.1f}/{n_max} {n_wall:.0f}s")
