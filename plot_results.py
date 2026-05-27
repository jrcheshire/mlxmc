"""Render the banana/funnel story to a PNG: the target shapes, the funnel-neck contrast,
the v-marginal bias and its non-centered fix, and the ensemble's high-D degradation.

Headless (Agg) -> writes hard_targets_figure.png. Regenerates samples via the same
hard_targets machinery, so the figure always matches the current code.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from hard_targets import (banana_logp, funnel_logp, funnel_nc_logp,
                          sample_hmc, sample_ensemble, mixing)
from warmup import warmup

N_LEAP = 12


def get_samples(logp, q0, e0, key, n_leap=N_LEAP):
    """(HMC samples, ensemble samples) as flat (n, D) arrays for a 2-D target."""
    kw, kh, ke = mx.random.split(key, 3)
    q_last, eps_bar, Minv = warmup(logp, q0, 600, n_leap, kw)
    hc, _ = sample_hmc(logp, q_last, eps_bar, Minv, kh, n_leap, 1500)
    ec, _ = sample_ensemble(logp, e0, ke, 3000, 1000)
    return np.array(hc).reshape(-1, 2), np.array(ec).reshape(-1, 2)


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

    # --- quick dimension scan (NC funnel): ESS/sec vs D ---
    Ds = [2, 5, 10, 25, 50]
    hmc_eps, ens_eps = [], []
    for D in Ds:
        key, kqh, kqe, kw, kh, ke = mx.random.split(key, 6)
        q0 = mx.random.normal(shape=(1000, D), key=kqh)
        e0 = mx.random.normal(shape=(1000, D), key=kqe)
        q_last, eps_bar, Minv = warmup(funnel_nc_logp, q0, 600, 10, kw)
        hc, hdt = sample_hmc(funnel_nc_logp, q_last, eps_bar, Minv, kh, 10, 800)
        ec, edt = sample_ensemble(funnel_nc_logp, e0, ke, 1500, 800)
        mx.eval(hc, ec)
        hmc_eps.append(mixing(hc, hdt)[1])
        ens_eps.append(mixing(ec, edt)[1])

    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2))
    fig.suptitle("Where affine invariance and a global mass matrix break down "
                 "(adapted dense-M HMC vs tuning-free ensemble)", fontsize=13)

    BLUE, ORANGE = "#2166ac", "#d6604d"
    scatter(ax[0, 0], thin(b_hmc), BLUE, "Banana — HMC follows the ridge", (-35, 35), (-50, 18))
    scatter(ax[0, 1], thin(f_hmc), BLUE, "Centered funnel — HMC: neck not reached", (-25, 25), (-15, 16))
    scatter(ax[0, 2], thin(n_hmc_orig), BLUE, "Non-centered funnel — HMC: neck reached", (-25, 25), (-15, 16))
    for a in ax[0]:
        a.set_xlabel("x"); a.set_ylabel("v" if a is not ax[0, 0] else "x2")
    ax[0, 0].set_xlabel("x1")

    # ensemble funnel for contrast in the neck-penetration panel context
    scatter(ax[1, 0], thin(f_ens), ORANGE, "Centered funnel — ensemble: neck reached", (-25, 25), (-15, 16))
    ax[1, 0].set_xlabel("x"); ax[1, 0].set_ylabel("v")

    # v-marginal: the bias and its fix
    vgrid = np.linspace(-14, 16, 400)
    ax[1, 1].plot(vgrid, np.exp(-vgrid**2 / 18) / np.sqrt(2 * np.pi * 9), "k--", lw=1.5, label="truth N(0,9)")
    for v, c, lbl in [(f_hmc[:, 0], BLUE, "centered HMC (biased)"),
                      (f_ens[:, 0], ORANGE, "centered ensemble"),
                      (n_hmc[:, 0], "#1a9850", "non-centered HMC")]:
        ax[1, 1].hist(v, bins=80, range=(-14, 16), density=True, histtype="step", lw=1.8, color=c, label=lbl)
    ax[1, 1].set(title="v marginal — reparam removes the bias", xlabel="v", ylabel="density")
    ax[1, 1].legend(fontsize=8)

    # dimension scan
    ax[1, 2].semilogy(Ds, hmc_eps, "o-", color=BLUE, label="adapted HMC")
    ax[1, 2].semilogy(Ds, ens_eps, "s-", color=ORANGE, label="ensemble")
    ax[1, 2].set(title="Non-centered funnel — ESS/sec vs dimension",
                 xlabel="dimension D", ylabel="ESS/sec (sampling phase)")
    ax[1, 2].legend(fontsize=9); ax[1, 2].grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "hard_targets_figure.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
