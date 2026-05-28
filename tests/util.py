"""Test helpers: reshape flat samples to structured chains and summarize per-dim stats."""
import numpy as np

from mlxmc.diagnostics import integrated_time


def structured(flat, n_steps, n_per):
    """Reshape a flat (T*N, D) sample array (row-major, T outer) back to (T, N, D).

    run_ensemble / run_hmc flatten their structured chain as stack((N,D), T).reshape(-1, D),
    so the inverse is reshape(T, N, D) with the same T (post-burn steps) and N (walkers/chains).
    """
    a = np.asarray(flat)
    return a.reshape(n_steps, n_per, a.shape[1])


def per_dim_stats(chain):
    """chain: (T, N, D). Returns a list of per-dim dicts: mean, std, tau, ess, se(of the mean)."""
    c = np.asarray(chain)
    T, N, D = c.shape
    flat = c.reshape(-1, D)
    out = []
    for d in range(D):
        tau = integrated_time(c[:, :, d])
        ess = T * N / max(tau, 1.0)
        std = flat[:, d].std()
        out.append({"mean": float(flat[:, d].mean()), "std": float(std),
                    "tau": float(tau), "ess": float(ess), "se": float(std / np.sqrt(ess))})
    return out
