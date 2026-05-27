"""Affine-invariant ensemble sampler (Goodman & Weare 2010) in MLX.

This is the algorithm behind `emcee`. It's gradient-free and invariant to
affine transforms of parameter space, so a badly-conditioned (elongated,
correlated) target is sampled as easily as an isotropic one -- no tuning.

MLX transforms on display:
  - mx.vmap     : batch a single-point log-density over all walkers
  - mx.compile  : fuse the stretch-move sweep into one graph
  - mx.random   : JAX-style functional keys (split per step), so the
                  compiled step is pure
"""
import time

import mlx.core as mx
import numpy as np


def make_sampler(logp_single, n_dim, a=2.0):
    """Return a compiled half-ensemble update for the G&W stretch move."""
    logp = mx.vmap(logp_single)  # (m, D) -> (m,)

    @mx.compile
    def update_half(active, complement, key):
        m = active.shape[0]
        k_part, k_z, k_acc = mx.random.split(key, 3)
        # Each active walker picks a partner from the *complementary* half.
        j = mx.random.randint(0, complement.shape[0], (m,), key=k_part)
        partners = mx.take(complement, j, axis=0)
        # Stretch factor z ~ g(z) ∝ 1/sqrt(z) on [1/a, a].
        u = mx.random.uniform(shape=(m,), key=k_z)
        z = ((a - 1.0) * u + 1.0) ** 2 / a
        proposal = partners + z[:, None] * (active - partners)
        # Metropolis accept with the (D-1) stretch Jacobian.
        log_ratio = (n_dim - 1) * mx.log(z) + logp(proposal) - logp(active)
        accept = mx.log(mx.random.uniform(shape=(m,), key=k_acc)) < log_ratio
        new_active = mx.where(accept[:, None], proposal, active)
        return new_active, accept.sum()

    return update_half


def run(logp_single, ensemble, n_steps, burn, key, a=2.0):
    n_walkers, n_dim = ensemble.shape
    half = n_walkers // 2
    update_half = make_sampler(logp_single, n_dim, a)

    chain, accepted = [], mx.array(0)
    e = ensemble
    for t in range(n_steps):
        key, k0, k1 = mx.random.split(key, 3)
        h0, h1 = e[:half], e[half:]
        h0, n0 = update_half(h0, h1, k0)      # update half 0 against half 1
        h1, n1 = update_half(h1, h0, k1)      # update half 1 against new half 0
        e = mx.concatenate([h0, h1], axis=0)
        accepted = accepted + n0 + n1
        mx.eval(e, accepted)                  # keep the lazy graph shallow
        if t >= burn:
            chain.append(e)
    samples = mx.stack(chain, axis=0).reshape(-1, n_dim)
    accept_frac = float(accepted) / (n_steps * n_walkers)
    return samples, accept_frac


if __name__ == "__main__":
    # A deliberately nasty target: strong correlation, ~25:1 variance ratio.
    mu_true = np.array([1.0, -2.0])
    Sigma_true = np.array([[25.0, 4.5], [4.5, 1.0]])   # corr = 0.9
    Sig_inv = mx.array(np.linalg.inv(Sigma_true))
    mu = mx.array(mu_true)

    def logp_single(x):                      # x: (D,) -> scalar
        d = x - mu
        return -0.5 * (d @ Sig_inv @ d)

    n_walkers, n_steps, burn = 2000, 3000, 1000
    key = mx.random.key(0)
    key, k_init = mx.random.split(key)
    ensemble = mx.random.normal(shape=(n_walkers, 2), key=k_init) * 5.0  # broad, generic start

    t0 = time.time()
    samples, acc = run(logp_single, ensemble, n_steps, burn, key)
    dt = time.time() - t0

    s = np.array(samples)
    print(f"walkers {n_walkers}  steps {n_steps}  post-burn samples {s.shape[0]:,}")
    print(f"acceptance fraction: {acc:.3f}   (healthy stretch-move range ~0.2-0.5)")
    print(f"wall time: {dt:.2f}s   ->  {s.shape[0] / dt:,.0f} samples/sec")
    print("\n            mean (true -> recovered)        std (true -> recovered)")
    for i in range(2):
        print(f"  dim {i}:   {mu_true[i]:+.3f} -> {s[:, i].mean():+.3f}"
              f"           {np.sqrt(Sigma_true[i, i]):.3f} -> {s[:, i].std():.3f}")
    print(f"\n  recovered corr: {np.corrcoef(s.T)[0, 1]:+.3f}   (true +0.900)")
