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
import mlx.core as mx

from mlxmc.result import Result


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


def run_ensemble(logp_single, ensemble, n_steps, burn, key, a=2.0):
    """Sample with the affine-invariant ensemble. Returns a `Result` with
    `independent_chains=False` -- the walkers are correlated, not independent chains, so
    cross-walker Rhat/ESS are optimistic (see `Result.summary`)."""
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
    accept_frac = float(accepted) / (n_steps * n_walkers)
    return Result.from_chain(mx.stack(chain, axis=0),     # (T, n_walkers, n_dim)
                             accept_frac=accept_frac, independent_chains=False)
