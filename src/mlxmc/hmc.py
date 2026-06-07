"""Hamiltonian Monte Carlo in MLX, with an identity mass matrix.

MLX transforms on display:
  - mx.grad  : gradient of the single-point log-density (the thing HMC needs
               and the ensemble sampler didn't)
  - mx.vmap  : compose grad ∘ vmap to batch the gradient over all chains
  - mx.compile : fuse the L-step leapfrog + Metropolis accept into one graph

Identity mass (no preconditioning), so the contrast with the affine-invariant
ensemble on an ill-conditioned target is visible. See `preconditioned` and
`warmup` for the mass-matrix versions.
"""
import mlx.core as mx

from mlxmc.result import Result


def make_hmc(logp_single, eps, n_leap):
    grad_logp = mx.vmap(mx.grad(logp_single))   # (n, D) -> (n, D)
    logp = mx.vmap(logp_single)                 # (n, D) -> (n,)

    @mx.compile
    def step(q, key):
        n, _ = q.shape
        k_p, k_acc = mx.random.split(key, 2)
        p0 = mx.random.normal(shape=q.shape, key=k_p)   # resample momentum ~ N(0, I)
        logp_q = logp(q)

        # Leapfrog: half-kick, then L drifts with full-kicks between, final half-kick.
        qq = q
        p = p0 + 0.5 * eps * grad_logp(qq)
        for i in range(n_leap):
            qq = qq + eps * p
            if i != n_leap - 1:
                p = p + eps * grad_logp(qq)
        p = p + 0.5 * eps * grad_logp(qq)

        # Metropolis on the Hamiltonian H = -logp + 0.5 |p|^2.
        logp_new = logp(qq)
        log_accept = (logp_new - logp_q) + 0.5 * ((p0 * p0).sum(1) - (p * p).sum(1))
        accept = mx.log(mx.random.uniform(shape=(n,), key=k_acc)) < log_accept
        q_new = mx.where(accept[:, None], qq, q)
        return q_new, accept.sum()

    return step


def run_hmc(logp_single, q0, n_steps, burn, eps, n_leap, key):
    """Sample with fixed-step, fixed-L HMC. Returns a `Result`."""
    step = make_hmc(logp_single, eps, n_leap)
    chain, accepted = [], mx.array(0)
    q = q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, na = step(q, k)
        accepted = accepted + na
        mx.eval(q, accepted)
        if t >= burn:
            chain.append(q)
    return Result.from_chain(mx.stack(chain, axis=0),
                             accept_frac=float(accepted) / (n_steps * q0.shape[0]))
