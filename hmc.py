"""Hamiltonian Monte Carlo in MLX, on the same target as ensemble_sampler.py.

MLX transforms on display:
  - mx.grad  : gradient of the single-point log-density (the thing HMC needs
               and the ensemble sampler didn't)
  - mx.vmap  : compose grad ∘ vmap to batch the gradient over all chains
  - mx.compile : fuse the L-step leapfrog + Metropolis accept into one graph

Run with an identity mass matrix (no preconditioning) so the contrast with
the affine-invariant ensemble on the ill-conditioned target is visible.
"""
import time

import mlx.core as mx
import numpy as np


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
    samples = mx.stack(chain, axis=0).reshape(-1, q0.shape[1])
    return samples, float(accepted) / (n_steps * q0.shape[0])


if __name__ == "__main__":
    # Same nasty target as the ensemble sampler: corr 0.9, 25:1 variance ratio.
    mu_true = np.array([1.0, -2.0])
    Sigma_true = np.array([[25.0, 4.5], [4.5, 1.0]])
    Sig_inv = mx.array(np.linalg.inv(Sigma_true))
    mu = mx.array(mu_true)

    def logp_single(x):
        d = x - mu
        return -0.5 * (d @ Sig_inv @ d)

    n_chains, n_steps, burn = 1000, 1500, 500
    eps, n_leap = 0.15, 40          # identity mass matrix; tuned by hand for stability
    key = mx.random.key(0)
    key, k_init = mx.random.split(key)
    q0 = mx.random.normal(shape=(n_chains, 2), key=k_init) * 5.0

    t0 = time.time()
    samples, acc = run_hmc(logp_single, q0, n_steps, burn, eps, n_leap, key)
    dt = time.time() - t0

    s = np.array(samples)
    grad_evals = n_steps * n_chains * (n_leap + 1)
    print(f"chains {n_chains}  steps {n_steps}  eps {eps}  L {n_leap}  post-burn samples {s.shape[0]:,}")
    print(f"acceptance fraction: {acc:.3f}   (HMC target ~0.6-0.9)")
    print(f"wall time: {dt:.2f}s   ->  {grad_evals / dt / 1e6:,.1f}M batched grad-evals/sec")
    print(f"gradient evals: {grad_evals:,}  (HMC pays L+1={n_leap+1} grads/sample; the ensemble paid ~1 density, 0 grads)")
    print("\n            mean (true -> recovered)        std (true -> recovered)")
    for i in range(2):
        print(f"  dim {i}:   {mu_true[i]:+.3f} -> {s[:, i].mean():+.3f}"
              f"           {np.sqrt(Sigma_true[i, i]):.3f} -> {s[:, i].std():.3f}")
    print(f"\n  recovered corr: {np.corrcoef(s.T)[0, 1]:+.3f}   (true +0.900)")
