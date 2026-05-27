"""Preconditioned HMC: mass matrix M = Sigma^{-1} makes the dynamics isotropic.

This closes the loop from the affine-invariance discussion: the mass matrix is
HMC's version of affine invariance -- but you must *supply* it (here we use the
true Sigma; in practice you'd estimate it during warmup, as NUTS/Stan do). With
the right M, HMC mixes with far fewer, cheaper leapfrog steps.

Compares ensemble vs identity-mass HMC vs preconditioned HMC by ESS/sec.
"""
import time

import mlx.core as mx
import numpy as np

import ess  # shared target, the other two samplers, and the ESS machinery


def make_phmc(logp_single, eps, n_leap, Minv, Mhalf):
    grad_logp = mx.vmap(mx.grad(logp_single))
    logp = mx.vmap(logp_single)
    Minv = mx.array(Minv)          # M^{-1} = Sigma   (drift + kinetic energy)
    Mhalf_T = mx.transpose(mx.array(Mhalf))   # chol(Sigma^{-1})^T  (momentum draw)

    def kinetic(p):                # 0.5 p^T M^{-1} p
        return 0.5 * ((p @ Minv) * p).sum(1)

    @mx.compile
    def step(q, key):
        n, _ = q.shape
        k_p, k_acc = mx.random.split(key, 2)
        z = mx.random.normal(shape=q.shape, key=k_p)
        p0 = z @ Mhalf_T           # ~ N(0, M)
        logp_q, K0 = logp(q), kinetic(p0)

        qq = q
        p = p0 + 0.5 * eps * grad_logp(qq)
        for i in range(n_leap):
            qq = qq + eps * (p @ Minv)          # drift uses M^{-1} = Sigma
            if i != n_leap - 1:
                p = p + eps * grad_logp(qq)
        p = p + 0.5 * eps * grad_logp(qq)

        log_accept = (logp(qq) - logp_q) + (K0 - kinetic(p))
        accept = mx.log(mx.random.uniform(shape=(n,), key=k_acc)) < log_accept
        return mx.where(accept[:, None], qq, q), accept.sum()

    return step


def run_phmc(q0, n_steps, burn, eps, n_leap, key, Minv, Mhalf):
    step = make_phmc(ess.logp_single, eps, n_leap, Minv, Mhalf)
    chain, q = [], q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, _ = step(q, k)
        mx.eval(q)
        if t >= burn:
            chain.append(q)
    return mx.stack(chain, axis=0)


if __name__ == "__main__":
    Sigma = ess.Sigma_np
    Minv = Sigma                              # M^{-1}
    Mhalf = np.linalg.cholesky(np.linalg.inv(Sigma))   # chol(M), M = Sigma^{-1}
    key = mx.random.key(0)

    key, ki = mx.random.split(key)
    ens0 = mx.random.normal(shape=(2000, 2), key=ki) * 5.0
    t0 = time.time()
    ec = ess.run_ensemble(ens0, 2000, 500, key)
    mx.eval(ec)
    e_ess, e_dt = ess.report(ec, "ensemble (no grad, no tuning)", time.time() - t0)

    key, ki = mx.random.split(key)
    q0 = mx.random.normal(shape=(1000, 2), key=ki) * 5.0
    t0 = time.time()
    hc = ess.run_hmc_chain(q0, 1500, 500, 0.15, 40, key)
    mx.eval(hc)
    h_ess, h_dt = ess.report(hc, "HMC identity mass (eps=0.15, L=40)", time.time() - t0)

    key, ki = mx.random.split(key)
    q0p = mx.random.normal(shape=(1000, 2), key=ki) * 5.0
    t0 = time.time()
    pc = run_phmc(q0p, 1500, 500, 0.7, 6, key, Minv, Mhalf)
    mx.eval(pc)
    p_ess, p_dt = ess.report(pc, "HMC preconditioned M=Sigma^-1 (eps=0.7, L=6)", time.time() - t0)

    print("\n=== ESS/sec ===")
    print(f"  ensemble:       {e_ess / e_dt:>10,.0f}")
    print(f"  HMC identity:   {h_ess / h_dt:>10,.0f}")
    print(f"  HMC precond:    {p_ess / p_dt:>10,.0f}   "
          f"({(p_ess / p_dt) / (h_ess / h_dt):.1f}x identity HMC, "
          f"L=6 vs 40 -> {7}/{41} the gradients per step)")
