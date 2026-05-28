"""Preconditioned HMC: mass matrix M = Sigma^{-1} makes the dynamics isotropic.

This closes the loop from the affine-invariance discussion: the mass matrix is
HMC's version of affine invariance -- but you must *supply* it (here you'd pass
the true Sigma; in practice you'd estimate it during warmup, as NUTS/Stan do).
With the right M, HMC mixes with far fewer, cheaper leapfrog steps.
"""
import mlx.core as mx


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


def run_phmc(logp_single, q0, n_steps, burn, eps, n_leap, key, Minv, Mhalf):
    """Sample with preconditioned (mass-matrix) HMC. Returns the structured (T, N, D) chain.

    `Minv` is M^{-1} (= the covariance you precondition with) and `Mhalf` is chol(M).
    """
    step = make_phmc(logp_single, eps, n_leap, Minv, Mhalf)
    chain, q = [], q0
    for t in range(n_steps):
        key, k = mx.random.split(key, 2)
        q, _ = step(q, k)
        mx.eval(q)
        if t >= burn:
            chain.append(q)
    return mx.stack(chain, axis=0)
