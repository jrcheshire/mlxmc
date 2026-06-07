"""A real little model, end to end: infer (mu, sigma) of a Normal from data, with the
positivity constraint sigma > 0 handled by a parameter transform so we never hand-write a
Jacobian. This exercises the whole trust layer:

    constrain(logp, transform)  ->  nuts_warmup  ->  run_nuts  ->  Result.summary()

with sigma sampled in unconstrained (log) space, the log-Jacobian added automatically, and
the summary (mean / sd / MCSE / bulk-tail ESS / R-hat / divergences) reported back in the
natural parameter space.

Run:  python examples/constrained_model.py
"""
import mlx.core as mx
import numpy as np

from mlxmc import Exp, Identity, Transform, constrain, nuts_warmup, run_nuts

# --- synthetic data: 200 draws from Normal(mu=2.0, sigma=1.5) ---
rng = np.random.default_rng(0)
TRUE_MU, TRUE_SIGMA = 2.0, 1.5
data = mx.array(rng.normal(TRUE_MU, TRUE_SIGMA, size=200).astype(np.float32))


def logp_constrained(theta):
    """Model on the natural parameters theta = (mu, sigma>0): Gaussian likelihood + weak
    priors mu ~ N(0, 10^2), sigma ~ HalfNormal(5)."""
    mu, sigma = theta[0], theta[1]
    ll = -0.5 * (((data - mu) / sigma) ** 2).sum() - data.shape[0] * mx.log(sigma)
    lp = -0.5 * (mu / 10.0) ** 2 - 0.5 * (sigma / 5.0) ** 2
    return ll + lp


# mu is unconstrained (Identity); sigma is positive (Exp). constrain() adds the log-Jacobian
# so we can sample in unconstrained space and still target the intended posterior.
logp_u, transform = constrain(logp_constrained, Transform([Identity(), Exp()]))


if __name__ == "__main__":
    key = mx.random.key(0)
    k_init, k_warm, k_nuts = mx.random.split(key, 3)
    u0 = mx.random.normal(shape=(8, 2), key=k_init) * 0.5     # unconstrained inits (sigma ~ 1)
    u_last, eps, Minv = nuts_warmup(logp_u, u0, n_warmup=500, key=k_warm)
    res = run_nuts(logp_u, u_last, n_samples=1000, eps=eps, Minv_np=Minv, key=k_nuts)
    res.transform = transform                                 # report in natural (mu, sigma) space
    res.param_names = ["mu", "sigma"]

    print(f"data: {data.shape[0]} points ~ Normal(mu={TRUE_MU}, sigma={TRUE_SIGMA})")
    print(f"sampled {res.n_chains} chains x {res.n_draws} draws in unconstrained space\n")
    print("posterior summary (natural parameter space):")
    res.summary()
