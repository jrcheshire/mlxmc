"""Example / benchmark target distributions, each a single-point log-density
`logp(x) -> scalar` suitable for any sampler in mlxmc.

  - Gaussian: strongly correlated, ill-conditioned -- easy for the affine-invariant
    ensemble, hard for identity-mass HMC. The canonical correctness target.
  - Banana (Haario twisted Gaussian): a curved ridge; local covariance rotates along it,
    so no single linear preconditioner (constant M) or affine map is right everywhere.
  - Funnel (Neal): v ~ N(0, 3^2), x_i | v ~ N(0, exp(v)). The scale of x spans orders of
    magnitude with v, so a constant M is wrong everywhere at once and the neck (v << 0)
    has stiff gradients that diverge in fp32. The honest fix is geometry-aware coordinates.
  - Non-centered funnel: sample (v, x̃) with x = x̃·exp(v/2); the v-dependent scale drops
    out, leaving a product of independent Gaussians -- the reparametrization that fixes HMC.

The TRUTH dicts give (dim index, true mean, true std) for the cleanest diagnostics.
"""
import mlx.core as mx
import numpy as np

# --------------------------------------------------------------- correlated Gaussian
GAUSSIAN_MU = np.array([1.0, -2.0])
GAUSSIAN_SIGMA = np.array([[25.0, 4.5], [4.5, 1.0]])   # corr 0.9, 25:1 variance ratio
_gauss_mu = mx.array(GAUSSIAN_MU)
_gauss_sig_inv = mx.array(np.linalg.inv(GAUSSIAN_SIGMA))


def gaussian_logp(x):                          # x: (D,) -> scalar
    d = x - _gauss_mu
    return -0.5 * (d @ _gauss_sig_inv @ d)


# --------------------------------------------------------------- banana (Haario)
B_BANANA = 0.05   # curvature; Var[x2] = 1 + 2 B^2 * 100^2


def banana_logp(x):
    """phi(x) = (x1, x2 + B x1^2 - 100 B, x3, ...) ~ N(0, diag(100, 1, 1, ...))."""
    x1, x2 = x[0], x[1]
    twisted = x2 + B_BANANA * x1 * x1 - 100.0 * B_BANANA
    rest = 0.5 * (x[2:] * x[2:]).sum() if x.shape[0] > 2 else 0.0
    return -(x1 * x1) / 200.0 - 0.5 * twisted * twisted - rest


BANANA_TRUTH = {
    "x1": (0, 0.0, 10.0),                                  # N(0, 100)
    "x2": (1, 0.0, np.sqrt(1.0 + 2.0 * B_BANANA**2 * 100.0**2)),
}


# --------------------------------------------------------------- funnel (Neal)
def funnel_logp(z):
    """v = z[0] ~ N(0, 9); x = z[1:] | v ~ N(0, exp(v)). The -0.5*(D-1)*v term is the
    v-dependent normalization -- it's what makes this a funnel, not a free-floating v."""
    v, x = z[0], z[1:]
    n_x = z.shape[0] - 1
    return -(v * v) / 18.0 - 0.5 * n_x * v - 0.5 * mx.exp(-v) * (x * x).sum()


def funnel_nc_logp(z):
    """Non-centered funnel: sample (v, x̃) with x̃ ~ N(0,1), and x = x̃·exp(v/2). In these
    coordinates the v-dependent scale vanishes from the density, leaving a *product of
    independent Gaussians* (v ~ N(0,9), x̃ ~ N(0,1)) -- no funnel geometry, so HMC's global
    metric is now correct everywhere. v's marginal is unchanged (N(0,9)), so the same
    truth/diagnostics apply, and this should flip the centered-funnel result."""
    v, xt = z[0], z[1:]
    return -(v * v) / 18.0 - 0.5 * (xt * xt).sum()


FUNNEL_TRUTH = {
    "v": (0, 0.0, 3.0),                                    # N(0, 9): the honest mixing test
}
