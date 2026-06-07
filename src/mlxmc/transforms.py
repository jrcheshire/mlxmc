"""Constrained-parameter transforms (bijectors) so real models work without hand-written
Jacobians.

The samplers all run in *unconstrained* R^D. Real models have constrained parameters --
scales/variances (> 0), probabilities and bounded quantities (in [low, high]). The textbook
recipe is to sample an unconstrained `u`, map it to the constrained `x = forward(u)`, and add
the log-Jacobian `log|d forward / d u|` to the log-density so the pushforward is the intended
distribution. Getting that Jacobian wrong silently biases the posterior with no error -- which
is exactly the trap a bare `logp(x)` interface leaves the user in. `constrain()` does it for
you (cf. Stan's transforms / NumPyro's auto-transforms / TFP bijectors).

Scope is deliberately small: a per-coordinate composition of `Identity`, `Exp` (positive),
and `Sigmoid(low, high)` (bounded). This is *not* a distribution library or a model DSL --
you still write the constrained-space log-density yourself; this just removes the
unconstraining + Jacobian bookkeeping.

Everything is written against an array-module handle `xp` (either `mlx.core` or `numpy`), so
the same math powers the differentiable MLX path used inside the sampler (`forward`,
`log_det_jacobian`) and the numpy path used to map finished draws back to the natural space
(`forward_np`, via `Result.constrained()`).
"""
import math

import mlx.core as mx
import numpy as np


def _softplus(z, xp):
    """log(1 + exp(z)), overflow-safe: max(z,0) + log1p(exp(-|z|))."""
    return xp.maximum(z, 0.0) + xp.log1p(xp.exp(-xp.abs(z)))


class Identity:
    """No constraint: x = u."""

    def fwd(self, u, xp):
        return u

    def inv(self, x, xp):
        return x

    def ldj(self, u, xp):
        return u * 0.0


class Exp:
    """Positive constraint x > 0 via x = exp(u); log|dx/du| = u."""

    def fwd(self, u, xp):
        return xp.exp(u)

    def inv(self, x, xp):
        return xp.log(x)

    def ldj(self, u, xp):
        return u


class Sigmoid:
    """Bounded constraint x in (low, high) via x = low + (high-low)*sigmoid(u).

    log|dx/du| = log(high-low) + log sigmoid(u) + log sigmoid(-u)
               = log(high-low) - softplus(-u) - softplus(u).
    """

    def __init__(self, low=0.0, high=1.0):
        if high <= low:
            raise ValueError(f"Sigmoid needs high > low, got low={low}, high={high}")
        self.low, self.high = float(low), float(high)

    def fwd(self, u, xp):
        return self.low + (self.high - self.low) / (1.0 + xp.exp(-u))

    def inv(self, x, xp):
        s = (x - self.low) / (self.high - self.low)
        return xp.log(s) - xp.log1p(-s)

    def ldj(self, u, xp):
        # math.log (a Python float) keeps this backend-agnostic: a numpy scalar here would
        # make numpy try to convert the traced MLX array under compile/vmap and blow up.
        return math.log(self.high - self.low) - _softplus(-u, xp) - _softplus(u, xp)


class Transform:
    """A per-coordinate stack of bijections (one per dimension). Maps unconstrained `u`
    (shape `(..., D)`) to constrained `x` of the same shape, elementwise per coordinate."""

    def __init__(self, bijections):
        self.bijections = list(bijections)

    @property
    def n_dim(self):
        return len(self.bijections)

    def _apply(self, arr, method, xp):
        cols = [getattr(b, method)(arr[..., d], xp) for d, b in enumerate(self.bijections)]
        return xp.stack(cols, axis=-1)

    def forward(self, u):
        """Unconstrained -> constrained, MLX (differentiable: use inside the sampler)."""
        return self._apply(u, "fwd", mx)

    def inverse(self, x):
        """Constrained -> unconstrained, MLX (e.g. to place an initial point)."""
        return self._apply(x, "inv", mx)

    def forward_np(self, u):
        """Unconstrained -> constrained, numpy (for mapping finished draws; no GPU touch)."""
        return self._apply(np.asarray(u), "fwd", np)

    def log_det_jacobian(self, u):
        """sum_d log|d forward_d / d u_d|, MLX. Added to the log-density by `constrain`."""
        cols = [b.ldj(u[..., d], mx) for d, b in enumerate(self.bijections)]
        return mx.stack(cols, axis=-1).sum(axis=-1)


def constrain(logp_constrained, transform):
    """Wrap a constrained-space log-density for unconstrained sampling.

    `logp_constrained(x) -> scalar` is your model on the natural (constrained) parameters.
    Returns `(logp_unconstrained, transform)` where

        logp_unconstrained(u) = logp_constrained(transform.forward(u))
                                + transform.log_det_jacobian(u)

    Sample `logp_unconstrained` with any mlxmc sampler, then map draws back with
    `transform.forward` (or attach the transform to the `Result` and call
    `Result.constrained()` / `Result.summary()`).
    """
    def logp_unconstrained(u):
        return logp_constrained(transform.forward(u)) + transform.log_det_jacobian(u)

    return logp_unconstrained, transform
