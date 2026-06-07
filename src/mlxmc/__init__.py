"""mlxmc — MCMC samplers in Apple MLX.

Affine-invariant ensemble, HMC (identity / preconditioned), Stan-style warmup
adaptation, and NUTS, plus ESS diagnostics and a set of example targets. Every
sampler takes a single-point log-density `logp(x) -> scalar` for `x` of shape
`(D,)`; batching over chains/walkers is handled internally with `vmap`.
"""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from mlxmc import targets
from mlxmc.diagnostics import (
    autocorr_1d,
    ess_bulk,
    ess_mean,
    ess_tail,
    integrated_time,
    mcse_mean,
    report,
    rhat,
)
from mlxmc.ensemble import make_sampler, run_ensemble
from mlxmc.hmc import make_hmc, run_hmc
from mlxmc.nuts import make_nuts, nuts_warmup, run_nuts
from mlxmc.preconditioned import make_phmc, run_phmc
from mlxmc.result import Result
from mlxmc.transforms import Exp, Identity, Sigmoid, Transform, constrain
from mlxmc.warmup import DualAveraging, make_warmup_step, run_chain, warmup

try:                                            # single source of truth: the installed metadata
    __version__ = _version("mlxmc")
except PackageNotFoundError:                     # not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "make_sampler", "run_ensemble",
    "make_hmc", "run_hmc",
    "make_phmc", "run_phmc",
    "DualAveraging", "make_warmup_step", "warmup", "run_chain",
    "make_nuts", "run_nuts", "nuts_warmup",
    "Result",
    "Transform", "Identity", "Exp", "Sigmoid", "constrain",
    "autocorr_1d", "integrated_time", "report",
    "rhat", "ess_bulk", "ess_tail", "ess_mean", "mcse_mean",
    "targets",
]
