"""mlxmc — MCMC samplers in Apple MLX.

Affine-invariant ensemble, HMC (identity / preconditioned), Stan-style warmup
adaptation, and NUTS, plus ESS diagnostics and a set of example targets. Every
sampler takes a single-point log-density `logp(x) -> scalar` for `x` of shape
`(D,)`; batching over chains/walkers is handled internally with `vmap`.
"""
from mlxmc import targets
from mlxmc.diagnostics import autocorr_1d, integrated_time, report
from mlxmc.ensemble import make_sampler, run_ensemble
from mlxmc.hmc import make_hmc, run_hmc
from mlxmc.nuts import make_nuts, run_nuts
from mlxmc.preconditioned import make_phmc, run_phmc
from mlxmc.warmup import DualAveraging, make_warmup_step, run_chain, warmup

__version__ = "0.1.0"

__all__ = [
    "make_sampler", "run_ensemble",
    "make_hmc", "run_hmc",
    "make_phmc", "run_phmc",
    "DualAveraging", "make_warmup_step", "warmup", "run_chain",
    "make_nuts", "run_nuts",
    "autocorr_1d", "integrated_time", "report",
    "targets",
]
