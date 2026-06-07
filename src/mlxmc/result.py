"""The unified return type for every sampler.

Before this, samplers returned a grab-bag of shapes -- `(flat, accept_frac)` for
ensemble/HMC, `(T, N, D)` for preconditioned/NUTS, plus loose extras -- so no code could be
written generically across samplers, and NUTS quietly dropped the divergence information it
already computes. `Result` fixes both: one object carrying the draws in a canonical
`(chain, draw, dim)` layout, the acceptance fraction, per-draw `sample_stats` (divergences,
tree depth), and a lazy bridge to the convergence diagnostics and to ArviZ.

Canonical axis order is `(chain, draw, dim)` to match ArviZ / the wider PPL world; the
samplers build their chains as `(T, N, D) = (draw, chain, dim)`, so `from_chain` transposes
once at the boundary and everything downstream is chain-major.
"""
from dataclasses import dataclass, field

import numpy as np

from mlxmc import diagnostics as _diag


@dataclass
class Result:
    """Posterior draws plus everything needed to judge whether to trust them.

    Attributes:
      samples: float array `(chain, draw, dim)`.
      accept_frac: mean Metropolis acceptance over the run (None if not tracked).
      sample_stats: per-draw arrays shaped `(chain, draw)` -- e.g. `diverging` (bool),
        `tree_depth` (int) for NUTS.
      independent_chains: True when the `chain` axis is genuinely independent chains (HMC /
        NUTS). False for the affine-invariant ensemble, whose walkers are correlated -- in
        which case cross-chain Rhat / ESS are optimistic and `summary()` says so.
      transform: optional `transforms.Transform` attached by `constrain()`; maps draws from
        the sampled (unconstrained) space back to the model's natural space.
      param_names: optional per-dim labels for `summary()`.
    """

    samples: np.ndarray
    accept_frac: float | None = None
    sample_stats: dict = field(default_factory=dict)
    independent_chains: bool = True
    transform: object | None = None
    param_names: list | None = None

    @classmethod
    def from_chain(cls, chain, *, accept_frac=None, sample_stats=None,
                   independent_chains=True, transform=None, param_names=None):
        """Build from a sampler's `(T, N, D) = (draw, chain, dim)` chain (mlx array or
        ndarray). `sample_stats` come in as `(T, N) = (draw, chain)` and are transposed to
        match the canonical `(chain, draw)` of `samples`."""
        arr = np.asarray(chain)
        if arr.ndim != 3:
            raise ValueError(f"expected a (draw, chain, dim) chain, got shape {arr.shape}")
        samples = np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))   # -> (chain, draw, dim)
        stats = {}
        for k, v in (sample_stats or {}).items():
            v = np.asarray(v)
            stats[k] = v.T if v.ndim == 2 else v                       # (draw, chain) -> (chain, draw)
        return cls(samples, accept_frac, stats, independent_chains, transform, param_names)

    # ---- shape accessors --------------------------------------------------------------
    @property
    def n_chains(self):
        return self.samples.shape[0]

    @property
    def n_draws(self):
        return self.samples.shape[1]

    @property
    def n_dim(self):
        return self.samples.shape[2]

    @property
    def flat(self):
        """All draws pooled across chains: `(chain * draw, dim)`."""
        return self.samples.reshape(-1, self.n_dim)

    @property
    def n_divergent(self):
        d = self.sample_stats.get("diverging")
        return int(np.asarray(d).sum()) if d is not None else None

    # ---- convergence diagnostics (per dimension) --------------------------------------
    def _per_dim(self, fn):
        return np.array([fn(self.samples[:, :, d]) for d in range(self.n_dim)])

    def rhat(self):
        """Rank-normalized split-Rhat per dim (Vehtari 2021); target < 1.01."""
        return self._per_dim(_diag.rhat)

    def ess_bulk(self):
        return self._per_dim(_diag.ess_bulk)

    def ess_tail(self):
        return self._per_dim(_diag.ess_tail)

    def mcse_mean(self):
        return self._per_dim(_diag.mcse_mean)

    # ---- constrained-space mapping ----------------------------------------------------
    def constrained(self):
        """Draws mapped to the model's natural space via the attached `transform`
        (identity if none). Returns `(chain, draw, dim)`."""
        if self.transform is None:
            return self.samples
        con = np.asarray(self.transform.forward_np(self.flat))
        return con.reshape(self.samples.shape)

    # ---- reporting --------------------------------------------------------------------
    def summary(self, constrained=True):
        """Per-parameter table (mean, sd, mcse, ess_bulk, ess_tail, r_hat) + divergence
        count. Pretty-prints and returns the rows as a list of dicts.

        `constrained=True` reports in the model's natural space when a `transform` is set.
        Diagnostics (Rhat/ESS) are always computed on the *sampled* (unconstrained) draws --
        that's where mixing actually happened; only the location/scale columns are mapped.
        """
        draws = self.constrained() if constrained else self.samples
        names = self.param_names or [f"x{d}" for d in range(self.n_dim)]
        rhat, eb, et = self.rhat(), self.ess_bulk(), self.ess_tail()
        rows = []
        for d in range(self.n_dim):
            col = draws[:, :, d]
            sd = col.std(ddof=1)
            rows.append({
                "param": names[d], "mean": float(col.mean()), "sd": float(sd),
                "mcse": float(sd / np.sqrt(eb[d])) if eb[d] > 0 else np.nan,
                "ess_bulk": float(eb[d]), "ess_tail": float(et[d]), "r_hat": float(rhat[d]),
            })

        head = f"{'param':>8} {'mean':>10} {'sd':>10} {'mcse':>9} {'ess_bulk':>9} {'ess_tail':>9} {'r_hat':>7}"
        print(head)
        print("-" * len(head))
        for r in rows:
            print(f"{r['param']:>8} {r['mean']:>10.4g} {r['sd']:>10.4g} {r['mcse']:>9.3g} "
                  f"{r['ess_bulk']:>9.0f} {r['ess_tail']:>9.0f} {r['r_hat']:>7.4f}")
        nd = self.n_divergent
        if nd is not None:
            total = self.n_chains * self.n_draws
            flag = "  <-- divergences indicate biased samples; reparametrize or lower eps" if nd else ""
            print(f"divergences: {nd:,} / {total:,}{flag}")
        if not self.independent_chains:
            print("note: ensemble walkers are correlated, not independent chains -- "
                  "r_hat/ess across them are optimistic; use ESS/sec (diagnostics.report) for efficiency")
        return rows

    # ---- interop ----------------------------------------------------------------------
    def to_arviz(self, constrained=True):
        """Convert to ArviZ (needs the optional `[arviz]` extra), so the full ArviZ
        diagnostic + plotting suite applies. Posterior is one variable `x` with a trailing
        parameter dimension; `sample_stats` carries `diverging`/`tree_depth`. Returns whatever
        `arviz.from_dict` produces for the installed version (a DataTree on arviz >= 1.0,
        an InferenceData before that)."""
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError("Result.to_arviz() requires ArviZ -- `pip install 'mlxmc[arviz]'` "
                              "(or `pixi run -e dev ...`)") from e
        draws = self.constrained() if constrained else self.samples
        groups = {"posterior": {"x": draws}}
        if self.sample_stats:
            groups["sample_stats"] = {k: np.asarray(v) for k, v in self.sample_stats.items()}
        return az.from_dict(groups)
