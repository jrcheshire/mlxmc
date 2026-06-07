# Changelog

All notable changes to `mlxmc` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-06

The "trust layer": convergence diagnostics, a unified result type, surfaced NUTS
divergences, and constrained-parameter transforms.

### Added
- Native convergence diagnostics in `diagnostics`: rank-normalized split-R̂, bulk/tail-ESS,
  and MCSE (Vehtari et al. 2021), validated against ArviZ.
- `Result` — the unified return type for every sampler: draws in `(chain, draw, dim)` layout,
  `sample_stats`, `.flat`, `.summary()`, `.constrained()`, and `.to_arviz()` (optional
  `[arviz]` extra).
- NUTS now reports divergences via `Result.sample_stats["diverging"]`.
- Constrained-parameter transforms (`transforms`): composable bijectors
  (`Identity`/`Exp`/`Sigmoid`) + `constrain()`, which adds the log-Jacobian automatically.
- `examples/constrained_model.py`.

### Changed
- **Breaking:** every `run_*` now returns a `Result` instead of the previous flat
  `(samples, accept_frac)` / `(steps, chains, D)` / `run_nuts` depth-tuple returns.
- `__version__` is single-sourced from the installed package metadata.

## [0.1.1] — 2026-06-04

Initial public release.

- Affine-invariant ensemble sampler (Goodman & Weare 2010).
- Hamiltonian Monte Carlo: identity-mass (`hmc`) and preconditioned (`preconditioned`).
- Stan-style warmup: dual-averaging step size + windowed dense mass-matrix estimation (`warmup`).
- NUTS (multinomial; Hoffman & Gelman 2014), vectorized over chains, with a NUTS-specific warmup (`nuts`).
- ESS / integrated-autocorrelation diagnostics (`diagnostics`).
- Example targets with known moments: correlated Gaussian, banana, centered / non-centered funnel (`targets`).

[0.2.0]: https://github.com/jrcheshire/mlxmc/releases/tag/v0.2.0
[0.1.1]: https://github.com/jrcheshire/mlxmc/releases/tag/v0.1.1
