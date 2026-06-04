# Changelog

All notable changes to `mlxmc` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

Initial public release.

- Affine-invariant ensemble sampler (Goodman & Weare 2010).
- Hamiltonian Monte Carlo: identity-mass (`hmc`) and preconditioned (`preconditioned`).
- Stan-style warmup: dual-averaging step size + windowed dense mass-matrix estimation (`warmup`).
- NUTS (multinomial; Hoffman & Gelman 2014), vectorized over chains, with a NUTS-specific warmup (`nuts`).
- ESS / integrated-autocorrelation diagnostics (`diagnostics`).
- Example targets with known moments: correlated Gaussian, banana, centered / non-centered funnel (`targets`).

[0.1.0]: https://github.com/jrcheshire/mlxmc/releases/tag/v0.1.0
