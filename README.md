# mlxmc

**MCMC samplers in Apple [MLX](https://github.com/ml-explore/mlx).** A small,
gradient-aware sampler toolkit built directly on the MLX transform stack
(`grad` / `vmap` / `compile`), filling the "BlackJAX-shaped gap" — MLX has no
mature probabilistic-programming layer yet.

> **Status: exploratory research code.** The samplers work and are validated
> (a test suite checks moment recovery, Σ-estimation, affine invariance, and the
> autocorrelation diagnostics on both the CPU and Metal backends), but the API is
> young and may change. Expect churn.

## What's here

The package lives under `src/mlxmc/`; runnable demos and the benchmark study are in
`examples/`.

| Module (`mlxmc.`) | Sampler / tool |
|---|---|
| `ensemble` | Affine-invariant ensemble (Goodman & Weare 2010 — the `emcee` stretch move). Gradient-free, tuning-free. `make_sampler`, `run_ensemble`. |
| `hmc` | Hamiltonian Monte Carlo, identity mass. `grad ∘ vmap` batched over chains. `make_hmc`, `run_hmc`. |
| `preconditioned` | Mass-matrix HMC (M = Σ⁻¹). `make_phmc`, `run_phmc`. |
| `warmup` | Stan-style warmup: dual-averaging step size + windowed **dense** mass-matrix estimation. `warmup`, `run_chain`. |
| `nuts` | NUTS (multinomial; Hoffman & Gelman 2014), vectorized over chains. `make_nuts`, `run_nuts`. |
| `diagnostics` | Effective sample size / integrated autocorrelation time (FFT + Sokal window); the cross-sampler **ESS/sec** metric. |
| `targets` | Example log-densities: correlated Gaussian, banana, centered / non-centered funnel, with known moments. |

| Example (`examples/`) | What it shows |
|---|---|
| `gaussian_ess.py` | Ensemble vs identity-mass HMC vs preconditioned HMC by ESS/sec on the Gaussian. |
| `warmup_validation.py` | Warmup recovers the true Σ and matches oracle ESS/sec. |
| `hard_targets.py` | Banana + funnel benchmark (`lscan` / `dscan` modes). |
| `nuts_funnel.py` | NUTS correctness on the Gaussian; `funnel` mode for the masking-overhead study. |
| `affine_invariance.py` | Empirical proof of affine invariance (same RNG → bit-identical acceptance under an affine map). |
| `plot_hard_targets.py` | Renders `hard_targets_figure.png` (needs the optional `viz` env). |

## Why MLX

`grad`, `vmap`, `jvp`/`vjp`, and `compile` transfer almost directly from JAX,
with JAX-style functional RNG keys (`mx.random.split`). The wrinkles that shape
this code:

- **No traced control-flow primitives** (no `while_loop` / `scan` / `cond`). MLX
  is eager execution plus `compile` of *static* graphs. Fixed-length unrolled
  loops (leapfrog, fixed-`L` HMC) compile fine; data-dependent trajectory length
  (NUTS) is the hard case — `mlxmc.nuts` runs every chain to a fixed `max_tree_depth`
  and **masks** finished chains.
- **fp32 on the GPU.** Apple Metal has no fp64 in hardware (MLX has fp64 only on
  the CPU backend). This is fine for sampling — Monte Carlo error (~1/√ESS) swamps
  fp32 roundoff (~1e-6) — but ill-conditioned linear algebra (covariance, Cholesky
  in warmup) is kept host-side in numpy fp64; only the leapfrog runs on the GPU.

## Install

This is a [pixi](https://pixi.sh) project (installs the package editable):

```bash
pixi install
pixi run python examples/gaussian_ess.py             # ensemble vs HMC vs preconditioned
pixi run python examples/nuts_funnel.py funnel        # several examples have demo modes
pixi run -e viz python examples/plot_hard_targets.py  # plotting needs the optional viz env
```

Or install into any environment with pip: `pip install -e .` (needs `mlx`, so arm64
macOS). Add the plotting extra with `pip install -e ".[viz]"` (matplotlib).

## Usage

Every sampler takes a single-point log-density `logp(x) -> scalar` for `x` of
shape `(D,)`; batching over walkers/chains is handled internally with `vmap`.
Positions are MLX arrays of shape `(n_chains, D)`.

```python
import mlx.core as mx
import numpy as np

# Target: a strongly correlated 2-D Gaussian (corr 0.9, 25:1 variance ratio).
# mlxmc.targets ships this one (as `gaussian_logp`) plus banana / funnel.
mu = mx.array([1.0, -2.0])
Sig_inv = mx.array(np.linalg.inv([[25.0, 4.5], [4.5, 1.0]]))

def logp(x):                              # x: (D,) -> scalar
    d = x - mu
    return -0.5 * (d @ Sig_inv @ d)

key = mx.random.key(0)
```

**Gradient-free ensemble** — no tuning, handles the ill-conditioning for free:

```python
from mlxmc import run_ensemble

key, k = mx.random.split(key)
ensemble = mx.random.normal(shape=(2000, 2), key=k) * 5.0     # (n_walkers, D)
samples, accept_frac = run_ensemble(logp, ensemble, n_steps=3000, burn=1000, key=key)
```

**HMC, hand-tuned**, and **NUTS after Stan-style warmup** (same `logp`):

```python
from mlxmc import run_hmc, warmup, run_nuts

key, k = mx.random.split(key)
q0 = mx.random.normal(shape=(1000, 2), key=k) * 5.0           # (n_chains, D)

samples, acc = run_hmc(logp, q0, n_steps=1500, burn=500,
                       eps=0.15, n_leap=40, key=key)

# Warmup adapts (eps, dense M); NUTS then adapts trajectory length itself.
q_last, eps, Minv = warmup(logp, q0, n_warmup=600, n_leap=8, key=key)
chain, mean_depth, max_depth = run_nuts(logp, q_last, n_samples=1500,
                                        eps=eps, Minv_np=Minv, key=key)
```

> **Return shapes differ by sampler.** `run_ensemble` and `run_hmc` return
> `(samples, accept_frac)` with `samples` flattened to `(n_draws, D)`.
> `run_phmc`, `run_chain` (post-warmup HMC), and `run_nuts` return a structured
> `(steps, chains, D)` chain — the layout `mlxmc.diagnostics` expects for ESS —
> and `run_nuts` additionally returns the mean/max tree depth.

## Findings

![Sampler benchmarks on the banana and funnel targets](https://raw.githubusercontent.com/jrcheshire/mlxmc/main/hard_targets_figure.png)

Validated on a corr-0.9, 25:1-variance Gaussian and on banana / funnel targets;
every number below is reproducible with the scripts in
[`examples/`](https://github.com/jrcheshire/mlxmc/tree/main/examples):

- **Affine-invariant ensemble** is the robust low-D default: gradient-free,
  tuning-free, handles ill-conditioning for free (acceptance is bit-identical
  under an affine map). But weaker per-step mixing and it degrades with dimension.
- **HMC** needs gradients and a tuned `eps`/`L`, but mixes far better
  (τ≈2 vs ≈26). A **warmup-adapted dense mass matrix** recovers the true Σ to
  <1% Frobenius error and buys ~7–11× the ESS/sec — HMC's version of affine
  invariance, earned rather than supplied.
- **Fixed-`L` HMC has a trajectory resonance:** on near-Gaussian targets, when
  `eps·L` lands near a multiple of 2π the trajectory returns to its start and
  mixing collapses. Jittering `eps` per trajectory cures it; NUTS's adaptive
  trajectory length is the principled fix.
- **NUTS** is validated exact on the Gaussian (recovered covariance 24.97 vs 25)
  and auto-tunes trajectory length, but vectorized NUTS pays a real masking cost
  when trajectory lengths are heterogeneous — with no `while_loop`, every chain
  runs to the deepest chain's tree depth, up to a ~30× wall-time penalty at the
  funnel mouth versus the same target reparametrized.
- **Geometry matters more than the sampler:** on the *centered* funnel the
  gradient-free ensemble beats a global-metric HMC, because a constant mass matrix
  is wrong everywhere when the scale is position-dependent; a **non-centered
  reparametrization** removes the geometry and makes HMC unbiased again.
- **ESS/sec is the honest efficiency metric** — acceptance fraction is a
  misleading proxy.

## Development

```bash
pixi run test                                   # full suite on the default device
MLXMC_TEST_DEVICE=cpu pixi run test             # force the CPU backend
MLXMC_TEST_DEVICE=gpu pixi run test             # force the Metal GPU
```

The suite (`tests/`) checks moment recovery for every sampler, warmup's Σ
estimate, the affine-invariance identity, and the autocorrelation-time
diagnostics. There's a GitHub Actions workflow (`.github/workflows/tests.yml`,
CPU + GPU matrix on an Apple-silicon runner), but automatic CI is **disabled** to
avoid burning macOS runner minutes — run it on demand from the Actions tab, or
re-enable the triggers in the workflow.

## References

- Goodman & Weare (2010), *Ensemble samplers with affine invariance.*
- Hoffman & Gelman (2014), *The No-U-Turn Sampler.*
- Betancourt (2017), *A Conceptual Introduction to Hamiltonian Monte Carlo.*

## License

[BSD-3-Clause](https://github.com/jrcheshire/mlxmc/blob/main/LICENSE).
