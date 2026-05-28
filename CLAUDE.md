# mlxmc — MCMC samplers in Apple MLX

Exploring **MLX for scientific compute** (an under-served open-source niche), starting
with MCMC samplers — the "BlackJAX-shaped gap" (MLX has no mature PPL). Also Jamie's
first hands-on taste of the MLX transform stack, coming from JAX.

## Environment
Pixi project; installs `mlxmc` editable (src layout). Core deps `mlx`, `numpy`; `pytest` in
the env. `matplotlib` is optional — it lives in a separate `viz` pixi environment / `[viz]`
pip extra (only `plot_hard_targets.py` uses it). Run examples / tests:
- `pixi run --manifest-path ~/mlxmc/pyproject.toml python examples/<name>.py`
- `pixi run --manifest-path ~/mlxmc/pyproject.toml -e viz python examples/plot_hard_targets.py` (plotting)
- `pixi run --manifest-path ~/mlxmc/pyproject.toml test` (pytest; `MLXMC_TEST_DEVICE=cpu|gpu` pins the backend)

Now a git repo (local `main`, GitHub remote) with BSD-3-Clause LICENSE, README, and a
GitHub Actions workflow (`.github/workflows/tests.yml`, CPU + GPU matrix on an Apple-silicon
runner). **Automatic CI is disabled** (`workflow_dispatch` only) — macOS runner minutes are
billed at 10x and this is a Mac-only package, so it's not worth running on every push for now;
run on demand or re-enable the push/PR triggers. Tests run locally via `pixi run test`. See [project-mlxmc].

## Layout
**Package `src/mlxmc/`** (the library — import `from mlxmc import ...`):
- **`ensemble.py`** — affine-invariant ensemble (Goodman & Weare 2010; the `emcee`
  algorithm). Gradient-free; stretch move; `mx.vmap` + `mx.compile`. `make_sampler`,
  `run_ensemble` (returns flat samples + accept_frac).
- **`hmc.py`** — Hamiltonian Monte Carlo. `mx.grad ∘ mx.vmap` (batched gradient over
  chains) + `mx.compile`. Identity mass matrix. `make_hmc`, `run_hmc`.
- **`preconditioned.py`** — mass-matrix HMC (M = Σ⁻¹). `make_phmc`, `run_phmc` (now takes
  `logp_single` like its siblings — the old `ess.logp_single` hardcoding was removed in the
  reorg). Returns the structured (T,N,D) chain.
- **`warmup.py`** — Stan-style warmup adaptation: dual-averaging step size + windowed
  *dense* mass-matrix estimation (`M⁻¹ = Cov`). Makes the preconditioning honest (M is
  estimated, not the supplied true Σ). `eps`/`M⁻¹`/`chol(M)ᵀ` passed as **array args** to
  the compiled step so per-iteration changes don't recompile; covariance + Cholesky run
  host-side in numpy fp64 (GPU only runs the fp32 leapfrog); NaN energies (funnel-neck
  divergences) are rejected, not propagated. `warmup`, `run_chain`, `DualAveraging`.
- **`nuts.py`** — NUTS (multinomial), vectorized over chains. Tree-doubling recursion in host
  Python; each leapfrog leaf `vmap`+`compile`'d; per-chain U-turn + masking (the MLX
  no-`while_loop` pattern). Validated exact on the Gaussian (cov 24.97 vs 25). `make_nuts`,
  `run_nuts`, and **`nuts_warmup`** — NUTS-specific warmup (dual-averaging on NUTS's own
  tree-averaged leaf-acceptance stat, H&G Alg 6), the principled alternative to borrowing
  `warmup`'s fixed-L eps. `eps` is a per-step arg to the step (so dual-averaging varies it
  without recompiling the leaf); `Minv` is closed over the compile, so a new M (per warmup
  window) rebuilds the step (few recompiles, not per step). **Gotcha:** the top-level sample
  update must gate on subtree validity `s'` (H&G Alg 3) — adopting proposals from
  internally-U-turned subtrees over-disperses, a bias that compounds with tree depth (caught
  vs known Σ; the mean alone looked fine). The leaf-accept stat, by contrast, counts *all*
  leaves (valid or not), per H&G Alg 6.
- **`diagnostics.py`** — effective sample size / integrated autocorrelation (emcee-style, FFT +
  Sokal window); `report` for ESS/sec. `integrated_time` skips zero-variance (stuck) walkers —
  one would otherwise NaN-poison the walker-averaged autocorrelation. Pure numpy, no MLX import.
- **`targets.py`** — example log-densities + known truths: correlated Gaussian (`gaussian_logp`,
  `GAUSSIAN_MU`/`GAUSSIAN_SIGMA`), banana, centered/non-centered funnel. (Split out of the old
  `ess.py`, which had mixed diagnostics + target + runners.)

**Examples `examples/`** (runnable demos / benchmark drivers; were the old top-level `__main__`s):
- **`gaussian_ess.py`** — ensemble vs identity-HMC vs preconditioned-HMC ESS/sec on the Gaussian.
- **`warmup_validation.py`** — warmup recovers Σ + matches oracle ESS/sec. `[L]` arg.
- **`hard_targets.py`** — banana + funnel benchmark harness (also defines the sampling-phase-timed
  `sample_hmc`/`sample_ensemble`, reused by `nuts_funnel` and `plot_hard_targets`). `[lscan|dscan]`.
- **`nuts_funnel.py`** — NUTS correctness on the Gaussian; `funnel` mode for the masking study.
- **`affine_invariance.py`** — empirically proves affine invariance (base + affine-transformed
  target, same RNG stream → identical acceptance).
- **`plot_hard_targets.py`** — renders `hard_targets_figure.png` (target shapes, funnel-neck
  contrast, v-marginal bias + non-centered fix, ESS/sec-vs-D).

**Tests `tests/`** (pytest; `conftest.py` pins device from `MLXMC_TEST_DEVICE`, `util.py` has
stat helpers): moment recovery for every sampler (`test_samplers_gaussian.py`, Standard
tolerances — mean <4·SE, std <5%, corr <0.03, NUTS cov <5% Frobenius), warmup Σ recovery
(<3% Frobenius), affine invariance (exact acceptance + <0.1 trajectory drift), and IAT on
white-noise/AR(1). 10 tests, ~7s GPU / ~16s CPU.

## Findings (2026-05-24, all on a corr-0.9, 25:1-variance 2-D Gaussian)
- **Affine-invariant ensemble:** tuning-free, gradient-free, handles ill-conditioning
  *for free* (proven: condition-1 vs condition-256 → bit-identical acceptance). But
  weaker per-step mixing (τ≈26) and degrades in high dimensions.
- **HMC:** needs gradients + hand-tuned `eps`/`L`, but mixes far better (τ≈2.2).
- **Preconditioned HMC (M=Σ⁻¹):** τ≈0.5 (negative-autocorr / antithetic regime),
  ~7× the ESS/sec of identity-mass HMC, ~11× the ensemble — with ~6× *fewer* leapfrog
  steps. The mass matrix is HMC's version of affine invariance, but you must supply it
  (here we used true Σ; real version estimates it during warmup = NUTS/Stan).
- **ESS is the honest efficiency metric; acceptance fraction is a misleading proxy**
  (0.99 acceptance looked "under-mixing" but τ said HMC was excellent).
- **Warmup adaptation (`warmup.py`) earns the metric:** dual-averaging eps + windowed
  dense covariance recovers true Σ to **<1% Frobenius error** (estimated `[24.8, 4.45;
  4.45, 0.99]` vs true `[25, 4.5; 4.5, 1]`), and where ESS is measurable (τ>1, e.g. L=1
  → τ≈2) the adapted sampler's ESS equals the oracle (true-Σ) sampler's to ~0.4%. The
  preconditioned-HMC headline above is now reproducible without supplying Σ.
- **ESS/sec is an unreliable *discriminator* once τ ≲ 1** (a NEW subtlety beyond "ESS >
  acceptance"). In the strong-antithetic regime the FFT+Sokal estimator's auto-window
  collapses to ~lag-1 and can return **negative τ / negative ESS** (L=3 here), making
  adapted-vs-oracle ratios meaningless (spurious 0.59×–0.65×). Compare at τ>1, or compare
  the recovered metric directly — not ESS/sec in the antithetic floor.

## Findings — harder targets (2026-05-24, `hard_targets.py`)
Refines the naive "hard targets → HMC's gradients win." ESS/sec below times the sampling
phase only (warmup/burn + compile excluded for both methods); HMC uses light `eps`-jitter
(see the resonance bullet).
- **Banana (Haario twist, B=0.05):** HMC wins by following the ridge with gradients, but a
  *global* M can't fit the curvature, so the win is modest — ~**1.7× ESS/sec at D=2,
  widening to ~4× by D=50**. Static-M HMC slightly under-explores the arm tips (x2 std
  ≈6.6–6.8 vs 7.14); the ensemble matches it at low D but degrades in *accuracy* at high D
  (x1 std 8.6 vs 10 at D=50).
- **Centered funnel (Neal):** the gradient-free ensemble **beats** static-M HMC on both
  accuracy and speed (**HMC 0.44×** ESS/sec). HMC is **biased** (v std ≈2.65 vs 3.0, can't
  reach the neck, stuck chains); an L-scan (6→48) shows the bias is **structural to the
  global metric, not tuning** — it never approaches truth. A constant M is wrong everywhere
  when the scale is position-dependent.
- **Non-centered reparam fixes the funnel:** sampling (v, x̃) with x = x̃·exp(v/2) removes
  the geometry (→ a product of Gaussians), so HMC becomes **unbiased** (v std 2.99), takes
  ~14× larger steps, and reclaims the lead (1.26× at D=2). Confirms the failure was the
  geometry, not HMC.
- **Fixed-L HMC trajectory resonance:** on near-Gaussian targets, `eps·L` near a multiple of
  2π returns the trajectory to its start → mixing collapses (τ jumps 0→64 between adjacent
  L). **Jittering `eps` per trajectory cures it**; without jitter, fixed-L HMC ESS/sec is
  unreliable. (NUTS's adaptive trajectory length is the principled fix — Phase 2.)
- **High-D — the ensemble degrades, HMC's lead widens** (the hypothesis's other half).
  Ensemble worst-dim τ grows ~linearly with D (NC funnel 27→188 over D=2→50; banana
  123→204) and its accuracy erodes; adapted HMC stays efficient to ~D=25 (then its own `eps`
  shrinks). NC-funnel HMC/ensemble ESS/sec: ~14× (D=2) → ~40× (D=10) → ~4× (D=50, both now
  degrading).
- **Takeaway:** gradient-free affine invariance is the robust *low-D default*; HMC's gradient
  + global-metric win needs mild curvature (banana) or geometry-aware coords
  (reparam/Riemannian) — but **scales far better with dimension**. A static M is a global
  *linear* fix: necessary, not sufficient.

## Findings — NUTS (2026-05-24, `nuts.py`)
Vectorized multinomial NUTS; reuses `warmup.py` for (eps, M) and adapts only trajectory length.
- **Correct** (the gate): exact covariance on the corr-0.9 Gaussian (24.97 vs 25). See the
  `nuts.py` Files note for the subtle subtree-validity-gating bug that inflated variance ~30%
  while the mean looked perfect — caught only vs known Σ.
- **NUTS does NOT cure the centered funnel:** it still uses a *global* M, so it's biased like
  fixed-L HMC (v std 2.69 vs 2.64; neck floor −6.4 vs −5.2). Adaptive trajectory length is no
  substitute for geometry-aware coordinates — the non-centered reparam is (NC funnel: NUTS
  unbiased, v std 3.00, reaches −15).
- **Masking overhead is the real MLX cost, and it's severe under heterogeneous trajectory
  lengths.** Centered funnel: a few chains hit `max_tree_depth=10` (wide mouth → global step
  too small) while mean depth is 3.0, and the batch pays the deepest chain → **151.9s vs 3.2s
  on the NC funnel** (mean/max depth 1.8/2). This masking cost is inherent to *vectorized*
  NUTS (JAX/NumPyro share it); MLX's specific limitation is having no escape hatch — no
  `while_loop` for a per-chain dynamic loop, so `max_tree_depth` is the only lever (caps cost,
  biases). The neck/mouth makes trajectory lengths heterogeneous, which is the worst case.
- **NUTS buys robustness, not peak speed:** auto-tunes L, dodges the fixed-L resonance, correct
  everywhere — but jittered fixed-L HMC beats its ESS/sec on easy targets (NC funnel 1.8M vs
  180k). NUTS = safe default; hand-tuned/jittered HMC = the speed play.
- **Gate 3 — D-scan on the NC funnel (2026-05-27, `examples/nuts_funnel.py dscan`):**
  NUTS holds accuracy across D=2→50 (dim0 v std **3.00 at every D**, true 3.0) and keeps
  τ ≤ 2.5; tree depth grows only gently (mean 1.8 → 3.0). At **D=50 adaptive-L NUTS beats
  fixed-L HMC** (τ 1.1 vs 21.1) — HMC's hand-fixed L=10 degrades at high D while NUTS
  adapts. The D=10 row caught a seed-dependent masking cameo (a few chains wandered to
  depth 5, mean 2.4 → wall ~4× neighbors), illustrating the batch-pays-deepest mechanism
  in miniature. The figure's new masking panel quantifies it at the funnel mouth: centered
  funnel mean depth 3.2 / max 10 / 31s wall vs NC funnel 1.8 / 2 / 1s — **31× wall-time
  ratio** for the same target up to a reparametrization.

## fp32 caveat — FUNDAMENTAL to MLX (and to scope decisions)
Apple Metal GPUs have **no float64** in hardware; MLX has fp64 only on the **CPU**
backend (forfeiting the GPU). So GPU work is fp32 (or fp16/bf16), permanently.
- **Fine for samplers**: Monte Carlo error (~1/√ESS) ≫ fp32 roundoff (~1e-6). The
  affine-test's 2e-2 trajectory drift was a *reproducibility* artifact (float32
  accumulation in stretched coords), **not** a sampling-accuracy problem — acceptance
  was bit-identical, posteriors correct. Watch precision-sensitive density evals (big
  log-likelihood sums, log-dets, long/stiff HMC trajectories); validate those vs fp64.
- **Real concern for fp64-demanding work** (science-grade PM N-body: dynamic range,
  conservation, backprop-through-time). Fine for a prototype, not production accuracy.
- The defensible Apple/MLX scientific niche: **fp32-tolerant, memory-bound,
  differentiable/stochastic** workloads — not fp64-precision HPC.

## MLX vs JAX notes
`grad`/`vmap`/`jvp`/`vjp`/`compile` transfer directly. Functional RNG keys
(`mx.random.split`). `mlx.core.fft` + some `linalg`; **no SHT** (no ducc0 equivalent).
`mlx.nn` modules are Equinox-like (params as attributes) but **mutable** (PyTorch-style
optimizer mutation), not immutable pytrees.
- **No traced control-flow primitives** (no `while_loop`/`scan`/`cond`/`fori_loop` — checked
  the `mlx.core` stub). MLX is eager + `compile` of *static* graphs, so JAX's `lax.while_loop`
  pattern doesn't transfer. Fixed-length unrolled loops (leapfrog, fixed-L HMC) compile fine;
  **data-dependent trajectory length (NUTS) is the hard case** — must run all chains to a
  fixed max tree depth and mask finished ones, or drop `vmap` and loop chains in Python.
- **To vary a parameter across `compile` calls without recompiling, pass it as an array
  argument, not a closed-over constant** (compile keys on shape/dtype/baked-constants, not
  array values). `warmup.py` does this for `eps`/`M⁻¹`; `n_leap` stays a Python int (structural).

## Next
**Done:** warmup mass-matrix adaptation; banana + centered/non-centered funnel + dim scan +
`eps`-jitter; **NUTS** validated exact on the Gaussian, funnel-tested, masking overhead
quantified; **NUTS D-scan (gate 3)** on NC funnel; **NUTS masking-overhead figure panel**;
**NUTS-specific warmup** (`nuts_warmup`) — dual-averaging on NUTS's leaf-accept stat; on the
Gaussian it tunes a distinctly larger eps than the borrowed fixed-L value (1.25 vs 1.10 at
L=8), recovers Σ identically, and mixes at least as well (τ 3.0 vs 3.8). Remaining:
- **Riemannian / in-place funnel:** a position-dependent metric to handle the *centered*
  funnel without reparametrizing (the non-centered trick doesn't generalize to all models).
- **Scale-up to tax the GPU** (the memory-bound MLX thesis) — current runs top out at D=50 /
  a few seconds and barely warm the M4 Max; push much higher D / many more chains. **The
  project's actual point**; correctness is in hand, scale is not.
- Possible upstream contribution: **MLX bindings to ducc0** (CPU C++ SHT/NUFFT — cleaner
  than the JAX version since unified memory avoids host↔device copies), best contributed
  *into ducc0* (it already has a JAX interface) rather than maintained standalone.
