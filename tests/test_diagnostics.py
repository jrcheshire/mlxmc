"""Integrated autocorrelation time on inputs with a known answer:
  - white noise  -> tau ~ 1   (Standard tolerance: tau in [0.8, 1.5])
  - AR(1), phi   -> tau = (1+phi)/(1-phi)   (within 20%)
"""
import numpy as np

from mlxmc.diagnostics import integrated_time


def test_iat_white_noise():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4000, 64))            # (T, N): independent draws -> tau ~ 1
    tau = integrated_time(y)
    assert 0.8 < tau < 1.5, f"white-noise tau {tau:.3f} not ~1"


def test_iat_ar1():
    # x_t = phi x_{t-1} + sqrt(1-phi^2) eps  -> stationary var 1, tau = (1+phi)/(1-phi).
    rng = np.random.default_rng(1)
    phi, T, N = 0.8, 8000, 64
    x = np.zeros((T, N))
    noise = rng.standard_normal((T, N)) * np.sqrt(1.0 - phi**2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + noise[t]
    tau = integrated_time(x)
    tau_true = (1.0 + phi) / (1.0 - phi)           # = 9.0
    assert abs(tau - tau_true) / tau_true < 0.20, f"AR(1) tau {tau:.2f} vs true {tau_true:.2f}"


def test_iat_skips_stuck_walker():
    # A constant (zero-variance) walker must be skipped, not NaN-poison the average.
    rng = np.random.default_rng(2)
    y = rng.standard_normal((2000, 16))
    y[:, 0] = 3.0                                   # stuck walker
    tau = integrated_time(y)
    assert np.isfinite(tau) and 0.8 < tau < 1.5, f"stuck-walker tau {tau}"
