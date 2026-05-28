"""Shared pytest config. Pins the MLX compute device from MLXMC_TEST_DEVICE so the CI
matrix can run the suite on both CPU and the Metal GPU.

  MLXMC_TEST_DEVICE = cpu | gpu   (unset -> MLX's native default: the GPU on Apple silicon)

mlxmc is fp32-on-GPU; the CPU backend also runs fp32 here (we don't request fp64), so the
two device legs exercise the same precision and the Standard tolerances apply to both.
"""
import os

import mlx.core as mx

_dev = os.environ.get("MLXMC_TEST_DEVICE", "").strip().lower()
if _dev == "cpu":
    mx.set_default_device(mx.cpu)
elif _dev in ("gpu", "metal"):
    mx.set_default_device(mx.gpu)


def pytest_report_header(config):
    return f"mlxmc: MLX default device = {mx.default_device()}"
