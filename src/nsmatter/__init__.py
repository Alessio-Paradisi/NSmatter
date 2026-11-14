"""nsmatter: variational Monte Carlo for nuclear and neutron-star matter."""

import os

# CPU by default: the code is built for many CPU devices in double precision, and
# on the lab GPUs (e.g. NVIDIA T1000) complex128 matrix products fail or are slow.
# JAX_PLATFORMS=cuda (or another value) set in the environment still wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# Double precision everywhere: determinants, Pfaffians and the SR matrix
# lose accuracy quickly in float32.
jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
