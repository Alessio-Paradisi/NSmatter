"""nsmatter: variational Monte Carlo for nuclear and neutron-star matter."""

import jax

# JAX uses 32-bit floats by default. Switch to 64-bit for the whole package:
# determinants and the optimizer matrix lose accuracy quickly in float32.
# This line runs once, the first time anything imports nsmatter.
jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
