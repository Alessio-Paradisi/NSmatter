"""nsmatter: variational Monte Carlo for nuclear and neutron-star matter."""

import jax

# Double precision everywhere: determinants, Pfaffians and the SR matrix
# lose accuracy quickly in float32.
jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
