"""Trial wavefunctions.

Every ansatz is a pair (params, log_psi), where

    log_psi(params, r, s, t) -> complex scalar

- params: variational parameters (a dict or nested structure of arrays)
- r: positions, shape (N, 3)
- s, t: spin and isospin labels (+1/-1), shape (N,)

log_psi returns ln Psi = ln|Psi| + i*phase, so the sign (or phase) of Psi
is carried by the imaginary part. One call handles one configuration;
batching over walkers is done outside with jax.vmap.
"""

# Re-export so that `from nsmatter.wavefunctions import make_slater` works.
from .slater import make_slater  # noqa: F401
