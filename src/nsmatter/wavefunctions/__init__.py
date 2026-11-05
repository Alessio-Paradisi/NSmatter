"""Trial wavefunctions.

Every ansatz is a pair (params, log_psi) where

    log_psi(params, r, s, t) -> complex scalar

with r of shape (N, 3), s and t of shape (N,) (spins and isospins, +1/-1).
log_psi returns ln Psi = ln|Psi| + i*phase, so a sign or phase is carried in
the imaginary part. Batching over walkers is done outside with jax.vmap.
"""

from .slater import make_slater  # noqa: F401
