"""Total local energy E_L(X) = (H Psi)(X) / Psi(X) for one walker.

Potential terms are functions with the signature
    term(log_psi, params, r, s, t) -> complex
They receive log_psi itself because spin-isospin operators (sigma.sigma,
tensor, ...) need Psi at modified configurations, not only at X.
The free-gas stage has no potential terms.
"""

from .hamiltonian.kinetic import kinetic_local_energy


def make_local_energy(log_psi, potential_terms=()):
    """Build local_energy(params, r, s, t) = kinetic + sum of potential terms."""

    def local_energy(params, r, s, t):
        e = kinetic_local_energy(log_psi, params, r, s, t)
        for term in potential_terms:
            e = e + term(log_psi, params, r, s, t)
        return e

    return local_energy
