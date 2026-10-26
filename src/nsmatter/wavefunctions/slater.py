"""Slater determinant of plane waves, one determinant per species.

With fixed labels, the Slater determinant of spin-isospin orbitals
factorizes into a product of determinants, one per populated species:

    Psi(R) = prod over species of det[ exp(i k_a . r_j) ]

with a running over the occupied momenta and j over the particles of that
species. It is the exact ground state of the free Fermi gas in the box,
with energy E = hbar^2/2m * sum over occupied k of |k|^2.
"""

import jax.numpy as jnp
import numpy as np

from ..constants import HBAR2_2M
from ..geometry import occupied_kvectors
from ..system import System


def make_slater(system: System):
    """Build the plane-wave Slater determinant for `system`.

    Returns (params, log_psi). Everything that does not change during the
    simulation (which particles belong to which species, their momenta)
    is computed here once and captured by log_psi.
    """
    L = system.box_length

    # For each populated species keep (index slice, occupied k-vectors).
    # Empty species are skipped: they contribute a factor 1.
    blocks = []
    for sl, count in zip(system.species_slices, system.counts):
        if count > 0:
            blocks.append((sl, jnp.asarray(occupied_kvectors(count, L))))

    # No variational parameters yet; kept so all ansatze share one interface.
    params = {}

    def log_psi(params, r, s, t):
        # Accumulate ln Psi as a sum of ln det over species (product -> sum).
        total = jnp.zeros((), dtype=jnp.complex128)
        for sl, k in blocks:
            # Phase matrix k_a . r_j, shape (n_orbitals, n_particles):
            # rows = orbitals, columns = particles of this species.
            phases = k @ r[sl].T
            mat = jnp.exp(1j * phases)
            # slogdet returns (sign, ln|det|) without overflow or underflow.
            # For a complex matrix, `sign` is a complex number of modulus 1
            # (or 0 at a node); ln(sign) = i * phase.
            sign, logabs = jnp.linalg.slogdet(mat)
            total = total + logabs + jnp.log(sign)
        return total

    return params, log_psi


# -----------------------------------------------------------------------------
# Reference energies (setup / analysis only: NumPy)
# -----------------------------------------------------------------------------
def free_gas_energy(system: System) -> float:
    """Exact energy of the box Slater determinant (MeV), total, not per particle.

    E = hbar^2/2m * sum over all species and occupied k of |k|^2.
    """
    L = system.box_length
    e = 0.0
    for count in system.counts:
        k = occupied_kvectors(count, L)
        e += HBAR2_2M * float((k**2).sum())
    return e


def free_gas_energy_infinite(system: System) -> float:
    """Free Fermi gas in the thermodynamic limit at the same species densities (MeV, total).

    Each species is a single-spin-state gas of density rho_s = N_s / V:
    kF_s = (6 pi^2 rho_s)^(1/3) and E_s / N_s = 3/5 * hbar^2 kF_s^2 / 2m.
    (The 6 pi^2 instead of 3 pi^2 is because each species has degeneracy 1.)
    """
    V = system.box_length**3
    e = 0.0
    for count in system.counts:
        if count == 0:
            continue
        kf = (6.0 * np.pi**2 * count / V) ** (1.0 / 3.0)
        e += count * 0.6 * HBAR2_2M * kf**2
    return e
