"""Slater determinant of spin-isospin plane-wave orbitals.

The occupied orbitals are plane waves exp(i k . r) with fixed spin and
isospin, filled in closed shells for each of the four species.
The determinant is built over all N particles, so that Psi depends on the
spin and isospin coordinates: the spin-isospin operators of the potential
need Psi at configurations with labels swapped or replaced.

Spins can be discrete labels (+1/-1) or spinors xi = (xi_up, xi_down)
(see spin.py); Psi is linear in each spinor.

It is the exact ground state of the free Fermi gas in the box, with energy
E = hbar^2/2m * sum over occupied k of |k|^2.
"""

import jax.numpy as jnp
import numpy as np

from ..constants import HBAR2_2M
from ..geometry import occupied_kvectors
from ..spin import as_spinors
from ..system import System


def make_slater(system: System):
    """Single N x N determinant of spin-isospin plane-wave orbitals.

    Orbital a: phi_a(r, xi, t) = exp(i k_a . r) * (component of xi along the
    orbital's spin) * [t == t_a].
    Matrix element M[a, j] = phi_a(r_j, xi_j, t_j).

    - labels in species order (normal case): M is block diagonal and
      det M = product of the four per-species determinants;
    - swapped labels or general spinors: det M is the amplitude of that
      spin-isospin configuration;
    - labels with the wrong composition: an empty row, det M = 0.
    Returns (params, log_psi).
    """
    L = system.box_length

    # Orbitals of every populated species, in species order:
    # momentum, spin label and isospin label of each orbital.
    ks, so, to = [], [], []
    for count, sz, tz in zip(system.counts, (1, -1, 1, -1), (-1, -1, 1, 1)):
        if count > 0:
            ks.append(occupied_kvectors(count, L))
            so += [sz] * count
            to += [tz] * count
    k = jnp.asarray(np.concatenate(ks))  # (N, 3) momenta
    # Spin part of each orbital as a basis spinor: index 0 for up, 1 for down.
    spin_index = jnp.asarray((1 - np.array(so)) // 2)
    t_orb = jnp.asarray(to)  # (N,) isospin of each orbital

    # No variational parameters; kept so all ansatze share one interface.
    params = {}

    def log_psi(params, r, s, t):
        xi = as_spinors(s)  # (N, 2): labels or spinors
        # Plane-wave part for every (orbital, particle) pair, shape (N, N).
        plane = jnp.exp(1j * (k @ r.T))
        # Spin factor of orbital a for particle j: the component of xi_j
        # along the orbital's spin (1 or 0 for discrete labels).
        spin = xi[:, spin_index].T  # (orbitals, particles)
        # Isospin: particle j must carry the isospin of orbital a.
        iso = t_orb[:, None] == t[None, :]
        mat = jnp.where(iso, plane * spin, 0.0)
        # ln det without overflow; ln(sign) carries the phase.
        # At a node (Psi = 0) logabs = -inf: those configurations have zero weight.
        sign, logabs = jnp.linalg.slogdet(mat)
        return logabs + jnp.log(sign)

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
