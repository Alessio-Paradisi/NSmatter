"""Slater determinant of spin-isospin plane-wave orbitals.

The occupied orbitals are plane waves exp(i k . r) with fixed spin and
isospin labels, filled in closed shells for each of the four species.
The determinant is built over all N particles, so that Psi depends on the
labels s, t: the spin-isospin operators of the potential need Psi at
configurations with labels swapped between particles.

It is the exact ground state of the free Fermi gas in the box, with energy
E = hbar^2/2m * sum over occupied k of |k|^2.
"""

import jax.numpy as jnp
import numpy as np

from ..constants import HBAR2_2M
from ..geometry import occupied_kvectors
from ..system import System


def make_slater(system: System):
    """Single N x N determinant of spin-isospin plane-wave orbitals.

    Orbital a: phi_a(r, s, t) = exp(i k_a . r) * [s == s_a] * [t == t_a].
    Matrix element M[a, j] = phi_a(r_j, s_j, t_j): nonzero only where the
    labels of particle j match those of orbital a.

    - labels in species order (normal sampling): M is block diagonal and
      det M = product of the four per-species determinants;
    - swapped labels (inside the local energy): the blocks rearrange and
      det M is the amplitude of that spin-isospin configuration;
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
    s_orb = jnp.asarray(so)  # (N,) spin of each orbital
    t_orb = jnp.asarray(to)  # (N,) isospin of each orbital

    # No variational parameters; kept so all ansatze share one interface.
    params = {}

    def log_psi(params, r, s, t):
        # Plane-wave part for every (orbital, particle) pair, shape (N, N).
        plane = jnp.exp(1j * (k @ r.T))
        # True where particle j carries the labels of orbital a.
        match = (s_orb[:, None] == s[None, :]) & (t_orb[:, None] == t[None, :])
        mat = jnp.where(match, plane, 0.0)
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
