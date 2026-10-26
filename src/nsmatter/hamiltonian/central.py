"""Central test potential: a sum of Gaussians, the same for every pair.

    v(r) = sum_k V_k exp(-kappa_k r^2)

In the periodic box a pair interacts with every image of its partner,
    v_box(r_ij) = sum over integer vectors n of v(|r_ij + L n|),
summed here over a cube of images large enough that the neglected tail
is below a tolerance. The same image machinery will serve the NV2 potentials.
"""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image, occupied_kvectors
from ..system import System

# Default test potential, as ((V_k in MeV, kappa_k in fm^-2), ...).
# Repulsive core + weak attraction (ranges borrowed from Minnesota).
# The attraction is weak on purpose: the volume integral
#   v~(0) = sum_k V_k (pi / kappa_k)^(3/2)
# must be positive, otherwise matter collapses (energy -> -infinity).
TOY = ((200.0, 1.487), (-30.0, 0.465))


def gaussian_sum(r2, terms):
    """v at squared distances r2 (any shape), in MeV."""
    return sum(v * jnp.exp(-k * r2) for v, k in terms)


def image_shifts(L, terms, tol=1e-10):
    """Image vectors L*n for all n in a cube [-m, m]^3, shape ((2m+1)^3, 3).

    m is chosen so every neglected image is farther than r_max, the
    distance beyond which |v| < tol (set by the longest-range Gaussian):
        V_max exp(-kappa_min r_max^2) = tol.
    Separations are taken as minimum images (components within +-L/2), so
    an image with some |n_i| >= m+1 is at least (m + 1/2) L away.
    The choice below is one step more conservative than strictly needed.
    """
    kappa_min = min(k for _, k in terms)
    vmax = sum(abs(v) for v, _ in terms)
    r_max = np.sqrt(np.log(vmax / tol) / kappa_min)
    m = int(np.ceil(r_max / L + 0.5))
    g = np.arange(-m, m + 1)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    return L * n.astype(float)


def make_central_potential(system: System, terms=TOY, tol=1e-10):
    """Potential term for make_local_energy: term(log_psi, params, r, s, t).

    A central potential is diagonal in position space and ignores spin and
    isospin, so log_psi, params and the labels are not used; the signature
    is shared with the NV2 terms, which will need them.
    """
    L = system.box_length
    # Setup: image vectors and the list of pairs i < j, computed once.
    shifts = jnp.asarray(image_shifts(L, terms, tol))
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)

    def potential(r):
        # Minimum-image separations of all pairs, shape (n_pairs, 3).
        d = minimum_image(r[i] - r[j], L)
        # Every image of every pair: shape (n_pairs, n_images, 3).
        dd = d[:, None, :] + shifts[None, :, :]
        r2 = jnp.sum(dd**2, axis=-1)
        # Sum over pairs and images.
        return jnp.sum(gaussian_sum(r2, terms))

    def term(log_psi, params, r, s, t):
        # Complex, like every local-energy contribution (real for this term).
        return potential(r).astype(jnp.complex128)

    return term


def hartree_fock_energy(system: System, terms=TOY):
    """Exact <V> in the plane-wave Slater determinant (MeV, total).

    For plane waves normalized in the box volume V:
        <V> = 1/(2V) [ N^2 v~(0) - sum_species sum_{a,b} v~(k_a - k_b) ]
    first term: direct (Hartree), every particle with every other;
    second term: exchange (Fock), only between identical particles, i.e.
    within a species. The a = b exchange terms cancel the self-interaction
    contained in N^2. v~(q) is the Fourier transform of v over all space:
        v~(q) = sum_k V_k (pi / kappa_k)^(3/2) exp(-q^2 / (4 kappa_k)).
    Summing over all images in real space is the same as using v~ at the
    box momenta, so this is exact for the periodic potential above.
    """
    L = system.box_length
    V = L**3
    N = system.n_particles

    def vq(q2):
        return sum(v * (np.pi / k) ** 1.5 * np.exp(-q2 / (4 * k)) for v, k in terms)

    direct = N**2 * vq(0.0)
    exchange = 0.0
    for count in system.counts:
        if count == 0:
            continue
        k = occupied_kvectors(count, L)
        # All momentum differences k_a - k_b within the species.
        dk = k[:, None, :] - k[None, :, :]
        exchange += vq((dk**2).sum(-1)).sum()
    return (direct - exchange) / (2 * V)
