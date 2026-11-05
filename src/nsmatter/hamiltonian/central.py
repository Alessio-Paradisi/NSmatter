"""Spin-isospin independent central potential: a sum of Gaussians, periodic in the box."""

import jax.numpy as jnp
import numpy as np

from ..geometry import minimum_image, occupied_kvectors
from ..system import System

TOY = ((200.0, 1.487), (-30.0, 0.465))


def gaussian_sum(r2, terms):
    return sum(v * jnp.exp(-k * r2) for v, k in terms)


def image_shifts(L, terms, tol=1e-10):
    kappa_min = min(k for _, k in terms)
    vmax = sum(abs(v) for v, _ in terms)
    r_max = np.sqrt(np.log(vmax / tol) / kappa_min)
    m = int(np.ceil(r_max / L + 0.5))
    g = np.arange(-m, m + 1)
    n = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    return L * n.astype(float)


def make_central_potential(system: System, terms=TOY, tol=1e-10):
    L = system.box_length
    shifts = jnp.asarray(image_shifts(L, terms, tol))
    i, j = np.triu_indices(system.n_particles, k=1)
    i, j = jnp.asarray(i), jnp.asarray(j)

    def potential(r):
        d = minimum_image(r[i] - r[j], L)
        dd = d[:, None, :] + shifts[None, :, :]
        r2 = jnp.sum(dd**2, axis=-1)
        return jnp.sum(gaussian_sum(r2, terms))

    def term(log_psi, params, r, s, t):
        return potential(r).astype(jnp.complex128)

    return term


def hartree_fock_energy(system: System, terms=TOY):
    L = system.box_length
    V = L**3
    N = system.n_particles

    def vq(q2):
        return sum(v * (np.pi / k) ** 1.5 * np.exp(-q2 / (4 * k)) for v, k in terms)

    direct = N**2 * vq(0.0)
    exchange = 0.0
    for count in system.counts:
        k = occupied_kvectors(count, L)
        if count == 0:
            continue
        dk = k[:, None, :] - k[None, :, :]
        exchange += vq((dk**2).sum(-1)).sum()
    return (direct - exchange) / (2 * V)
