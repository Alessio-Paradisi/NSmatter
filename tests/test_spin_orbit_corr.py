"""Spin-orbit correlations u(r) L.S [tau.tau] in the trial wavefunction.

Reference (independent of the code's derivative contractions): the correlation factor
    C = 1 + sum_{i<j} sum_targets <current|S_a|target> (L_a Phi)(target) / Phi
built from the definition, with
    (L_a Phi)(target) = -i (r x d_r)_a Phi(target),  d_r = (1/2) d/du of
    Phi(r_i + u, r_j - u) by automatic differentiation of a plain determinant, and the
    isospin factors of tau.tau (unchanged targets: +-1, swapped: 2 for unlike pairs).
Then: pair moves and the fast potential terms with spin-orbit correlations against
full evaluations.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401
from nsmatter.geometry import minimum_image, occupied_kvectors
from nsmatter.hamiltonian.momentum import CHANNELS as MOMENTUM_CHANNELS
from nsmatter.hamiltonian.momentum import make_momentum_potential
from nsmatter.hamiltonian.pair_operators import CHANNELS as STATIC_CHANNELS
from nsmatter.hamiltonian.pair_operators import make_static_potential
from nsmatter.sampler import random_spinors, uniform_walkers
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import S_PAIR, make_correlated
from nsmatter.wavefunctions.jastrow import jastrow_basis

SYSTEM = System(7, 1, 1, 7, 0.1)  # n and p, both spins: tau.tau swaps matter
N = SYSTEM.n_particles


def slater_det(r, xi, t):
    """Phi = det M, M[a, q] = exp(i k_a.r_q) xi_q[spin_a] [t_q = t_a] (same orbitals as the code)."""
    ks, so, to = [], [], []
    for count, sz, tz in zip(SYSTEM.counts, (0, 1, 0, 1), (-1, -1, 1, 1)):
        if count:
            ks.append(occupied_kvectors(count, SYSTEM.box_length))
            so += [sz] * count
            to += [tz] * count
    k, so, to = jnp.asarray(np.concatenate(ks)), jnp.asarray(so), jnp.asarray(to)
    M = jnp.exp(1j * k @ r.T) * xi[:, so].T * (to[:, None] == t[None, :])
    return jnp.linalg.det(M)


def setup(seed=0):
    params, log_psi = make_correlated(SYSTEM, n_terms=3, spin_orbit=True)
    key = jax.random.PRNGKey(seed)
    params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), params)
    r = uniform_walkers(jax.random.PRNGKey(seed + 1), 1, N, SYSTEM.box_length)[0]
    xi = random_spinors(jax.random.PRNGKey(seed + 2), (N,))
    return params, log_psi, r, xi, jnp.asarray(SYSTEM.isospins)


def test_spin_orbit_correlation_factor_from_definition():
    params, log_psi, r, xi, t = setup()
    only_so = {"jastrow": jnp.zeros_like(params["jastrow"]),
               "corr": params["corr"].at[:5].set(0.0)}  # only the ls, lst rows
    zero = jax.tree_util.tree_map(jnp.zeros_like, only_so)
    C_code = np.exp(complex(log_psi(only_so, r, xi, t) - log_psi(zero, r, xi, t)))

    L, rc = SYSTEM.box_length, 0.5 * SYSTEM.box_length
    phi0 = slater_det(r, xi, t)
    basis = jnp.eye(2, dtype=complex)
    total = 0.0
    for p, q in itertools.combinations(range(N), 2):
        d = minimum_image(r[p] - r[q], L)
        u = jastrow_basis(jnp.sum(d**2), rc, 3) @ only_so["corr"].T  # u[5] = ls, u[6] = lst
        bra = jnp.kron(xi[p], xi[q])
        like = t[p] == t[q]
        for b1, b2 in itertools.product(range(2), repeat=2):
            ket = jnp.kron(basis[b1], basis[b2])
            s_vec = jnp.einsum("c,acd,d->a", bra, S_PAIR, ket)
            spins = xi.at[p].set(basis[b1]).at[q].set(basis[b2])
            for swap in (False, True):
                if swap and like:
                    continue
                tt = t.at[p].set(t[q]).at[q].set(t[p]) if swap else t
                iso = (2.0 if swap else (1.0 if like else -1.0))
                f = lambda w: slater_det(r.at[p].add(w).at[q].add(-w), spins, tt)  # noqa: E731,B023
                grad = jax.jacfwd(f)(jnp.zeros(3)) / 2.0  # d_r Phi(target)
                L_phi = -1j * jnp.cross(d, grad)
                coef = (0.0 if swap else u[5]) + u[6] * iso
                total = total + coef * jnp.dot(s_vec, L_phi)
    C_ref = 1.0 + complex(total / phi0)
    assert abs(C_code - C_ref) < 1e-10 * max(1.0, abs(C_ref)), (C_code, C_ref)
    assert abs(C_ref - 1.0) > 1e-3  # not trivially 1


def test_pair_moves_with_spin_orbit():
    params, log_psi, r, xi, t = setup(1)
    prepare, move = log_psi.pair_move
    cache = prepare(params, r, xi, t)
    lp0 = log_psi(params, r, xi, t)
    u = jnp.array([0.3, -0.7, 0.2])
    for k, (p, q) in enumerate([(0, 1), (0, 8), (7, 9), (3, 15)]):
        xp, xq = random_spinors(jax.random.PRNGKey(30 + k), (2,))
        for tp, tq in ((t[p], t[q]), (t[q], t[p])):
            fast = move(params, cache, p, q, u, xp, xq, tp, tq)
            full = log_psi(params, r.at[p].add(u).at[q].add(-u), xi.at[p].set(xp).at[q].set(xq),
                           t.at[p].set(tp).at[q].set(tq)) - lp0
            assert np.isclose(np.exp(complex(fast)), np.exp(complex(full)), rtol=1e-10)


def test_potential_terms_fast_and_full_with_spin_orbit():
    params, log_psi, r, xi, t = setup(2)
    full_psi = lambda *a: log_psi(*a)  # noqa: E731
    static = make_static_potential(SYSTEM, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(STATIC_CHANNELS)})
    mom = make_momentum_potential(SYSTEM, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(MOMENTUM_CHANNELS)})
    for term in (static, mom):
        assert np.isclose(complex(term(log_psi, params, r, xi, t)), complex(term(full_psi, params, r, xi, t)),
                          rtol=1e-10)
