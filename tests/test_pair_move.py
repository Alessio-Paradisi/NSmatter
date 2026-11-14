"""Pair moves (rank-2 updates of the correlated wavefunction) against full evaluations.

- move(...) = ln Psi(X') - ln Psi(X) for moved positions, new spinors and swapped isospins;
- the static and momentum potential terms give the same energy with the fast path
  (log_psi.pair_move) and with full evaluations of Psi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401
from nsmatter.hamiltonian.momentum import CHANNELS as MOMENTUM_CHANNELS
from nsmatter.hamiltonian.momentum import make_momentum_potential
from nsmatter.hamiltonian.pair_operators import CHANNELS as STATIC_CHANNELS
from nsmatter.hamiltonian.pair_operators import make_static_potential
from nsmatter.sampler import random_spinors, uniform_walkers
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated

SYSTEMS = [System(7, 1, 1, 7, 0.1), System(7, 7, 0, 0, 0.16)]


def setup(system):
    params, log_psi = make_correlated(system, n_terms=3)
    params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(jax.random.PRNGKey(0), p.shape), params)
    N = system.n_particles
    r = uniform_walkers(jax.random.PRNGKey(1), 1, N, system.box_length)[0]
    xi = random_spinors(jax.random.PRNGKey(2), (N,))
    return params, log_psi, r, xi, jnp.asarray(system.isospins)


def test_move_matches_full_evaluation():
    params, log_psi, r, xi, t = setup(SYSTEMS[0])
    prepare, move = log_psi.pair_move
    cache = prepare(params, r, xi, t)
    lp0 = log_psi(params, r, xi, t)
    u = jnp.array([0.3, -0.7, 0.2])
    for k, (p, q) in enumerate([(0, 1), (0, 8), (7, 9), (3, 15), (8, 9)]):
        xp, xq = random_spinors(jax.random.PRNGKey(10 + k), (2,))
        for tp, tq in ((t[p], t[q]), (t[q], t[p])):  # isospins unchanged / swapped
            fast = move(params, cache, p, q, u, xp, xq, tp, tq)
            full = log_psi(params, r.at[p].add(u).at[q].add(-u), xi.at[p].set(xp).at[q].set(xq),
                           t.at[p].set(tp).at[q].set(tq)) - lp0
            assert np.isclose(np.exp(complex(fast)), np.exp(complex(full)), rtol=1e-11)


@pytest.mark.parametrize("system", SYSTEMS, ids=["np-asymmetric", "pnm"])
def test_fast_and_full_potential_terms_agree(system):
    params, log_psi, r, xi, t = setup(system)
    full_psi = lambda *a: log_psi(*a)  # noqa: E731  (no pair_move attribute -> full evaluations)
    static = make_static_potential(system, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(STATIC_CHANNELS)})
    mom = make_momentum_potential(system, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(MOMENTUM_CHANNELS)})
    for term in (static, mom):
        fast = complex(term(log_psi, params, r, xi, t))
        full = complex(term(full_psi, params, r, xi, t))
        assert np.isclose(fast, full, rtol=1e-11)


def test_energy_components_add_up():
    """make_local_energy_components: kinetic + every channel = make_local_energy."""
    from nsmatter.local_energy import make_local_energy, make_local_energy_components

    system = SYSTEMS[0]
    params, log_psi, r, xi, t = setup(system)
    static = make_static_potential(system, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(STATIC_CHANNELS)})
    mom = make_momentum_potential(system, {ch: ((5.0 + k, 0.9),) for k, ch in enumerate(MOMENTUM_CHANNELS)})
    comps = make_local_energy_components(log_psi, (static, mom))(params, r, xi, t)
    total = make_local_energy(log_psi, (static, mom))(params, r, xi, t)
    assert set(comps) == {"kinetic"} | {f"v_{c}" for c in (*STATIC_CHANNELS, *MOMENTUM_CHANNELS)}
    assert np.isclose(sum(comps.values()), total, rtol=1e-12)


def test_pair_chunks_do_not_change_results():
    """map_pairs in chunks (with and without padding) = all pairs at once, also with spin-orbit correlations."""
    from nsmatter.hamiltonian import pair_operators

    for system in SYSTEMS:
        params, log_psi, r, xi, t = setup(system)
        static = make_static_potential(system, {ch: ((10.0 + k, 0.8 + 0.1 * k),) for k, ch in enumerate(STATIC_CHANNELS)})
        mom = make_momentum_potential(system, {ch: ((5.0 + k, 0.9),) for k, ch in enumerate(MOMENTUM_CHANNELS)})
        try:
            ref = [complex(term(log_psi, params, r, xi, t)) for term in (static, mom)]
            for chunk in (7, 10):  # 7 divides the 91 pairs of 14 particles, 10 does not (padding)
                pair_operators.set_pair_chunk(chunk)
                got = [complex(term(log_psi, params, r, xi, t)) for term in (static, mom)]
                assert np.allclose(got, ref, rtol=1e-12), (chunk, got, ref)
        finally:
            pair_operators.set_pair_chunk(None)
