"""Pair observables against exact identities.

Sum over pairs: sum_{i<j} sigma_i.sigma_j = 2 S(S+1) - 3N/2 and likewise for tau.
For an eigenstate of S^2 (T^2) this holds for the LOCAL estimator at every
configuration. Closed-shell Slater determinants have S = 0 (and T = 0 for
symmetric matter); central and spin-isospin correlations keep them (they commute
with S^2 and T^2), tensor correlations would not, so those are left at zero.
The Jastrow has separate channels for like and unlike pairs (labels tau_z):
that breaks isospin symmetry, so both channels get the same parameters here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nsmatter  # noqa: F401
from nsmatter.observables import make_pair_observables, normalize
from nsmatter.sampler import random_spinors, uniform_walkers
from nsmatter.system import System
from nsmatter.wavefunctions.correlations import make_correlated


@pytest.mark.parametrize("pops,S_T", [((7, 7, 0, 0), (0, 7)), ((1, 1, 1, 1), (0, 0))], ids=["pnm", "alpha-like"])
def test_total_spin_isospin_identities(pops, S_T):
    system = System(*pops, 0.1)
    N = system.n_particles
    params, log_psi = make_correlated(system, n_terms=3)
    key = jax.random.PRNGKey(0)
    row = 0.3 * jax.random.normal(key, params["jastrow"].shape[1:])
    params["jastrow"] = jnp.stack([row, row])  # same for like and unlike pairs: isospin invariant
    params["corr"] = params["corr"].at[:3].set(0.3 * jax.random.normal(key, (3, params["corr"].shape[1])))  # s, t, st
    obs = make_pair_observables(system, n_bins=10)
    t = jnp.asarray(system.isospins)
    S, T = S_T
    for w in range(3):
        r = uniform_walkers(jax.random.PRNGKey(10 + w), 1, N, system.box_length)[0]
        xi = random_spinors(jax.random.PRNGKey(20 + w), (N,))
        o = obs(log_psi, params, r, xi, t)
        assert np.isclose(o["sigma_total"], 2 * S * (S + 1) - 1.5 * N, atol=1e-9)
        assert np.isclose(o["tau_total"], 2 * T * (T + 1) - 1.5 * N, atol=1e-9)


def test_uniform_pairs_give_flat_g():
    """Uncorrelated uniform points: g -> (N - 1) / N in every bin (statistically)."""
    system = System(7, 7, 0, 0, 0.1)
    N = system.n_particles
    obs = make_pair_observables(system, n_bins=8)
    r = uniform_walkers(jax.random.PRNGKey(1), 4000, N, system.box_length)
    counts = jax.vmap(lambda x: obs(lambda *a: 0.0 * a[1].sum(), None, x, jnp.ones((N, 2)) / np.sqrt(2),
                                    jnp.asarray(system.isospins))["counts"])(r)
    g = normalize({"counts": counts.sum(0), "sigma": counts.sum(0), "tau": counts.sum(0)}, 4000, system, obs.dr)["g"]
    assert np.allclose(g[0][2:], (N - 1) / N, rtol=0.05)  # nn; first bins have few counts
