"""Optimize the operator-correlated wavefunction with sampled spinors.

Psi = exp(J) [1 + sum U_ij] Phi, against a v6 test potential with a tensor
force. With --correlations jastrow only the central Jastrow is optimized
(the U_ij stay zero), which shows what the operator correlations add.

Usage (from ~/projects/nsmatter):
    uv run python scripts/optimize_correlated.py --correlations jastrow
    uv run python scripts/optimize_correlated.py --correlations all
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.hamiltonian.central import TOY
from nsmatter.hamiltonian.pair_operators import make_v6_potential
from nsmatter.local_energy import make_local_energy
from nsmatter.optimize import apply_update, log_derivatives_spin, sr_direction
from nsmatter.sampler import make_log_prob_spin, metropolis_spin, random_spinors, thermalize_spin, uniform_walkers
from nsmatter.stats import blocking_error
from nsmatter.system import System
from jax.flatten_util import ravel_pytree

from nsmatter.wavefunctions.correlations import make_correlated

# Test v6 interaction: central repulsive core + weak attraction, and a
# tensor force times tau.tau (as in one-pion exchange), v_tnt(r) = V r^2 exp(-kappa r^2).
V6_TEST = {"c": TOY, "tnt": ((10.0, 0.6),)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-up", type=int, default=1)
    p.add_argument("--n-down", type=int, default=1)
    p.add_argument("--p-up", type=int, default=1)
    p.add_argument("--p-down", type=int, default=1)
    p.add_argument("--density", type=float, default=0.04, help="fm^-3")
    p.add_argument("--correlations", choices=("jastrow", "all"), default="all")
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--walkers", type=int, default=512)
    p.add_argument("--steps-between", type=int, default=10)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.003, help="SR step (MeV^-1); smaller for larger N")
    p.add_argument("--diag-shift", type=float, default=0.05)
    p.add_argument("--measurements", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    system = System(a.n_up, a.n_down, a.p_up, a.p_down, a.density)
    N, L = system.n_particles, system.box_length
    print(system.describe())
    print("device:", jax.devices()[0], "| correlations:", a.correlations)

    params, log_psi = make_correlated(system, n_terms=a.n_terms)
    t = jnp.asarray(system.isospins)
    potential = make_v6_potential(system, V6_TEST)

    # Walkers carry positions AND spinors: map over both.
    e_loc = jax.jit(jax.vmap(make_local_energy(log_psi, (potential,)), in_axes=(None, 0, 0, None)))
    log_derivs = jax.jit(lambda p, r, xi: log_derivatives_spin(log_psi, p, r, xi, t))
    log_prob = make_log_prob_spin(log_psi, t)

    # With --correlations jastrow, the operator-correlation parameters are frozen
    # at zero. The mask is built with the same flattening as the parameters
    # (ravel_pytree orders dict keys alphabetically: "corr" comes first).
    keep = {"jastrow": jnp.ones_like(params["jastrow"]), "corr": jnp.zeros_like(params["corr"])}
    keep_flat = ravel_pytree(keep)[0] if a.correlations == "jastrow" else 1.0

    def mask(delta):
        return delta * keep_flat

    key = jax.random.PRNGKey(a.seed)
    k1, k2, key = jax.random.split(key, 3)
    r = uniform_walkers(k1, a.walkers, N, L)
    xi = random_spinors(k2, (a.walkers, N))
    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, 0.1 * L / np.sqrt(N), 0.5, L)

    print("\n step   E/N (MeV)    +-     var(E_L)/N^2  acc_r  acc_s")
    t0 = time.time()
    for it in range(a.iterations):
        key, sub = jax.random.split(key)
        r, xi, acc_r, acc_s = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, a.steps_between)
        e = e_loc(params, r, xi, t).real
        O = log_derivs(params, r, xi)
        delta, _, _ = sr_direction(O, e, diag_shift=a.diag_shift)
        params = apply_update(params, mask(delta), a.lr)
        mean, err = float(e.mean()) / N, float(e.std()) / np.sqrt(a.walkers) / N
        print(f"{it:5d}  {mean:10.4f}  {err:7.4f}  {float(e.var()) / N**2:12.4f}  {float(acc_r):5.3f}  {float(acc_s):5.3f}")
        if not np.isfinite(mean):
            raise RuntimeError("energy is not finite: reduce --lr or increase --diag-shift")
    print(f"optimization: {time.time() - t0:.1f} s")

    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, step, spin_step, L, n_rounds=5)
    energies = []
    for _ in range(a.measurements):
        key, sub = jax.random.split(key)
        r, xi, _, _ = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, a.steps_between)
        energies.append(float(e_loc(params, r, xi, t).real.mean()) / N)
    mean, err = blocking_error(energies)
    print(f"\nE/N (variational)   {mean:.4f} +- {err:.4f} MeV   [{a.correlations}]")
    print("operator correlation parameters (rows: s, t, st, tn, tnt):")
    print(np.array2string(np.asarray(params["corr"]), precision=3))


if __name__ == "__main__":
    main()
