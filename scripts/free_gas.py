"""VMC of the free Fermi gas in a periodic box (benchmark run).

Samples |Psi|^2 for the plane-wave Slater determinant and measures the local
energy. Psi is an exact eigenstate, so the VMC energy must equal the exact
box energy with zero variance. This checks sampler, kinetic energy and
statistics together, end to end.

Usage (from ~/projects/nsmatter):
    uv run python scripts/free_gas.py
    uv run python scripts/free_gas.py --n-up 7 --n-down 7 --p-up 7 --p-down 7
    uv run python scripts/free_gas.py --n-up 19 --n-down 19 --density 0.08
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.local_energy import make_local_energy
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking_error, integrated_autocorrelation_time
from nsmatter.system import System
from nsmatter.wavefunctions.slater import free_gas_energy, free_gas_energy_infinite, make_slater


def parse_args():
    """Command-line options: composition, density and Monte Carlo settings."""
    p = argparse.ArgumentParser()
    # Composition: number of particles per species (each a closed shell or 0).
    p.add_argument("--n-up", type=int, default=7)
    p.add_argument("--n-down", type=int, default=7)
    p.add_argument("--p-up", type=int, default=0)
    p.add_argument("--p-down", type=int, default=0)
    p.add_argument("--density", type=float, default=0.16, help="baryon density (fm^-3)")
    # Monte Carlo settings.
    p.add_argument("--walkers", type=int, default=256, help="parallel chains")
    p.add_argument("--measurements", type=int, default=200)
    p.add_argument("--steps-between", type=int, default=10, help="Metropolis steps between measurements")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()

    # --- system -------------------------------------------------------------
    system = System(a.n_up, a.n_down, a.p_up, a.p_down, a.density)
    N, L = system.n_particles, system.box_length
    print(system.describe())
    print("device:", jax.devices()[0])

    # --- wavefunction and compiled functions --------------------------------
    params, log_psi = make_slater(system)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    log_prob = make_log_prob(log_psi, s, t)
    # Local energy for all walkers at once: vmap over positions (axis 0),
    # share params and labels (None). jit compiles it on the first call.
    e_loc = jax.jit(jax.vmap(make_local_energy(log_psi), in_axes=(None, 0, None, None)))

    # --- initial walkers ----------------------------------------------------
    key = jax.random.PRNGKey(a.seed)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, a.walkers, N, L)

    # --- thermalization -----------------------------------------------------
    # Starting step: a small fraction of the box, smaller for more particles
    # (all particles move at once, so the acceptance drops with N).
    t0 = time.time()
    r, step, key = thermalize(log_prob, params, key, r, step=0.1 * L / np.sqrt(N), L=L)
    print(f"thermalized: step = {step:.4f} fm ({time.time() - t0:.1f} s)")

    # --- production ---------------------------------------------------------
    # Each measurement: a few Metropolis steps, then E_L on every walker,
    # averaged over walkers -> one number per measurement (per particle).
    energies, accs = [], []
    t0 = time.time()
    for _ in range(a.measurements):
        key, sub = jax.random.split(key)
        r, acc = metropolis(log_prob, params, sub, r, step, L, a.steps_between)
        e = e_loc(params, r, s, t)
        energies.append(float(e.real.mean()) / N)
        accs.append(float(acc))
    print(f"production: {a.measurements} measurements x {a.walkers} walkers ({time.time() - t0:.1f} s)")

    # --- results ------------------------------------------------------------
    mean, err = blocking_error(energies)
    exact = free_gas_energy(system) / N
    inf = free_gas_energy_infinite(system) / N
    print(f"acceptance          {np.mean(accs):.3f}")
    print(f"E/N VMC             {mean:.8f} +- {err:.2e} MeV")
    print(f"E/N exact (box)     {exact:.8f} MeV")
    print(f"E/N infinite gas    {inf:.8f} MeV")
    print(f"finite-size shift   {inf - exact:+.6f} MeV per particle")
    # With zero variance tau_int is meaningless: report it only if E_L fluctuates.
    if np.std(energies) > 1e-10 * abs(mean):
        print(f"tau_int             {integrated_autocorrelation_time(energies):.2f} measurements")
    else:
        print("tau_int             n/a (zero variance: Psi is an exact eigenstate)")


if __name__ == "__main__":
    main()
