"""Optimize a Jastrow-Slater wavefunction with SR, then measure its energy.

Two phases:
1. optimization: at each step, sample, compute E_L and the log-derivatives,
   move the parameters along the SR direction. The energy printed here is
   noisy and biased by the moving parameters: it only shows convergence.
2. evaluation: parameters frozen, long sampling run, blocking error.
   This is the number to report (a variational upper bound).

Usage (from ~/projects/nsmatter):
    uv run python scripts/optimize_jastrow.py
    uv run python scripts/optimize_jastrow.py --n-up 7 --n-down 7 --p-up 7 --p-down 7 --lr 0.001
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import nsmatter  # noqa: F401  (float64)
from nsmatter.hamiltonian.central import hartree_fock_energy, make_central_potential
from nsmatter.local_energy import make_local_energy
from nsmatter.optimize import apply_update, log_derivatives, sr_direction
from nsmatter.sampler import make_log_prob, metropolis, thermalize, uniform_walkers
from nsmatter.stats import blocking_error, integrated_autocorrelation_time
from nsmatter.system import System
from nsmatter.wavefunctions.jastrow import make_jastrow_slater
from nsmatter.wavefunctions.slater import free_gas_energy


def parse_args():
    p = argparse.ArgumentParser()
    # Composition and density.
    p.add_argument("--n-up", type=int, default=7)
    p.add_argument("--n-down", type=int, default=7)
    p.add_argument("--p-up", type=int, default=0)
    p.add_argument("--p-down", type=int, default=0)
    p.add_argument("--density", type=float, default=0.16, help="fm^-3")
    # Wavefunction.
    p.add_argument("--n-terms", type=int, default=3, help="basis functions per Jastrow channel")
    # Sampling.
    p.add_argument("--walkers", type=int, default=256)
    p.add_argument("--steps-between", type=int, default=10, help="Metropolis steps between measurements")
    # Optimization.
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.002, help="SR step (MeV^-1); smaller for larger N")
    p.add_argument("--diag-shift", type=float, default=0.05)
    # Evaluation.
    p.add_argument("--measurements", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="jastrow_params.npz", help="where to save the optimized parameters")
    return p.parse_args()


def main():
    a = parse_args()

    # --- system, wavefunction, Hamiltonian ----------------------------------
    system = System(a.n_up, a.n_down, a.p_up, a.p_down, a.density)
    N, L = system.n_particles, system.box_length
    print(system.describe())
    print("device:", jax.devices()[0])

    params, log_psi = make_jastrow_slater(system, n_terms=a.n_terms)
    s, t = jnp.asarray(system.spins), jnp.asarray(system.isospins)
    potential = make_central_potential(system)

    # Compiled functions, all batched over walkers.
    e_loc = jax.jit(jax.vmap(make_local_energy(log_psi, (potential,)), in_axes=(None, 0, None, None)))
    log_derivs = jax.jit(lambda p, r: log_derivatives(log_psi, p, r, s, t))
    log_prob = make_log_prob(log_psi, s, t)

    # Reference: J = 0 gives exactly free gas + Hartree-Fock.
    start = (free_gas_energy(system) + hartree_fock_energy(system)) / N
    print(f"starting energy (J = 0, exact): {start:.4f} MeV per particle")

    # --- initial walkers and thermalization ---------------------------------
    key = jax.random.PRNGKey(a.seed)
    key, sub = jax.random.split(key)
    r = uniform_walkers(sub, a.walkers, N, L)
    r, step, key = thermalize(log_prob, params, key, r, 0.1 * L / np.sqrt(N), L)

    # --- phase 1: optimization ----------------------------------------------
    print("\n step   E/N (MeV)   +-      var(E_L)/N^2   acc    |delta|")
    t0 = time.time()
    for it in range(a.iterations):
        key, sub = jax.random.split(key)
        r, acc = metropolis(log_prob, params, sub, r, step, L, a.steps_between)
        e = e_loc(params, r, s, t).real
        O = log_derivs(params, r)
        delta, _, _ = sr_direction(O, e, diag_shift=a.diag_shift)
        params = apply_update(params, delta, a.lr)
        # Walkers are independent chains, so at a single step the error of
        # the walker average is simply std / sqrt(W).
        mean, err = float(e.mean()) / N, float(e.std()) / np.sqrt(a.walkers) / N
        var = float(e.var()) / N**2
        print(f"{it:5d}  {mean:10.4f}  {err:7.4f}  {var:12.4f}  {float(acc):5.3f}  {float(jnp.linalg.norm(a.lr * delta)):.4f}")
        # Stop at the first sign of a blow-up instead of continuing with garbage.
        if not np.isfinite(mean):
            raise RuntimeError("energy is not finite: reduce --lr or increase --diag-shift")
    print(f"optimization: {time.time() - t0:.1f} s")

    # --- phase 2: evaluation with frozen parameters -------------------------
    # Re-thermalize: the last parameter change moved the target distribution.
    r, step, key = thermalize(log_prob, params, key, r, step, L, n_rounds=5)
    energies = []
    t0 = time.time()
    for _ in range(a.measurements):
        key, sub = jax.random.split(key)
        r, _ = metropolis(log_prob, params, sub, r, step, L, a.steps_between)
        energies.append(float(e_loc(params, r, s, t).real.mean()) / N)
    mean, err = blocking_error(energies)
    print(f"\nevaluation: {a.measurements} measurements ({time.time() - t0:.1f} s)")
    print(f"E/N (variational)   {mean:.4f} +- {err:.4f} MeV")
    print(f"gain over J = 0     {start - mean:.4f} MeV per particle")
    print(f"tau_int             {integrated_autocorrelation_time(energies):.2f} measurements")

    # --- save ----------------------------------------------------------------
    np.savez(a.output, jastrow=np.asarray(params["jastrow"]), density=a.density, counts=np.array(system.counts))
    print(f"parameters saved to {a.output}")


if __name__ == "__main__":
    main()
