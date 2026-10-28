"""VMC with an NV2 potential: optimize Psi = exp(J) [1 + sum U_ij] Phi with SR, then measure E/N.

Default: pure neutron matter, 14 neutrons (7 up + 7 down, closed shells) at
rho = 0.16 fm^-3 with NV2-Ib. Any of the four populations can be set, e.g.
symmetric nuclear matter with 28 nucleons: --n-up 7 --n-down 7 --p-up 7 --p-down 7.
The result is appended to results/nv2_vmc.jsonl (one JSON line per run).

Usage (from ~/projects/nsmatter, venv active):
    python scripts/run_nv2.py --model Ib --density 0.16
    python scripts/run_nv2.py --model IIb --density 0.08 --iterations 150
    python scripts/run_nv2.py --static-only   # no L.S, L^2, (L.S)^2: quick test only

Parallelism: on CPU the walkers are split over --devices virtual devices (default:
all cores), one per core. --walkers must be a multiple of --devices.
Stability: if E/N rises by more than 3 standard errors after an SR step, the
step is undone and the learning rate halved.
"""

import argparse
import json
import os
import pathlib
import sys
import time


def _cpu_devices_from_argv():
    """--devices must be known before jax is imported (it sets the XLA flags)."""
    for k, arg in enumerate(sys.argv):
        if arg == "--devices" and k + 1 < len(sys.argv):
            return int(sys.argv[k + 1])
        if arg.startswith("--devices="):
            return int(arg.split("=", 1)[1])
    return len(os.sched_getaffinity(0))


os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + f" --xla_force_host_platform_device_count={_cpu_devices_from_argv()}")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import nsmatter  # noqa: F401, E402  (float64)
from nsmatter.local_energy import make_local_energy  # noqa: E402
from nsmatter.optimize import apply_update, log_derivatives_spin, sr_direction  # noqa: E402
from nsmatter.potentials.nv2_vmc import make_nv2_potential, model_key  # noqa: E402
from nsmatter.sampler import make_log_prob_spin, metropolis_spin, random_spinors, thermalize_spin, uniform_walkers  # noqa: E402
from nsmatter.stats import blocking_error  # noqa: E402
from nsmatter.system import System  # noqa: E402
from nsmatter.wavefunctions.correlations import make_correlated  # noqa: E402
from nsmatter.wavefunctions.slater import free_gas_energy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Ib", help="Ia, Ib, IIa, IIb (or a, b, at, bt)")
    p.add_argument("--n-up", type=int, default=7)
    p.add_argument("--n-down", type=int, default=7)
    p.add_argument("--p-up", type=int, default=0)
    p.add_argument("--p-down", type=int, default=0)
    p.add_argument("--density", type=float, default=0.16, help="fm^-3")
    p.add_argument("--static-only", action="store_true", help="drop momentum-dependent operators")
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--walkers", type=int, default=256)
    p.add_argument("--batch", type=int, default=4, help="walkers per local-energy chunk (memory)")
    p.add_argument("--devices", type=int, default=None, help="CPU devices (default: all cores)")
    p.add_argument("--steps-between", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001, help="SR step (MeV^-1)")
    p.add_argument("--diag-shift", type=float, default=0.1)
    p.add_argument("--measurements", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/nv2_vmc.jsonl")
    return p.parse_args()


def main():
    a = parse_args()
    system = System(a.n_up, a.n_down, a.p_up, a.p_down, a.density)
    N, L = system.n_particles, system.box_length
    print(system.describe())
    print(f"model NV2-{a.model} ({model_key(a.model)}) | static only: {a.static_only} | devices: {jax.local_device_count()} x {jax.devices()[0].platform}")
    print(f"free gas E/N = {free_gas_energy(system) / N:.4f} MeV")

    params, log_psi = make_correlated(system, n_terms=a.n_terms)
    t = jnp.asarray(system.isospins)
    potential = make_nv2_potential(system, a.model, momentum_terms=not a.static_only)

    # Local energy in chunks of --batch walkers (vmap inside a chunk, sequential
    # over chunks): the momentum operators are memory-hungry.
    # The walkers are split over the devices (pmap), one chunk per core.
    el_one = make_local_energy(log_psi, (potential,))
    n_dev = jax.local_device_count()
    if a.walkers % n_dev:
        raise ValueError(f"--walkers ({a.walkers}) must be a multiple of the number of devices ({n_dev})")
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("w",))
    P = jax.sharding.PartitionSpec
    e_chunk = jax.shard_map(  # each device gets walkers / n_dev walkers
        lambda p, r, xi, t: jax.lax.map(lambda w: el_one(p, w[0], w[1], t), (r, xi), batch_size=a.batch),
        mesh=mesh, in_specs=(P(), P("w"), P("w"), P()), out_specs=P("w"))
    e_loc = jax.jit(e_chunk)

    log_derivs = jax.jit(lambda p, r, xi: log_derivatives_spin(log_psi, p, r, xi, t))
    log_prob = make_log_prob_spin(log_psi, t)

    key = jax.random.PRNGKey(a.seed)
    k1, k2, key = jax.random.split(key, 3)
    r = uniform_walkers(k1, a.walkers, N, L)
    xi = random_spinors(k2, (a.walkers, N))
    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, 0.1 * L / np.sqrt(N), 0.5, L)

    print("\n step   E/N (MeV)    +-     var(E_L)/N^2  acc_r  acc_s   s/iter")
    t0 = time.time()
    history = []
    lr = a.lr
    previous = None  # (params, E/N, error) before the last SR step
    for it in range(a.iterations):
        t_it = time.time()
        key, sub = jax.random.split(key)
        r, xi, acc_r, acc_s = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, a.steps_between)
        e = e_loc(params, r, xi, t).real
        mean, err = float(e.mean()) / N, float(e.std()) / np.sqrt(a.walkers) / N
        if not np.isfinite(mean):
            raise RuntimeError("energy is not finite: reduce --lr or increase --diag-shift")
        note = ""
        if previous is not None and mean - previous[1] > 3 * np.hypot(err, previous[2]):
            # The last step made things clearly worse: undo it, halve the step.
            params, lr = previous[0], lr / 2
            note = f"  step undone, lr -> {lr:.2e}"
            previous = None
        else:
            O = log_derivs(params, r, xi)
            delta, _, _ = sr_direction(O, e, diag_shift=a.diag_shift)
            previous = (params, mean, err)
            params = apply_update(params, delta, lr)
        history.append(mean)
        print(f"{it:5d}  {mean:10.4f}  {err:7.4f}  {float(e.var()) / N**2:12.4f}  "
              f"{float(acc_r):5.3f}  {float(acc_s):5.3f}  {time.time() - t_it:6.1f}{note}", flush=True)
    print(f"optimization: {time.time() - t0:.1f} s")

    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, step, spin_step, L, n_rounds=5)
    energies = []
    for _ in range(a.measurements):
        key, sub = jax.random.split(key)
        r, xi, _, _ = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, a.steps_between)
        energies.append(float(e_loc(params, r, xi, t).real.mean()) / N)
    mean, err = blocking_error(energies)
    print(f"\nE/N (variational)   {mean:.4f} +- {err:.4f} MeV   [NV2-{a.model}, rho = {a.density}]")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": a.model, "density": a.density, "populations": [a.n_up, a.n_down, a.p_up, a.p_down],
        "static_only": a.static_only, "walkers": a.walkers, "iterations": a.iterations,
        "E_per_N": mean, "error": err, "history": history,
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), params),
    }
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
