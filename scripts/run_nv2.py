"""VMC with an NV2 potential: optimize Psi = exp(J) [1 + sum U_ij] Phi with SR, then measure E/N.

Default: pure neutron matter, 14 neutrons (7 up + 7 down, closed shells) at
rho = 0.16 fm^-3 with NV2-Ib. Any of the four populations can be set, e.g.
symmetric nuclear matter with 28 nucleons: --n-up 7 --n-down 7 --p-up 7 --p-down 7.
The result is appended to results/nv2_vmc.jsonl (one JSON line per run) with:
  E/N and its error, kinetic energy and the potential energy of every NV2 channel
  (all per particle, with errors; also summed by operator group, nv2_vmc.GROUPS),
  pair distributions g_nn, g_np, g_pp(r), spin and
  isospin correlations C_sigma(r), C_tau(r) (observables.py), <S^2>, <T^2>, the
  free-gas energies (box and infinite), all settings, code version and timings.
Configurations go to results/configs/<run>.npz: the final positions, spinors and
spin directions of all walkers, a trajectory of --track walkers (one frame per
measurement, for animations) and the per-measurement series of every quantity.

Usage (from ~/projects/nsmatter, venv active):
    python scripts/run_nv2.py --model Ib --density 0.16
    python scripts/run_nv2.py --model IIb --density 0.08 --iterations 150
    python scripts/run_nv2.py --static-only   # no L.S, L^2, (L.S)^2: quick test only
    python scripts/run_nv2.py --model Ib --three-body Ib*   # NV2+3 (pure neutron matter only)
    python scripts/run_nv2.py --model Ib --spin-orbit-corr  # trial wavefunction with L.S correlations

Parallelism: on CPU the walkers are split over --devices virtual devices (default:
all cores), one per core. --walkers must be a multiple of --devices.
Stability: if E/N rises by more than 3 standard errors after an SR step, the
step is undone and the learning rate halved.
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings


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
from nsmatter.local_energy import make_local_energy, make_local_energy_components  # noqa: E402
from nsmatter.observables import CLASSES, make_pair_observables, normalize, spin_vectors  # noqa: E402
from nsmatter.optimize import apply_update, log_derivatives_spin, sr_direction  # noqa: E402
from nsmatter.potentials.nv2_vmc import GROUPS, make_nv2_potential, model_key  # noqa: E402
from nsmatter.potentials.nv3 import CD_CE, make_nv3_potential  # noqa: E402
from nsmatter.sampler import make_log_prob_spin, metropolis_spin, random_spinors, thermalize_spin, uniform_walkers  # noqa: E402
from nsmatter.stats import blocking_error  # noqa: E402
from nsmatter.system import System  # noqa: E402
from nsmatter.wavefunctions.correlations import make_correlated  # noqa: E402
from nsmatter.wavefunctions.slater import free_gas_energy, free_gas_energy_infinite  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Ib", help="Ia, Ib, IIa, IIb (or a, b, at, bt)")
    p.add_argument("--n-up", type=int, default=7)
    p.add_argument("--n-down", type=int, default=7)
    p.add_argument("--p-up", type=int, default=0)
    p.add_argument("--p-down", type=int, default=0)
    p.add_argument("--density", type=float, default=0.16, help="fm^-3")
    p.add_argument("--static-only", action="store_true", help="drop momentum-dependent operators")
    p.add_argument("--three-body", default=None, choices=sorted(CD_CE),
                   help="add the NV3 force of this model (e.g. Ib*; same NV2 model as --model); neutron matter only")
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--spin-orbit-corr", action="store_true",
                   help="add spin-orbit correlations u(r) L.S [tau.tau] to the trial wavefunction")
    p.add_argument("--walkers", type=int, default=256)
    p.add_argument("--batch", type=int, default=None,
                   help="walkers per local-energy chunk per core (memory); default 4, or 1 with --spin-orbit-corr")
    p.add_argument("--devices", type=int, default=None, help="CPU devices (default: all cores)")
    p.add_argument("--steps-between", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001, help="SR step (MeV^-1)")
    p.add_argument("--diag-shift", type=float, default=0.1)
    p.add_argument("--measurements", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=40, help="distance bins for g(r), C(r) (up to L/2)")
    p.add_argument("--track", type=int, default=4, help="walkers whose trajectory is saved (animations)")
    p.add_argument("--out", default="results/nv2_vmc.jsonl")
    return p.parse_args()


def main():
    a = parse_args()
    if a.batch is None:  # spin-orbit correlations need about 1.8 GB per walker in flight
        a.batch = 1 if a.spin_orbit_corr else 4
    system = System(a.n_up, a.n_down, a.p_up, a.p_down, a.density)
    N, L = system.n_particles, system.box_length
    print(system.describe())
    print(f"model NV2-{a.model} ({model_key(a.model)}) | static only: {a.static_only} | devices: {jax.local_device_count()} x {jax.devices()[0].platform}")
    print(f"free gas E/N = {free_gas_energy(system) / N:.4f} MeV")

    params, log_psi = make_correlated(system, n_terms=a.n_terms, spin_orbit=a.spin_orbit_corr)
    if a.spin_orbit_corr:
        print("wavefunction: with spin-orbit correlations")
    t = jnp.asarray(system.isospins)
    potentials = (make_nv2_potential(system, a.model, momentum_terms=not a.static_only),)
    if a.three_body:
        if model_key(a.three_body.rstrip("*")) != model_key(a.model):
            raise ValueError(f"--three-body {a.three_body} belongs to another NV2 model than --model {a.model}")
        potentials = potentials + (make_nv3_potential(system, a.three_body),)
        print(f"three-body force: NV3-{a.three_body} (cD, cE = {CD_CE[a.three_body]})")

    # Local energy in chunks of --batch walkers (vmap inside a chunk, sequential
    # over chunks): the momentum operators are memory-hungry.
    # The walkers are split over the devices (pmap), one chunk per core.
    el_one = make_local_energy(log_psi, potentials)
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

    t_opt = time.time() - t0

    # --- measurement: energy components and pair observables in one pass ---------
    comps_one = make_local_energy_components(log_psi, potentials)
    observe = make_pair_observables(system, n_bins=a.bins)

    def measure_one(p, r_w, xi_w, t):
        return comps_one(p, r_w, xi_w, t), observe(log_psi, p, r_w, xi_w, t)

    m_chunk = jax.shard_map(
        lambda p, r, xi, t: jax.lax.map(lambda w: measure_one(p, w[0], w[1], t), (r, xi), batch_size=a.batch),
        mesh=mesh, in_specs=(P(), P("w"), P("w"), P()), out_specs=P("w"))
    measure = jax.jit(m_chunk)

    r, xi, step, spin_step, key = thermalize_spin(log_prob, params, key, r, xi, step, spin_step, L, n_rounds=5)
    t1 = time.time()
    series = {}  # name -> list over measurements (walker averages, per particle for energies)
    pair_series = {k: [] for k in ("counts", "sigma", "tau")}
    traj_r, traj_spin = [], []
    for _ in range(a.measurements):
        key, sub = jax.random.split(key)
        r, xi, acc_r, acc_s = metropolis_spin(log_prob, params, sub, r, xi, step, spin_step, L, a.steps_between)
        comps, obs = measure(params, r, xi, t)
        comps = {k: np.asarray(v).real for k, v in comps.items()}  # (walkers,) each
        total = sum(comps.values())
        potential_sum = total - comps["kinetic"]
        row = {"total": total, "kinetic": comps["kinetic"], "potential": potential_sum}
        for g, chans in GROUPS.items():
            present = [f"v_{c}" for c in chans if f"v_{c}" in comps]
            if present:
                row[f"group_{g}"] = sum(comps[c] for c in present)
        for g, chans in GROUPS.items():
            row.update({f"v_{c}": comps[f"v_{c}"] for c in chans if f"v_{c}" in comps})
        for k, v in row.items():
            series.setdefault(k, []).append(float(v.mean()) / N)
        series.setdefault("variance", []).append(float(total.var()) / N**2)
        series.setdefault("S2", []).append(float((np.asarray(obs["sigma_total"]).mean() + 1.5 * N) / 2))
        series.setdefault("T2", []).append(float((np.asarray(obs["tau_total"]).mean() + 1.5 * N) / 2))
        series.setdefault("acc_r", []).append(float(acc_r))
        series.setdefault("acc_s", []).append(float(acc_s))
        for k in pair_series:
            pair_series[k].append(np.asarray(obs[k]).sum(axis=0))  # summed over walkers, (3, bins)
        traj_r.append(np.asarray(r[: a.track]))
        traj_spin.append(np.asarray(spin_vectors(xi[: a.track])))
    t_meas = time.time() - t1

    mean, err = blocking_error(series["total"])
    label = f"NV2-{a.model}" + (f" + NV3-{a.three_body}" if a.three_body else "")
    print(f"\nE/N (variational)   {mean:.4f} +- {err:.4f} MeV   [{label}, rho = {a.density}]")
    components = {k: list(map(float, blocking_error(v))) for k, v in series.items()}
    print("  kinetic   {:9.4f} +- {:.4f}".format(*components["kinetic"]))
    print("  potential {:9.4f} +- {:.4f}".format(*components["potential"]))
    for g, chans in GROUPS.items():
        if f"group_{g}" in components:
            print("    {:17s} {:9.4f} +- {:.4f}".format(g, *components[f"group_{g}"]))
            for c in chans:
                if f"v_{c}" in components:
                    print("      {:15s} {:9.4f} +- {:.4f}".format(c, *components[f"v_{c}"]))

    # pair observables: normalized per measurement, then mean and standard error
    per_step = [normalize({k: pair_series[k][n] for k in pair_series}, a.walkers, system, observe.dr)
                for n in range(a.measurements)]
    structure = {"r": observe.bins.tolist()}
    for q in ("g", "C_sigma", "C_tau"):
        stack = np.array([ps[q] for ps in per_step])  # (meas, 3, bins); NaN where a bin had no pairs
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(stack, axis=0)
            e_ = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(stack), axis=0), 1))
        structure[q] = {c: np.where(np.isfinite(m[k]), m[k], None).tolist() for k, c in enumerate(CLASSES)}
        structure[q + "_err"] = {c: np.where(np.isfinite(e_[k]), e_[k], None).tolist() for k, c in enumerate(CLASSES)}

    # --- outputs ---------------------------------------------------------------------
    out = pathlib.Path(a.out)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = (f"{a.model}{'+' + a.three_body.replace('*', 's') if a.three_body else ''}"
           f"{'_LS' if a.spin_orbit_corr else ''}"
           f"_rho{a.density:.3f}_{a.n_up}{a.n_down}{a.p_up}{a.p_down}_{stamp}")
    cfg_path = out.parent / "configs" / f"{tag}.npz"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg_path, r=np.asarray(r), xi=np.asarray(xi), spin=np.asarray(spin_vectors(xi)), t=np.asarray(t),
        L=L, density=a.density, traj_r=np.array(traj_r), traj_spin=np.array(traj_spin),
        pair_counts=np.array(pair_series["counts"]), pair_sigma=np.array(pair_series["sigma"]),
        pair_tau=np.array(pair_series["tau"]), bins=observe.bins, dr=observe.dr,
        **{f"series_{k}": np.array(v) for k, v in series.items()})

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=pathlib.Path(__file__).parent,
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    settings = {k: v for k, v in vars(a).items() if k not in ("out",)}
    record = {
        "model": a.model, "density": a.density, "populations": [a.n_up, a.n_down, a.p_up, a.p_down],
        "static_only": a.static_only, "three_body": a.three_body,
        "wavefunction": "LS" if a.spin_orbit_corr else "basic", "walkers": a.walkers, "iterations": a.iterations,
        "E_per_N": mean, "error": err,
        "N": N, "L": L, "proton_fraction": system.proton_fraction, "polarization": system.polarization,
        "free_gas_box": free_gas_energy(system) / N, "free_gas_infinite": free_gas_energy_infinite(system) / N,
        "components": components,  # name -> [mean, error], per particle (energies in MeV)
        "structure": structure,
        "history": history,
        "params": jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), params),
        "settings": settings, "commit": commit, "date": stamp,
        "time_optimization_s": t_opt, "time_measurement_s": t_meas, "devices": n_dev,
        "configs": str(cfg_path),
    }
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved to {out} and {cfg_path}")


if __name__ == "__main__":
    main()
