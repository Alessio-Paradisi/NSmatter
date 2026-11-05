"""Run run_nv2.py on a grid of models x densities (the EOS scan).

Each point runs in its own process (clean memory, one crash does not stop the
scan) and appends to results/nv2_vmc.jsonl; its terminal output goes to
results/logs/. Points already in the results file are skipped, so an
interrupted scan can simply be restarted with the same command.
Options not listed below are passed on to run_nv2.py unchanged.

Usage (from ~/projects/nsmatter, venv active):
    python scripts/eos_scan.py --models Ib IIb --densities 0.04 0.08 0.12 0.16 0.20 0.24 \\
        -- --walkers 132 --iterations 100
    # with the NV3 three-nucleon force (neutron matter):
    python scripts/eos_scan.py --models Ib -- --three-body Ib* --walkers 132 --iterations 60
    # symmetric nuclear matter (28 nucleons):
    python scripts/eos_scan.py --models Ib --populations 7 7 7 7 --densities 0.12 0.16 0.20
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def done_points(path):
    """(model, density, populations, three_body) already in the results file (full Hamiltonian only)."""
    points = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("static_only"):
                    points.add((rec["model"], round(rec["density"], 6), tuple(rec["populations"]),
                                rec.get("three_body"), rec.get("wavefunction", "basic")))
    return points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["Ia", "Ib", "IIa", "IIb"])
    p.add_argument("--densities", nargs="+", type=float, default=[0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
    p.add_argument("--populations", nargs=4, type=int, default=[7, 7, 0, 0], metavar=("NU", "ND", "PU", "PD"))
    p.add_argument("--out", default="results/nv2_vmc.jsonl")
    a, extra = p.parse_known_args()
    extra = [x for x in extra if x != "--"]
    three_body = extra[extra.index("--three-body") + 1] if "--three-body" in extra else None
    wf = "LS" if "--spin-orbit-corr" in extra else "basic"

    out = pathlib.Path(a.out)
    logs = out.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pops = tuple(a.populations)
    nu, nd, pu, pd = pops

    todo = [(m, d) for m in a.models for d in a.densities
            if (m, round(d, 6), pops, three_body, wf) not in done_points(out)]
    print(f"{len(todo)} points to run (populations {pops}); extra options: {' '.join(extra) or '-'}")
    for k, (model, rho) in enumerate(todo):
        tb = f"+{three_body.replace('*', 's')}" if three_body else ""
        log = logs / f"{model}{tb}{'_LS' if wf == 'LS' else ''}_rho{rho:.3f}_{nu}{nd}{pu}{pd}.log"
        cmd = [sys.executable, str(HERE / "run_nv2.py"), "--model", model, "--density", str(rho),
               "--n-up", str(nu), "--n-down", str(nd), "--p-up", str(pu), "--p-down", str(pd),
               "--out", str(out), *extra]
        t0 = time.time()
        print(f"[{k + 1}/{len(todo)}] NV2-{model} rho = {rho} ... ", end="", flush=True)
        with log.open("w") as f:
            status = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        last = log.read_text().strip().splitlines()
        result = next((x for x in reversed(last) if x.startswith("E/N (variational)")), "no result")
        print(f"{'ok' if status == 0 else 'FAILED'} ({(time.time() - t0) / 60:.1f} min)  {result}")
    print("scan finished")


if __name__ == "__main__":
    main()
