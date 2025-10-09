# scripts/run_bench.py
from __future__ import annotations
import argparse, csv, glob, json, os, time
from pathlib import Path
from typing import Dict, Any, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyscipopt import Model
from separators.python.minupdown_py import MinUpDownSepa

def run_one(mps_path: Path, scenario: str, json_sidecar: Optional[Path], timelimit: int, threads: int) -> Dict[str, Any]:
    m = Model()
    m.setParam("limits/time", float(timelimit))
    m.setParam("parallel/maxnthreads", threads)
    m.readProblem(str(mps_path))
    # If sidecar JSON exists, feed it to the sepa via a param (the sepa reads it)
    if json_sidecar and json_sidecar.exists():
        # Create a user string param if it doesn't exist; fallback to setting on sepa object via attribute
        try:
            m.addStringParam("sepa/minupdown/json", "path to minupdown JSON", "", None, None)
            m.setParam("sepa/minupdown/json", str(json_sidecar))
        except Exception:
            pass

    # Add separator(s) for scenarios
    sepa_objects = []
    if scenario == "minupdown":
        sepa = MinUpDownSepa(minviolation=1e-3, rootonly=False)
        m.includeSepa(sepa, "minupdown_py", "min up/down separator (python)", priority=1000, freq=1)
        sepa_objects.append(sepa)

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    # Gather metrics (use PySCIPOpt API)
    status = m.getStatus()
    metrics: Dict[str, Any] = {
        "instance": mps_path.name,
        "scenario": scenario,
        "status": status,
        "solve_time": m.getSolvingTime(),            # seconds
        "elapsed_wall": elapsed,
        "reading_time": m.getReadingTime(),
        "presolving_time": m.getPresolvingTime(),
        "n_nodes": m.getNNodes(),
        "n_total_nodes": m.getNTotalNodes(),
        "n_lps": m.getNLPs(),
        "lp_iterations": m.getNLPIterations(),

        "n_cuts_in_pool": m.getNCuts(),
        "n_cuts_applied": m.getNCutsApplied(),
        "n_sepa_rounds": m.getNSepaRounds(),
        "primal_bound": m.getPrimalbound(),
        "dual_bound": m.getDualbound(),
        "gap": m.getGap(),                           # relative gap
        "best_obj": m.getObjVal(original=True) if status in ("optimal","timelimit","gaplimit","bestsollimit") else None,
        "n_solutions": m.getNSolsFound(),
    }

    # Best solution timing (if any)
    sols = m.getSols()
    if sols:
        # time when SCIP found the best known solution
        metrics["time_best_sol"] = m.getSolTime(sols[0])
    else:
        metrics["time_best_sol"] = None

    # Separator-specific counters
    if scenario == "minupdown" and sepa_objects:
        metrics["minupdown_cuts"] = sepa_objects[0].ncuts_added

    # (Optional) print a one-line log
    print(f"[{scenario}] {mps_path.name}: status={status} time={metrics['solve_time']:.2f}s nodes={metrics['n_nodes']} gap={metrics['gap']:.4f}")

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", default="bench_results.csv")
    ap.add_argument("--timelimit", type=int, default=600)   # 10 minutes
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    instances = sorted(glob.glob(os.path.join(args.data, "*.mps")))
    if not instances:
        raise SystemExit(f"No .mps found under {args.data}")

    rows = []
    for inst in instances:
        p = Path(inst)
        json_sidecar = p.with_suffix(".json")  # optional sidecar with Lup/Ldown
        # baseline
        rows.append(run_one(p, "baseline", json_sidecar=None, timelimit=args.timelimit, threads=args.threads))
        # each separator alone
        rows.append(run_one(p, "minupdown", json_sidecar=json_sidecar, timelimit=args.timelimit, threads=args.threads))

    # Write CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
