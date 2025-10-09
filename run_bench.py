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
from separators.python.demand_cover_py import DemandReserveCoverSepa
from separators.python.min_generation_cover_py import MinGenCoverSepa

def run_one(mps_path: Path, scenario: str, timelimit: int, threads: int) -> Dict[str, Any]:
    json_sidecar = mps_path.with_suffix(".minud.json")

    m = Model()
    m.setParam("limits/time", float(timelimit))
    m.setParam("parallel/maxnthreads", threads)
    m.readProblem(str(mps_path))

    # choose which separator to build (or None for baseline)
    sepa_factory = None
    if scenario == "minupdown":
        sepa_factory = lambda: MinUpDownSepa(
            minviolation=1e-4, rootonly=False,
            json_path=str(json_sidecar) if json_sidecar.exists() else None,
            maxcuts_per_round=100
        )
    elif scenario == "demcov":
        sepa_factory = lambda: DemandReserveCoverSepa(
            json_path=str(json_sidecar) if json_sidecar.exists() else None,
            minvio=1e-4, rootonly=False, maxcuts_per_t=1
        )
    elif scenario == "mingen":
        sepa_factory = lambda: MinGenCoverSepa(
            json_path=str(json_sidecar) if json_sidecar.exists() else None,
            minvio=1e-4, rootonly=False, maxcuts_per_t=1
        )

    row = solve_with(m, sepa_factory)
    row.update({"instance": mps_path.name, "scenario": scenario, "best_obj": row.get("primal_bound")})
    return row



from typing import Callable, Optional, Dict, Any
import time
from pyscipopt import Model

def solve_with(model: Model, sepa_factory: Optional[Callable[[], object]]) -> Dict[str, Any]:
    """
    If sepa_factory is provided, build the separator and include it
    exactly once on 'model'. Then optimize and return standard metrics.
    """
    sepa_obj = None
    if sepa_factory is not None:
        sepa_obj = sepa_factory()
        # unique name per class avoids name collisions
        model.includeSepa(sepa_obj, sepa_obj.__class__.__name__, sepa_obj.__doc__ or "", 900, 1)

    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0

    status = model.getStatus()
    bestsol = model.getBestSol()
    row = {
        "status": status,
        "solve_time": model.getSolvingTime(),
        "elapsed_wall": elapsed,
        "reading_time": model.getReadingTime(),
        "presolving_time": model.getPresolvingTime(),
        "n_nodes": model.getNNodes(),
        "n_total_nodes": model.getNTotalNodes(),
        "n_lps": model.getNLPs(),
        "lp_iterations": model.getNLPIterations(),
        "n_cuts_in_pool": model.getNCuts(),
        "n_cuts_applied": model.getNCutsApplied(),
        "n_sepa_rounds": model.getNSepaRounds(),
        "primal_bound": model.getPrimalbound(),
        "dual_bound": model.getDualbound(),
        "gap": model.getGap(),
        "n_solutions": model.getNSolsFound(),
        "time_best_sol": model.getSolTime(bestsol) if bestsol is not None else None,
    }
    # if the sepa tracks a counter, include it
    if sepa_obj is not None and hasattr(sepa_obj, "ncuts_added"):
        row["custom_cuts"] = getattr(sepa_obj, "ncuts_added", 0)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="baseline,minupdown",
                 help="comma-separated list: baseline,minupdown,demcov,mingen")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", default="bench_results.csv")
    ap.add_argument("--timelimit", type=int, default=600)   # 10 minutes
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    instances = sorted(glob.glob(os.path.join(args.data, "*.mps")))
    if not instances:
        raise SystemExit(f"No .mps found under {args.data}")

    # rows = []
    # for inst in instances:
    #     p = Path(inst)
    #     json_sidecar = p.with_suffix(".minud.json")  # optional sidecar with Lup/Ldown
    #     # baseline
    #     rows.append(run_one(p, "baseline", json_sidecar=None, timelimit=args.timelimit, threads=args.threads))
    #     # each separator alone
    #     rows.append(run_one(p, "minupdown", json_sidecar=json_sidecar, timelimit=args.timelimit, threads=args.threads))

    rows = []
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    for inst in instances:
        p = Path(inst)
        # use the sidecar name you’re actually writing; if you followed my earlier patch it’s .minud.json
        for s in scenarios:
            rows.append(run_one(p, s, timelimit=args.timelimit, threads=args.threads))



    # Write CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
