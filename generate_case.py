#!/usr/bin/env python
"""Example code to generate and solve a Unit Commitment (UC) MILP model."""

import sys
import os

import argparse
import pyomo.environ as pyo
from helpers import parsecase
import pickle
import model as ucml
import pandapower as pp
# import pandas as pd
# import matplotlib.pyplot as plt
from pyomo.opt import SolverStatus, TerminationCondition
import json

output_dir = os.environ.get("OUTPUT_DIR", "data")
START_ID = 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parser = argparse.ArgumentParser(
    description="Generate a case file with optional verbose output and no-solving option"
)
parser.add_argument(
    "bus_system",
    type=str,
    help="Name of the case to generate"
)
parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="Enable verbose output"
)
parser.add_argument(
    "--nosolve",
    action="store_true",
    help="Generate the case without solving it"
)
args = parser.parse_args()

bus_system = args.bus_system

def export_mps_only(model, outpath):
    """
    Write an MPS with symbolic names (so variable names don't get mangled).
    Tries Gurobi persistent (fast) and falls back to LP writer if needed.
    """
    # Try GurobiPersistent without optimizing
    try:
        from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
        opt = GurobiPersistent()
        opt.set_instance(model)
        # Write with symbolic labels
        opt.write(outpath, io_options={"symbolic_solver_labels": True})
        return
    except Exception:
        pass

    # Fallback: write LP (readable) if MPS writer isn't available
    try:
        model.write(outpath.replace(".mps", ".lp"), io_options={"symbolic_solver_labels": True})
        return
    except Exception as e:
        raise RuntimeError(f"Failed to export MPS/LP: {e}")


def load_bus_system(bus_system_name):
    bus_systems = {
        "case4gs": pp.networks.case4gs,
        "case5": pp.networks.case5,
        "case6ww": pp.networks.case6ww,
        "case9": pp.networks.case9,
        "case14": pp.networks.case14,
        "case24_ieee_rts": pp.networks.case24_ieee_rts,
        "case30": pp.networks.case30,
        "case33bw": pp.networks.case33bw,
        "case39": pp.networks.case39,
        "case57": pp.networks.case57,
        "case118": pp.networks.case118,
        "case145": pp.networks.case145,
        "illinois200": pp.networks.case_illinois200,
        "case300": pp.networks.case300,
        "case1354pegase": pp.networks.case1354pegase,
        "case1888rte": pp.networks.case1888rte,
        "case2848rte": pp.networks.case2848rte,
        "case2869pegase": pp.networks.case2869pegase,
        "case3120sp": pp.networks.case3120sp,
        "case6470rte": pp.networks.case6470rte,
    }

    if bus_system_name.lower() in bus_systems:
        net = bus_systems[bus_system_name.lower()]()
    else:
        raise ValueError(f"Bus system '{bus_system_name}' not recognized.")
    return net


net = load_bus_system(bus_system)
if "gen" in net and len(net.gen) > 0:
    thermal_gens = net.gen.reset_index(drop=True)
elif "ext_grid" in net and len(net.ext_grid) > 0:
    thermal_gens = net.ext_grid.reset_index(drop=True)
else:
    raise ValueError("No generators or ext_grid found in this network!")
num_therm = len(thermal_gens)
num_nodes = len(net.bus)
num_lines = len(net.line)
time_periods = 24


def get_next_output_id():
    counter_file = "data/output_counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            output_id = int(f.read()) + 1
    else:
        output_id = START_ID
    with open(counter_file, "w") as f:
        f.write(str(output_id))
    return output_id


output_id = get_next_output_id()
model_name = f"output_{output_id}_{bus_system}"

db = parsecase(
    net,
    thermal_gens=thermal_gens,
    time_periods=time_periods,
    num_nodes=num_nodes,
    num_lines=num_lines,
)

with open("UCdata.p", "rb") as f:
    p_data = pickle.loads(pickle.load(f))

# right after loading p_data:
try:
    # p_data[None] holds the parameters
    import json
    p = p_data[None]

    meta = {
        "Lup":   {str(g): int(p["Ton"][g])  for g in p["Ton"]},
        "Ldown": {str(g): int(p["Toff"][g]) for g in p["Toff"]},
        # Optional but useful for other separators:
        "Pmin":  {str(g): float(p["Pmin"][g]) for g in p.get("Pmin", {})} if "Pmin" in p else {},
        "Pmax":  {str(g): float(p["Pmax"][g]) for g in p.get("Pmax", {})} if "Pmax" in p else {},
        "Demand": [float(d) for d in p.get("Demand", [])] if "Demand" in p else [],
        "Reserve": [float(r) for r in p.get("Reserve", [])] if "Reserve" in p else []
    }
    with open(os.path.join(output_dir, model_name + ".minud.json"), "w") as jf:
        json.dump(meta, jf, indent=2)

except Exception:
    pass

num_nodes = len(net.bus)
num_lines = len(net.line)
num_demands = len(net.load)
model = ucml.opt_model_generator(
    num_therm=num_therm,
    time_periods=time_periods,
    num_nodes=num_nodes,
    num_lines=num_lines,
)



instance = model.create_instance(data=p_data)

mps_path = os.path.join(output_dir, model_name + ".mps")

if args.nosolve:
    # ensure symbolic names so separators can parse u/v/w indices
    export_mps_only(model, mps_path)
    print(f"[nosolve] exported {mps_path} (and .minud.json if available)")
    sys.exit(0)


instance.write(os.path.join(output_dir, model_name + ".mps"))
solver = pyo.SolverFactory("gurobi")
solver.options["NonConvex"] = 2
result = solver.solve(instance, tee=False)

if args.verbose:
    print("Variables:")
    for v in instance.component_data_objects(pyo.Var):
        print(f"  {v.name} = {v.value}")

    print("\nConstraints:")
    for c in instance.component_data_objects(pyo.Constraint):
        print(f"  {c.name}: {c.expr}")


if (
    result.solver.status == SolverStatus.ok
    and result.solver.termination_condition == TerminationCondition.optimal
):
    if hasattr(result.problem, "upper_bound") and hasattr(
        result.problem, "lower_bound"
    ):
        primal_bound = result.problem.upper_bound
        dual_bound = result.problem.lower_bound
        print("Primal bound:", primal_bound)
        print("Dual bound:", dual_bound)
    else:
        primal_bound = pyo.value(instance.obj)
        dual_bound = None

    with open(os.path.join(output_dir, model_name + ".json"), "w") as out:
        out.write(json.dumps({"dual_bound": dual_bound, "primal_bound": primal_bound}))
    print("Optimal value:", pyo.value(instance.obj))
else:
    print("No optimal value found")
    sys.exit(1)

# Gtherm = range(1, num_therm + 1)

# df = pd.DataFrame(
#    {
#        "thermal": [
#            sum(pyo.value(instance.p[g, t]) for g in Gtherm)
#            for t in range(1, time_periods + 1)
#        ],
#    }
# )

# uncomment for data table and plot generation
# df.to_csv(os.path.join(output_dir, model_name + "_results.csv"), index=False)

# ax = df[["thermal"]].plot.area(stacked=True, figsize=(12, 6))
# plt.xlabel("Time Period")
# plt.ylabel("Power Output (MW)")
# plt.title("Unit Commitment Results")
# plt.legend(title="Generator Type")
# plt.savefig(os.path.join(output_dir, model_name + "_plot.png"))
