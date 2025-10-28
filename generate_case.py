# Fixed generate_case.py (relevant section)
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
from pyomo.opt import SolverStatus, TerminationCondition
import json
import time

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

def export_mps_only(instance, outpath):
    """
    Write an MPS with symbolic names from a concrete instance.
    """
    try:
        # Try writing MPS directly
        instance.write(outpath, io_options={"symbolic_solver_labels": True})
        return
    except Exception as e1:
        # Try with LP format as fallback
        try:
            instance.write(outpath.replace(".mps", ".lp"), io_options={"symbolic_solver_labels": True})
            print(f"Note: Exported as LP format instead of MPS")
            return
        except Exception as e2:
            # Last resort: write without symbolic labels
            try:
                instance.write(outpath)
                print(f"Warning: Exported without symbolic labels")
                return
            except Exception as e3:
                raise RuntimeError(f"Failed to export MPS/LP: {e3}")

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

def get_next_output_id():
    counter_file = os.path.join(output_dir, "output_counter.txt")
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            output_id = int(f.read()) + 1
    else:
        output_id = START_ID
    with open(counter_file, "w") as f:
        f.write(str(output_id))
    return output_id

# Main execution
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

output_id = get_next_output_id()
model_name = f"output_{output_id}_{bus_system}"

# Generate data
db = parsecase(
    net,
    thermal_gens=thermal_gens,
    time_periods=time_periods,
    num_nodes=num_nodes,
    num_lines=num_lines,
)

with open("UCdata.p", "rb") as f:
    p_data = pickle.loads(pickle.load(f))

# Save metadata for separators
try:
    p = p_data[None]
    meta = {
        "Lup":   {str(g): int(p["Ton"][g])  for g in p["Ton"]},
        "Ldown": {str(g): int(p["Toff"][g]) for g in p["Toff"]},
        "Pmin":  {str(g): float(p["Pmin"][g]) for g in p.get("Pmin", {})} if "Pmin" in p else {},
        "Pmax":  {str(g): float(p["Pmax"][g]) for g in p.get("Pmax", {})} if "Pmax" in p else {},
        "Demand": [float(sum(p["D"][(t, i)] for i in range(1, num_nodes + 1))) for t in range(1, time_periods + 1)],
        "Reserve": [float(p["Pre"][t]) for t in range(1, time_periods + 1)] if "Pre" in p else []
    }
    with open(os.path.join(output_dir, model_name + ".minud.json"), "w") as jf:
        json.dump(meta, jf, indent=2)
    print(f"Saved metadata to {model_name}.minud.json")
except Exception as e:
    print(f"Warning: Could not save metadata: {e}")

# Create abstract model
model = ucml.opt_model_generator(
    num_therm=num_therm,
    time_periods=time_periods,
    num_nodes=num_nodes,
    num_lines=num_lines,
)

# Create concrete instance with data
instance = model.create_instance(data=p_data)
mps_path = os.path.join(output_dir, model_name + ".mps")

if args.nosolve:
    # Export the concrete instance, not the abstract model
    export_mps_only(instance, mps_path)
    print(f"[nosolve] Exported {mps_path} (and .minud.json if available)")
    print(f"Instance size: {num_therm} generators, {num_nodes} nodes, {num_lines} lines, {time_periods} time periods")
    
    # Quick stats about the instance
    num_vars = len(list(instance.component_data_objects(pyo.Var)))
    num_constraints = len(list(instance.component_data_objects(pyo.Constraint)))
    print(f"Model size: {num_vars} variables, {num_constraints} constraints")
    sys.exit(0)

# If not nosolve, continue with solving
solver = pyo.SolverFactory("gurobi")
solver.options["NonConvex"] = 2
solver.options["MIPGap"] = 0.01  # Set 1% gap for faster solving

print(f"Solving with Gurobi (1% gap)...")
t_start = time.time()
result = solver.solve(instance, tee=args.verbose)
t_solve = time.time() - t_start

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
        print(f"Primal bound: {primal_bound}")
        print(f"Dual bound: {dual_bound}")
        print(f"Gap: {abs(primal_bound - dual_bound) / abs(primal_bound) * 100:.2f}%")
    else:
        primal_bound = pyo.value(instance.obj)
        dual_bound = None

    with open(os.path.join(output_dir, model_name + ".json"), "w") as out:
        out.write(json.dumps({
            "dual_bound": dual_bound, 
            "primal_bound": primal_bound,
            "solve_time": t_solve,
            "gap": 0.01
        }))
    print(f"Optimal value: {pyo.value(instance.obj)}")
    print(f"Solve time: {t_solve:.2f} seconds")
else:
    print("No optimal value found")
    sys.exit(1)
