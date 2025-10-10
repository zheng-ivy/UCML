"""Optimization Model Generator"""

from pyomo.environ import (
    AbstractModel,
    Set,
    RangeSet,
    Constraint,
    Param,
    NonNegativeReals,
    Reals,
    Binary,
    Objective,
    Var,
    minimize,
)


def get_sets(model, num_therm, time_periods=24, num_nodes=0, num_lines=0):
    """
    Initializes sets for the UC model (thermal units only).

    Sets Defined:
        G: Set of thermal generators.
        T: Time periods.
        N: Network nodes.
        L: Transmission lines.
    """
    model.G = Set(initialize=range(1, num_therm + 1))
    model.T = RangeSet(1, time_periods)
    model.N = RangeSet(1, num_nodes)
    model.L = RangeSet(1, num_lines)


def get_parameters(model):
    """
    Initializes the parameters for the Section II UC model (thermal units only).

    Parameters Defined:
        OpEx[g]: Linear operating cost coefficient.
        Csu[g], Csd[g]: Start-up and shut-down cost coefficients.
        Pmin[g], Pmax[g]: Minimum and maximum output.
        Pramp[g]: Ramp limit.
        Psu[g], Psd[g]: Start-up and shut-down power limits.
        Rmax[g]: Unit reserve capacity limit.
        Ton[g], Toff[g]: Minimum up and down times.
        Pre[t]: System reserve requirement.
        R[i, l]: Power Transfer Distribution Factor (PTDF R_{j-l}).
        Fmax[l]: Line flow limit.
        a[g, i]: Generator-to-node association (a_{g-j}).
        D[t, i]: Nodal demand D_{j,t}.
    """
    model.OpEx = Param(model.G)
    model.Csu = Param(model.G)
    model.Csd = Param(model.G)

    model.Pmin = Param(model.G)
    model.Pmax = Param(model.G)

    model.Pramp = Param(model.G)
    model.Psu = Param(model.G)
    model.Psd = Param(model.G)
    model.Rmax = Param(model.G)
    model.Fmax = Param(model.L)
    model.R = Param(model.N, model.L)
    model.D = Param(model.T, model.N, default=0)
    model.Ton = Param(model.G)
    model.Toff = Param(model.G)
    model.Pre = Param(model.T)
    model.a = Param(model.G, model.N, default=0)


def get_variables(model):
    """
    Defines only the UC-relevant decision variables per Section II:
        - u[g,t]: on/off status
        - v[g,t]: start-up indicator
        - w[g,t]: shut-down indicator
        - p[g,t]: generation output
        - r[g,t]: reserve of unit g
        - f[l,t]: line flow
    """
    model.u = Var(model.G, model.T, within=Binary)
    model.v = Var(model.G, model.T, within=Binary)
    model.w = Var(model.G, model.T, within=Binary)
    model.p = Var(model.G, model.T, within=NonNegativeReals)
    model.r = Var(model.G, model.T, within=NonNegativeReals)
    model.f = Var(model.L, model.T, within=Reals)


def get_objective(model):
    """
    Section II objective (1):
    min sum_t sum_g [ OpEx_g * p_{g,t} + CSU_g * v_{g,t} + CSD_g * w_{g,t} ]

    constraints imply p_{g,t} = 0 if u_{g,t} = 0
    """

    def total_cost(model):
        return sum(
            model.OpEx[g] * model.p[g, t]
            + model.Csu[g] * model.v[g, t]
            + model.Csd[g] * model.w[g, t]
            for g in model.G
            for t in model.T
        )

    model.obj = Objective(rule=total_cost, sense=minimize)


def get_uc_constraints(model):
    """
    Implements Section II constraints (2)-(9) of https://ieeexplore.ieee.org/document/10141514.
    """

    # (2) Unit generation limits
    def gen_limits_min(model, g, t):
        return model.Pmin[g] * model.u[g, t] <= model.p[g, t]

    def gen_limits_max(model, g, t):
        return model.p[g, t] <= model.Pmax[g] * model.u[g, t]

    model.gen_min = Constraint(model.G, model.T, rule=gen_limits_min)
    model.gen_max = Constraint(model.G, model.T, rule=gen_limits_max)

    # (3) Unit reserve constraint
    def reserve_headroom(model, g, t):
        return model.p[g, t] + model.r[g, t] <= model.Pmax[g] * model.u[g, t]

    def reserve_cap(model, g, t):
        return model.r[g, t] <= model.Rmax[g] * model.u[g, t]

    model.reserve_headroom = Constraint(model.G, model.T, rule=reserve_headroom)
    model.reserve_cap = Constraint(model.G, model.T, rule=reserve_cap)

    # (4) Ramping with startup/shutdown limits
    def ramp_up(model, g, t):
        if t == model.T.first():
            return Constraint.Skip
        return (
            model.p[g, t] - model.p[g, t - 1]
            <= model.Pramp[g] * model.u[g, t - 1] + model.Psu[g] * model.v[g, t]
        )

    def ramp_down(model, g, t):
        if t == model.T.first():
            return Constraint.Skip
        return (
            model.p[g, t - 1] - model.p[g, t]
            <= model.Pramp[g] * model.u[g, t] + model.Psd[g] * model.w[g, t]
        )

    model.ramp_up = Constraint(model.G, model.T, rule=ramp_up)
    model.ramp_down = Constraint(model.G, model.T, rule=ramp_down)

    # (5) Minimum up/down time
    def min_up_time(model, g, t):
        if t <= model.Ton[g]:
            return Constraint.Skip
        return (
            sum(model.v[g, tau] for tau in range(t - int(model.Ton[g]) + 1, t + 1))
            <= model.u[g, t]
        )

    def min_down_time(model, g, t):
        if t <= model.Toff[g]:
            return Constraint.Skip
        return (
            sum(model.w[g, tau] for tau in range(t - int(model.Toff[g]) + 1, t + 1))
            <= 1 - model.u[g, t]
        )

    model.min_up_time = Constraint(model.G, model.T, rule=min_up_time)
    model.min_down_time = Constraint(model.G, model.T, rule=min_down_time)

    # (6) Logical constraints
    def logic_balance(model, g, t):
        if t == model.T.first():
            return Constraint.Skip
        return model.u[g, t] - model.u[g, t - 1] == model.v[g, t] - model.w[g, t]

    def logic_mutex(model, g, t):
        return model.v[g, t] + model.w[g, t] <= 1

    model.logic_balance = Constraint(model.G, model.T, rule=logic_balance)
    model.logic_mutex = Constraint(model.G, model.T, rule=logic_mutex)

    # (7) Power balance across system
    def power_balance(model, t):
        return sum(model.p[g, t] for g in model.G) == sum(
            model.D[t, i] for i in model.N
        )

    model.power_balance = Constraint(model.T, rule=power_balance)

    # (8) System reserve requirement
    def system_reserve(model, t):
        return sum(model.r[g, t] for g in model.G) >= model.Pre[t]

    model.system_reserve = Constraint(model.T, rule=system_reserve)

    # (9) Line flows with PTDF and limits
    def line_flow(model, line, t):
        return model.f[line, t] == sum(
            model.R[i, line] *
            (sum(model.a[g, i] * model.p[g, t] for g in model.G) - model.D[t, i])
            for i in model.N
        )

    # def line_flow(model, line, t):
    #     nodal_injection = sum(
    #         model.R[i, line]
    #         * (sum(model.a[g, i] * model.p[g, t] for g in model.G) - model.D[t, i])
    #         for i in model.N
    #     )
    #     return model.f[line, t] == nodal_injection

    def line_min(model, line, t):
        return -model.Fmax[line] <= model.f[line, t]

    def line_max(model, line, t):
        return model.f[line, t] <= model.Fmax[line]

    model.line_flow = Constraint(model.L, model.T, rule=line_flow)
    model.line_min = Constraint(model.L, model.T, rule=line_min)
    model.line_max = Constraint(model.L, model.T, rule=line_max)


def opt_model_generator(num_therm=0, time_periods=24, num_nodes=0, num_lines=0):
    """
    Generates the complete optimization model by initializing sets, parameters, variables, constraints, and the objective function.
    """
    model_name = "Unit Commitment Model"
    model = AbstractModel(model_name)

    get_sets(model, num_therm, time_periods, num_nodes, num_lines)
    get_parameters(model)
    get_variables(model)
    get_uc_constraints(model)
    get_objective(model)

    return model
