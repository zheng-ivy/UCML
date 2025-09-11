"""Helper functions for using generated optimization models"""

import pickle
import random
import numpy as np


def parsecase(net, thermal_gens, num_nodes, num_lines, time_periods=24):
    """
    Parses a pandapower network and generates parameters for the Section II UC model.
    """
    p = {}
    G = range(1, len(thermal_gens) + 1)
    T = range(1, time_periods + 1)
    N = range(1, num_nodes + 1)
    L = range(1, num_lines + 1)

    therm_types = ["coal", "ccgt"]
    gen_types = {g: random.choice(therm_types) for g in G}

    p["OpEx"] = {
        g: (
            random_range(38, 42) if gen_types[g] == "coal" else random_range(28.5, 31.5)
        )
        for g in G
    }
    p["Csu"] = {
        g: (
            random_range(95, 105)
            if gen_types[g] == "coal"
            else random_range(47.5, 52.5)
        )
        for g in G
    }
    p["Csd"] = {
        g: (
            random_range(23.75, 26.25)
            if gen_types[g] == "coal"
            else random_range(9.5, 10.5)
        )
        for g in G
    }

    p["Pmax"] = {}
    p["Pmin"] = {}
    for g in G:
        gen_idx = g - 1
        p["Pmax"][g] = thermal_gens.iloc[gen_idx]["max_p_mw"]
        p["Pmin"][g] = thermal_gens.iloc[gen_idx]["min_p_mw"]

    p["Pramp"] = {
        g: (p["Pmax"][g] * 1.0 if gen_types[g] == "coal" else p["Pmax"][g] * 2.85)
        for g in G
    }
    p["Rmax"] = {g: 0.2 * p["Pmax"][g] for g in G}

    total_load = net.load["p_mw"].sum()

    def demand_curve(t, total_load):
        """
        Generates a realistic electricity demand curve for a 24-hour period with hourly timesteps.
        """
        if total_load > 5000:  # Large systems (like 118 bus)
            base_load_factor = 0.4
            volatility = 0.3
        elif total_load > 1000:  # Medium systems (like 30 bus)
            base_load_factor = 0.35
            volatility = 0.4
        else:  # Small systems (like 6 bus)
            base_load_factor = 0.3
            volatility = 0.5
        base_load = total_load * base_load_factor

        peak_scale = volatility
        morning_peak = gaussian_peak(t, mu=9, sigma=1.5, amplitude=0.4 * peak_scale)
        evening_peak = gaussian_peak(t, mu=19, sigma=2.0, amplitude=0.5 * peak_scale)
        midday = gaussian_peak(t, mu=14, sigma=4.0, amplitude=0.3 * peak_scale)
        night_valley = gaussian_valley(t, mu=4, sigma=2.5, amplitude=0.3 * peak_scale)

        daily_pattern = 1.0 + morning_peak + evening_peak + midday - night_valley

        noise_scale = (
            0.02 if total_load <= 1000 else 0.01
        )  # Smaller systems have more noise
        noise = noise_scale * np.random.normal()

        final_demand = base_load * daily_pattern * (1 + noise)

        min_load = base_load * (
            0.5 if total_load > 5000 else 0.4
        )  # Larger systems have higher minimum
        max_load = base_load * (
            1.8 if total_load > 5000 else 2.0
        )  # Larger systems have lower peaks

        return max(min_load, min(final_demand, max_load))

    def gaussian_peak(x, mu, sigma, amplitude):
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gaussian_valley(x, mu, sigma, amplitude):
        return amplitude * (1 - np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    p["D"] = {
        (t, i): demand_curve(t, total_load)
        * (net.load[net.load.bus == (i - 1)]["p_mw"].sum() / total_load)
        for t in T
        for i in N
    }

    def reserve_requirement(t):
        """
        Calculates the reserve requirement for a given time period.
        """
        return 0.05 * sum(p["D"][(t, i)] for i in N)

    p["Pre"] = {t: reserve_requirement(t) for t in T}

    p["Fmax"] = {line: 250 for line in L}

    p["Ton"] = {
        g: (
            random_range(9.5, 10.5)
            if gen_types[g] == "coal"
            else random_range(4.75, 5.25)
        )
        for g in G
    }
    p["Toff"] = p["Ton"].copy()
    p["Psu"] = {
        g: (
            random_range(4.75, 5.25)
            if gen_types[g] == "coal"
            else random_range(0.95, 1.05)
        )
        for g in G
    }
    p["Psd"] = p["Psu"].copy()

    p["a"] = {}
    for g in G:
        gen_idx = g - 1
        bus = thermal_gens.iloc[gen_idx]["bus"] + 1
        for n in N:
            p["a"][(g, n)] = 1 if n == bus else 0

    p["R"] = {}
    for line in L:
        from_bus = net.line.at[line - 1, "from_bus"] + 1
        to_bus = net.line.at[line - 1, "to_bus"] + 1
        for n in N:
            p["R"][(n, line)] = 1 if n == from_bus else (-1 if n == to_bus else 0)

    data = pickle.dumps({None: p})
    with open("UCdata.p", "wb") as f:
        pickle.dump(data, f)
    return data


def random_range(min_val, max_val):
    """
    Returns randomly generated value within [min_val, max_val].
    """
    return min_val + random.random() * (max_val - min_val)
