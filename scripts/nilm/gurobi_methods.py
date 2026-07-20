"""
Gurobi MIQP solvers for NILM disaggregation.

Two formulations, both with 3 power levels per device, a per-activation penalty
and a soft time-window penalty:

  solve_activation — event devices only; always-on load must be removed from the
                     signal beforehand (see baseline_load).
  solve_full       — always-on devices modeled as variables (exactly one power
                     level active at every timestep), no baseline subtraction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.time_windows import (
    WINDOW_PENALTY_MAX_FACTOR,
    WINDOW_PENALTY_RAMP_MIN,
    out_of_season_mask,
    window_penalty_factors,
)

logger = logging.getLogger(__name__)


def solve_activation(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Disaggregate event devices from a baseline-subtracted signal.

    Each device gets 3 power levels (p_typical ± power_level_variation) and a
    binary ON/OFF aggregate x[i,t].  The objective minimises the squared
    reconstruction error plus two penalties, both scaled by p_typical² so they
    are commensurate with that error:

      - per activation: ``activation_penalty * p_typical² * u[i,t]`` on every
        rising edge, which discourages rapid ON/OFF cycling.  At 1.0 one extra
        activation costs as much as one slot ON against zero signal.
      - out of time window: ``time_window_penalty * p_typical² * f(t) * x[i,t]``,
        where f(t) grows with the distance from the nearest allowed window (1.0
        on the edge, +1 per window_penalty_ramp_min minutes, capped) so that
        slots far from the declared window are progressively more expensive.

    Min ON duration and max consecutive ON stay hard constraints.

    Always-on load is expected to be already removed from ``signal``; use
    solve_full to model always-on devices as variables instead.

    Args:
        signal: Aggregate power series (any regular DatetimeIndex frequency).
        devices: Device profiles to disaggregate.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Maps device name to minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Maps device name to maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Maps device name to allowed daily windows as
            list of (start_min, end_min) in minutes-of-day (soft penalty).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor for per-activation penalty (× p_typical²).

    Returns:
        (disaggregation, info) where disaggregation maps device name to power series.
    """
    if not _HAS_GUROBI:
        raise ImportError("gurobipy is not installed.")

    T = len(signal)
    n = len(devices)
    K = 3
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}
    empty_info: dict[str, float | int | bool | None] = {
        "status": None, "has_solution": False, "is_optimal": False,
        "objective_value": None, "runtime_sec": 0.0, "mip_gap": None,
        "n_devices": n, "n_timesteps": T,
    }

    if T == 0 or n == 0:
        return zero_output, empty_info

    valid = signal.notna().to_numpy()
    y = signal.to_numpy(dtype=float)
    y_safe = np.where(valid, y, 0.0)

    power_levels: list[list[float]] = [
        [
            dev.p_typical_w * (1.0 - power_level_variation),
            dev.p_typical_w,
            dev.p_typical_w * (1.0 + power_level_variation),
        ]
        for dev in devices
    ]

    min_on_slots: dict[int, int] = {}
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on_slots[i] = int(a)

    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            lim = max_consecutive_on_by_device.get(dev.name)
            if lim is not None and int(lim) >= 1:
                max_consecutive_limits[i] = int(lim)

    out_of_window_factors: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                out_of_window_factors[i] = window_penalty_factors(
                    signal.index,
                    windows,
                    ramp_min=window_penalty_ramp_min,
                    max_factor=window_penalty_max_factor,
                )

    try:
        model = gp.Model("nilm_activation")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        z = model.addVars(n, T, K, vtype=GRB.BINARY, name="z")
        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        for i in range(n):
            for t in range(T):
                model.addConstr(
                    x[i, t] == gp.quicksum(z[i, t, k] for k in range(K)),
                    name=f"link_{i}_{t}",
                )

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0
                    for k in range(K):
                        z[i, t, k].UB = 0.0

        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        for i, a_i in min_on_slots.items():
            for t in range(T - a_i + 1):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + a_i)) >= a_i * u[i, t],
                    name=f"mindur_{i}_{t}",
                )

        for i, b_i in max_consecutive_limits.items():
            if b_i >= T:
                continue
            for t in range(T - b_i):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + b_i + 1)) <= b_i,
                    name=f"maxdur_{i}_{t}",
                )

        objective = gp.QuadExpr()
        for t in range(T):
            if not valid[t]:
                continue
            power_sum = gp.quicksum(
                power_levels[i][k] * z[i, t, k]
                for i in range(n)
                for k in range(K)
            )
            residual = power_sum - y_safe[t]
            objective += residual * residual

        # Soft time-window penalty (scaled by p_typical², grows with distance)
        for i, factors in out_of_window_factors.items():
            p_sq = devices[i].p_typical_w ** 2
            for t in range(T):
                if factors[t] > 0.0 and valid[t]:
                    objective += (time_window_penalty * p_sq * factors[t]) * x[i, t]

        # Activation penalty: discourage rapid cycling (scaled by p_typical²)
        for i, dev in enumerate(devices):
            p_sq = dev.p_typical_w ** 2
            for t in range(T):
                if valid[t]:
                    objective += (activation_penalty * p_sq) * u[i, t]

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        has_solution = model.SolCount > 0
        info: dict[str, float | int | bool | None] = {
            "status": int(model.Status),
            "has_solution": has_solution,
            "is_optimal": model.Status == GRB.OPTIMAL,
            "objective_value": float(model.ObjVal) if has_solution else None,
            "runtime_sec": float(model.Runtime),
            "mip_gap": float(model.MIPGap) if has_solution else None,
            "n_devices": n,
            "n_timesteps": T,
        }

        if not has_solution:
            return zero_output, info

        disaggregation: dict[str, pd.Series] = {}
        for i, dev in enumerate(devices):
            power_sol = np.array(
                [
                    sum(power_levels[i][k] * round(z[i, t, k].X) for k in range(K))
                    for t in range(T)
                ],
                dtype=float,
            )
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)
        return disaggregation, info

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during solve_activation: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": n, "n_timesteps": T,
        }


def solve_full(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
    duration_penalty_block: float = 0.0,
    duration_penalty_daily: float = 0.0,
    expected_daily_on_slots_by_device: dict[str, float] | None = None,
    over_activation_penalty_by_device: dict[str, float] | None = None,
    remaining_activations_by_device: dict[str, float] | None = None,
    window_vagueness_weight: float = 0.0,
    season_penalty: float = 0.0,
    active_months_by_device: dict[str, list[int]] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Disaggregate all devices from the raw signal, always-on ones included.

    Unlike solve_activation, always-on devices are not estimated upfront and
    subtracted: they enter the model as variables, so the solver jointly
    optimises always-on level selection and event-device activation.  All
    devices receive 3 power levels (p_typical ± power_level_variation).

    Always-on devices (dev.always_on = True):
      - Exactly one power level active at every valid timestep: Σ_k z[i,t,k] = 1
      - No activation/deactivation variables (u, dw) — they never turn off
      - No time-window, min-ON, or max-consecutive constraints
      - Their contribution appears directly in the reconstruction error

    Event devices (dev.always_on = False):
      - Same as solve_activation: binary ON/OFF aggregate, 3 levels, activation
        penalty, soft time-window penalty, min-ON and max-consecutive constraints

    Args:
        signal: Aggregate power series on the full signal (not residual).
        devices: All device profiles, including always-on ones.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Minimum consecutive ON timesteps (event devices only).
        max_consecutive_on_by_device: Maximum consecutive ON timesteps (event devices only).
        allowed_on_windows_by_device: Soft time-window penalty windows (event devices only).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor for per-activation penalty (× p_typical²).
        duration_penalty_block: If > 0, the min-ON and max-consecutive limits become
            soft: a block may be shorter or longer than its bounds, paying
            ``duration_penalty_block * p_typical² * (slots of deviation)``.  At 0
            (default) both stay hard constraints.
        duration_penalty_daily: If > 0, penalise the deviation of a device's total
            daily ON time from expected_daily_on_slots_by_device, at
            ``duration_penalty_daily * p_typical² * |slots deviation|``.
        expected_daily_on_slots_by_device: Expected ON slots per day per device,
            required for duration_penalty_daily to have any effect.
        over_activation_penalty_by_device: Extra cost (× p_typical²) charged on every
            activation past remaining_activations_by_device.  Per device because the
            break-even scales with the device's typical ON length in slots: explaining
            one activation gains (slots × p²), so the surcharge must exceed that.  The
            caller carries the weekly counter across days, since each day is its own model.
        remaining_activations_by_device: Activations still available this week for each
            device on entry to this day.  0 means the weekly quota is already spent, so
            every activation today pays the surcharge.
        window_vagueness_weight: Weight on the vagueness floor of the time-window
            penalty.  Above 0, a device that declared a narrow window pays less than
            one that declared none, so the specific device wins a tie instead of the
            undeclared one being free everywhere.
        season_penalty: Cost (× p_typical²) per ON slot falling in a month the device
            was not declared active.  A high value effectively removes the device out
            of season while still letting overwhelming evidence override it.
        active_months_by_device: Months (1-12) each device was declared active in.

    Returns:
        (disaggregation, info) where disaggregation maps every device name to a power series.
    """
    if not _HAS_GUROBI:
        raise ImportError("gurobipy is not installed.")

    T = len(signal)
    K = 3
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}
    empty_info: dict[str, float | int | bool | None] = {
        "status": None, "has_solution": False, "is_optimal": False,
        "objective_value": None, "runtime_sec": 0.0, "mip_gap": None,
        "n_devices": len(devices), "n_timesteps": T,
    }

    if T == 0 or not devices:
        return zero_output, empty_info

    valid = signal.notna().to_numpy()
    y = signal.to_numpy(dtype=float)
    y_safe = np.where(valid, y, 0.0)

    # Split devices by type
    ao_devices = [(i, d) for i, d in enumerate(devices) if d.always_on]
    ev_devices = [(i, d) for i, d in enumerate(devices) if not d.always_on]
    n_ao = len(ao_devices)
    n_ev = len(ev_devices)

    # Power levels for every device
    def _levels(dev: DeviceProfile) -> list[float]:
        return [
            dev.p_typical_w * (1.0 - power_level_variation),
            dev.p_typical_w,
            dev.p_typical_w * (1.0 + power_level_variation),
        ]

    ao_levels = [_levels(d) for _, d in ao_devices]
    ev_levels = [_levels(d) for _, d in ev_devices]

    # Pre-process event-device constraint dicts (indexed by position in ev_devices)
    min_on_slots: dict[int, int] = {}
    if min_on_slots_by_device:
        for j, (_, dev) in enumerate(ev_devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on_slots[j] = int(a)

    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for j, (_, dev) in enumerate(ev_devices):
            lim = max_consecutive_on_by_device.get(dev.name)
            if lim is not None and int(lim) >= 1:
                max_consecutive_limits[j] = int(lim)

    out_of_window_factors: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device or window_vagueness_weight > 0.0:
        windows_by_device = allowed_on_windows_by_device or {}
        for j, (_, dev) in enumerate(ev_devices):
            windows = windows_by_device.get(dev.name)
            # With a vagueness floor every device is charged, including the ones that
            # declared nothing — that is precisely what the floor is for.
            if not windows and window_vagueness_weight <= 0.0:
                continue
            factors = window_penalty_factors(
                signal.index,
                windows,
                ramp_min=window_penalty_ramp_min,
                max_factor=window_penalty_max_factor,
                vagueness_weight=window_vagueness_weight,
            )
            if factors.any():
                out_of_window_factors[j] = factors

    out_of_season: dict[int, np.ndarray] = {}
    if season_penalty > 0.0 and active_months_by_device:
        for j, (_, dev) in enumerate(ev_devices):
            mask = out_of_season_mask(signal.index, active_months_by_device.get(dev.name))
            if mask.any():
                out_of_season[j] = mask

    try:
        model = gp.Model("nilm_full")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        # ---- Always-on variables ----------------------------------------- #
        # z_ao[j, t, k]: level k of always-on device j at timestep t
        # Σ_k z_ao[j,t,k] = 1 for all valid t  (exactly one level, always active)
        z_ao = model.addVars(n_ao, T, K, vtype=GRB.BINARY, name="z_ao") if n_ao else {}

        for j in range(n_ao):
            for t in range(T):
                if not valid[t]:
                    for k in range(K):
                        z_ao[j, t, k].UB = 0.0
                else:
                    model.addConstr(
                        gp.quicksum(z_ao[j, t, k] for k in range(K)) == 1,
                        name=f"ao_on_{j}_{t}",
                    )

        # ---- Event variables (as in solve_activation) -------------------- #
        z_ev = model.addVars(n_ev, T, K, vtype=GRB.BINARY, name="z_ev") if n_ev else {}
        x_ev = model.addVars(n_ev, T, vtype=GRB.BINARY, name="x_ev") if n_ev else {}
        u_ev = model.addVars(n_ev, T, vtype=GRB.BINARY, name="u_ev") if n_ev else {}
        dw_ev = model.addVars(n_ev, T, vtype=GRB.BINARY, name="dw_ev") if n_ev else {}

        for j in range(n_ev):
            for t in range(T):
                model.addConstr(
                    x_ev[j, t] == gp.quicksum(z_ev[j, t, k] for k in range(K)),
                    name=f"ev_link_{j}_{t}",
                )

        for t in range(T):
            if not valid[t]:
                for j in range(n_ev):
                    x_ev[j, t].UB = 0.0
                    u_ev[j, t].UB = 0.0
                    dw_ev[j, t].UB = 0.0
                    for k in range(K):
                        z_ev[j, t, k].UB = 0.0

        for j in range(n_ev):
            model.addConstr(u_ev[j, 0] == x_ev[j, 0], name=f"ev_init_u_{j}")
            model.addConstr(dw_ev[j, 0] == 0, name=f"ev_init_dw_{j}")
            for t in range(1, T):
                model.addConstr(
                    x_ev[j, t] - x_ev[j, t - 1] == u_ev[j, t] - dw_ev[j, t],
                    name=f"ev_trans_{j}_{t}",
                )
                model.addConstr(u_ev[j, t] + dw_ev[j, t] <= 1, name=f"ev_mutex_{j}_{t}")

        # Duration limits.  With duration_penalty_block > 0 they are relaxed with a
        # non-negative slack that is charged in the objective; a run of length L
        # exceeding b by k slots makes exactly k of these windows infeasible by 1,
        # so the total charge is linear in the overshoot (and likewise for
        # shortfalls below a).
        soft_duration = duration_penalty_block > 0.0
        s_short = {}
        s_over = {}

        for j, a_j in min_on_slots.items():
            for t in range(T - a_j + 1):
                covered = gp.quicksum(x_ev[j, tau] for tau in range(t, t + a_j))
                if soft_duration:
                    slack = model.addVar(lb=0.0, ub=float(a_j), name=f"ev_shortdur_{j}_{t}")
                    s_short[j, t] = slack
                    model.addConstr(covered + slack >= a_j * u_ev[j, t], name=f"ev_mindur_{j}_{t}")
                else:
                    model.addConstr(covered >= a_j * u_ev[j, t], name=f"ev_mindur_{j}_{t}")

        for j, b_j in max_consecutive_limits.items():
            if b_j >= T:
                continue
            for t in range(T - b_j):
                covered = gp.quicksum(x_ev[j, tau] for tau in range(t, t + b_j + 1))
                if soft_duration:
                    slack = model.addVar(lb=0.0, ub=float(T), name=f"ev_overdur_{j}_{t}")
                    s_over[j, t] = slack
                    model.addConstr(covered - slack <= b_j, name=f"ev_maxdur_{j}_{t}")
                else:
                    model.addConstr(covered <= b_j, name=f"ev_maxdur_{j}_{t}")

        # Daily ON-time budget: |Σ_t x[i,t] − expected| split into two non-negative
        # deviations, charged symmetrically.
        daily_dev = {}
        if duration_penalty_daily > 0.0 and expected_daily_on_slots_by_device:
            for j, (_, dev) in enumerate(ev_devices):
                expected = expected_daily_on_slots_by_device.get(dev.name)
                if expected is None or expected < 0:
                    continue
                over = model.addVar(lb=0.0, ub=float(T), name=f"ev_dayover_{j}")
                under = model.addVar(lb=0.0, ub=float(T), name=f"ev_dayunder_{j}")
                daily_dev[j] = (over, under)
                model.addConstr(
                    gp.quicksum(x_ev[j, t] for t in range(T)) - float(expected) == over - under,
                    name=f"ev_daybudget_{j}",
                )

        # Weekly activation quota: the first r_i activations of the week cost only
        # activation_penalty; each one past it also pays over_activation_penalty.
        # r_i arrives as a constant because days are solved independently.
        excess_act = {}
        if over_activation_penalty_by_device and remaining_activations_by_device:
            for j, (_, dev) in enumerate(ev_devices):
                remaining = remaining_activations_by_device.get(dev.name)
                penalty = over_activation_penalty_by_device.get(dev.name)
                if remaining is None or not penalty or penalty <= 0:
                    continue
                excess = model.addVar(lb=0.0, ub=float(T), name=f"ev_overact_{j}")
                excess_act[j] = excess
                model.addConstr(
                    gp.quicksum(u_ev[j, t] for t in range(T)) - excess
                    <= max(0.0, float(remaining)),
                    name=f"ev_quota_{j}",
                )

        # ---- Objective --------------------------------------------------- #
        objective = gp.QuadExpr()
        for t in range(T):
            if not valid[t]:
                continue
            power_sum = gp.LinExpr()
            for j in range(n_ao):
                for k in range(K):
                    power_sum += ao_levels[j][k] * z_ao[j, t, k]
            for j in range(n_ev):
                for k in range(K):
                    power_sum += ev_levels[j][k] * z_ev[j, t, k]
            residual = power_sum - y_safe[t]
            objective += residual * residual

        # Soft time-window penalty (event devices only, grows with distance)
        for j, factors in out_of_window_factors.items():
            p_sq = ev_devices[j][1].p_typical_w ** 2
            for t in range(T):
                if factors[t] > 0.0 and valid[t]:
                    objective += (time_window_penalty * p_sq * factors[t]) * x_ev[j, t]

        # Out-of-season penalty (event devices only)
        for j, mask in out_of_season.items():
            p_sq = ev_devices[j][1].p_typical_w ** 2
            for t in range(T):
                if mask[t] and valid[t]:
                    objective += (season_penalty * p_sq) * x_ev[j, t]

        # Activation penalty (event devices only)
        for j, (_, dev) in enumerate(ev_devices):
            p_sq = dev.p_typical_w ** 2
            for t in range(T):
                if valid[t]:
                    objective += (activation_penalty * p_sq) * u_ev[j, t]

        # Duration deviation penalties (event devices only)
        for (j, t), slack in s_short.items():
            objective += (duration_penalty_block * ev_devices[j][1].p_typical_w ** 2) * slack
        for (j, t), slack in s_over.items():
            objective += (duration_penalty_block * ev_devices[j][1].p_typical_w ** 2) * slack
        for j, (over, under) in daily_dev.items():
            scale = duration_penalty_daily * ev_devices[j][1].p_typical_w ** 2
            objective += scale * over + scale * under

        # Activations past the weekly quota
        for j, excess in excess_act.items():
            dev = ev_devices[j][1]
            lam = over_activation_penalty_by_device[dev.name]
            objective += (lam * dev.p_typical_w ** 2) * excess

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        has_solution = model.SolCount > 0
        info: dict[str, float | int | bool | None] = {
            "status": int(model.Status),
            "has_solution": has_solution,
            "is_optimal": model.Status == GRB.OPTIMAL,
            "objective_value": float(model.ObjVal) if has_solution else None,
            "runtime_sec": float(model.Runtime),
            "mip_gap": float(model.MIPGap) if has_solution else None,
            "n_devices": len(devices),
            "n_timesteps": T,
        }

        if not has_solution:
            return zero_output, info

        disaggregation: dict[str, pd.Series] = {}
        for j, (orig_i, dev) in enumerate(ao_devices):
            power_sol = np.array(
                [sum(ao_levels[j][k] * round(z_ao[j, t, k].X) for k in range(K)) for t in range(T)],
                dtype=float,
            )
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

        for j, (orig_i, dev) in enumerate(ev_devices):
            power_sol = np.array(
                [sum(ev_levels[j][k] * round(z_ev[j, t, k].X) for k in range(K)) for t in range(T)],
                dtype=float,
            )
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

        return disaggregation, info

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during solve_full: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": len(devices), "n_timesteps": T,
        }

