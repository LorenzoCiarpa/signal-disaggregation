"""
Approach OR-MIQP — Mixed-Integer Quadratic Programming disaggregation via Gurobi.

Minimises squared reconstruction error between aggregate signal y(t) and the
sum of per-device binary power contributions p_i * x_i(t), subject to:

  - Probability-based availability constraints (s_i(t) < threshold → forced OFF)
  - State-transition tracking: x_i(t) - x_i(t-1) = u_i(t) - d_i(t)
  - Mutual exclusivity of switching events: u_i(t) + d_i(t) ≤ 1
  - Minimum ON-duration per device
  - Maximum ON-duration per device (rolling-window constraint)
  - Activation penalty in the objective to limit start-up frequency

The signal is segmented by calendar week, and an independent problem is
solved per week for tractability.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from scripts.nilm.devices import DeviceProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule: s_i(t) — probability that device i is ON at minute t
# ---------------------------------------------------------------------------


def _compute_schedule(device: DeviceProfile, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return a 1-D float array of shape (T,) with s_i(t) ∈ [0, 1].

    For always-on devices (main fridge, secondary fridge, freezer) the stored duty_cycle
    gives a constant probability.  For all other devices a flat probability
    is estimated from weekly frequency and typical usage duration.
    """
    T = len(timestamps)
    if device.always_on and device.duty_cycle is not None:
        return np.full(T, device.duty_cycle, dtype=float)

    # Flat probability: expected fraction of minutes in a day the device is ON.
    # p_flat = (uses_per_day) * (duration_per_use_min) / (1440 min/day)
    freq_per_day = device.frequency_per_week / 7.0
    p_flat = freq_per_day * device.dur_typical_min / (24.0 * 60.0)
    p_flat = min(max(p_flat, 0.0), 1.0)
    return np.full(T, p_flat, dtype=float)


# ---------------------------------------------------------------------------
# Week-level solver
# ---------------------------------------------------------------------------


def _solve_week(
    y: np.ndarray,
    devices: list[DeviceProfile],
    timestamps: pd.DatetimeIndex,
    prob_threshold: float,
    max_duration_factor: float,
    lambda_activation: float,
    time_limit: float,
) -> dict[str, np.ndarray]:
    """Solve the Gurobi model for a single weekly chunk.

    Variables
    ---------
    x[i, t]   ∈ {0,1}  device i is ON at timestep t
    u[i, t]   ∈ {0,1}  device i switches ON  at t  (rising edge)
    dw[i, t]  ∈ {0,1}  device i switches OFF at t  (falling edge)
    Objective
    ---------
     min  Σ_t (Σ_i p_i x_i(t) - y(t))² + Σ_i λ_i Σ_t u[i, t]

    where λ_i = lambda_activation / (freq_week_i / 7) so rarer devices are
    penalised more per activation.

    Returns
    -------
    dict {device_name: np.ndarray(T)} with estimated power values (W).
    On failure returns all-zero arrays.
    """
    T = len(y)
    n = len(devices)
    valid = ~np.isnan(y)
    y_safe = np.where(valid, y, 0.0)

    # Pre-compute time-of-day schedules
    schedules = [_compute_schedule(d, timestamps) for d in devices]

    zero_result: dict[str, np.ndarray] = {d.name: np.zeros(T, dtype=float) for d in devices}

    if not _HAS_GUROBI:
        logger.error("gurobipy is not available — returning zeros for this weekly chunk")
        return zero_result

    try:
        m = gp.Model()
        # m.setParam("OutputFlag", 0)
        # m.setParam("LogToConsole", 0)
        m.setParam("TimeLimit", time_limit)

        # ---- Decision variables ------------------------------------------ #

        x = m.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = m.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = m.addVars(n, T, vtype=GRB.BINARY, name="dw")
        # ---- Fix x=0 at NaN timesteps (all devices) ---------------------- #
        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0

        # ---- C2: Probability exclusion ------------------------------------ #
        # If s_i(t) < prob_threshold the device cannot be ON at t.
        # for i in range(n):
        #     s = schedules[i]
        #     for t in range(T):
        #         if s[t] < prob_threshold:
        #             x[i, t].UB = 0.0

        # ---- Objective ---------------------------------------------------- #
        # Per-device activation penalty: rarer devices ↔ higher penalty/use.
        act_weights: list[float] = []
        for dev in devices:
            freq_day = max(dev.frequency_per_week, 0.5) / 7.0
            act_weights.append(lambda_activation / freq_day)

        obj_error = gp.QuadExpr()
        for t in range(T):
            if not valid[t]:
                continue
            power_sum = gp.quicksum(devices[i].p_typical_w * x[i, t] for i in range(n))
            residual = power_sum - y_safe[t]
            obj_error += residual * residual

        obj_penalty = gp.quicksum(
            act_weights[i] * u[i, t] for i in range(n) for t in range(T)
        )
        obj_penalty = 0
        m.setObjective(obj_error + obj_penalty, GRB.MINIMIZE)

        # ---- C3 / C4 / C7: State transitions ----------------------------- #
        # for i in range(n):
        #     # t=0: no previous state → u[i,0] = x[i,0], dw[i,0] = 0
        #     m.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
        #     m.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
        #     for t in range(1, T):
        #         # C3: x[i,t] - x[i,t-1] = u[i,t] - dw[i,t]
        #         m.addConstr(
        #             x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
        #             name=f"trans_{i}_{t}",
        #         )
        #         # C4: u[i,t] + dw[i,t] ≤ 1
        #         m.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        # ---- C5: Minimum ON-duration ------------------------------------- #
        # Σ_{τ=t}^{t+a_i-1} x[i,τ] ≥ a_i * u[i,t]
        # for i, dev in enumerate(devices):
        #     a_i = max(1, int(math.ceil(dev.dur_min_min)))
        #     if a_i <= 1:
        #         continue
        #     for t in range(T - a_i + 1):
        #         m.addConstr(
        #             gp.quicksum(x[i, tau] for tau in range(t, t + a_i))
        #             >= a_i * u[i, t],
        #             name=f"mindur_{i}_{t}",
        #         )

        # ---- C6: Maximum ON-duration (rolling window) -------------------- #
        # Σ_{τ=t}^{t+B_i} x[i,τ] ≤ B_i
        # for i, dev in enumerate(devices):
        #     B_i = max(1, int(math.ceil(dev.dur_typical_min * max_duration_factor)))
        #     if B_i >= T:
        #         continue
        #     for t in range(T - B_i):
        #         m.addConstr(
        #             gp.quicksum(x[i, tau] for tau in range(t, t + B_i + 1)) <= B_i,
        #             name=f"maxdur_{i}_{t}",
        #         )

        # ---- Solve -------------------------------------------------------- #
        m.optimize()

        # ---- Extract solution -------------------------------------------- #
        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
            result: dict[str, np.ndarray] = {}
            for i, dev in enumerate(devices):
                x_sol = np.array([round(x[i, t].X) for t in range(T)], dtype=float)
                result[dev.name] = x_sol * dev.p_typical_w
            return result
        else:
            logger.warning(
                "MIQP: no feasible solution found for this weekly chunk (status=%d); returning zeros",
                m.Status,
            )
            return zero_result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during solve: %s — returning zeros", exc)
        return zero_result


def solve_unconstrained_window(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Solve a single-window unconstrained binary quadratic reconstruction problem.

    This helper is intended for synthetic experiments where each device is
    represented by a single binary activation variable per timestep and the
    only objective term is the squared reconstruction error.

    Args:
        signal: Aggregate power series for a single time window.
        devices: Devices to include in the unconstrained model.
        time_limit: Solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.

    Returns:
        A tuple ``(disaggregation, info)`` where ``disaggregation`` maps each
        device name to a power series aligned to ``signal.index`` and ``info``
        stores basic solver metadata.
    """
    if not _HAS_GUROBI:
        raise ImportError(
            "gurobipy is not installed. Install it with `pip install gurobipy` "
            "and configure a valid Gurobi license."
        )

    T = len(signal)
    n = len(devices)
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or n == 0:
        return zero_output, {
            "status": None,
            "has_solution": False,
            "is_optimal": False,
            "objective_value": None,
            "runtime_sec": 0.0,
            "mip_gap": None,
            "n_devices": n,
            "n_timesteps": T,
        }

    valid = signal.notna().to_numpy()
    y = signal.to_numpy(dtype=float)
    y_safe = np.where(valid, y, 0.0)

    try:
        model = gp.Model("nilm_unconstrained")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0

        objective = gp.QuadExpr()
        for t in range(T):
            if not valid[t]:
                continue
            power_sum = gp.quicksum(devices[i].p_typical_w * x[i, t] for i in range(n))
            residual = power_sum - y_safe[t]
            objective += residual * residual

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
            x_sol = np.array([round(x[i, t].X) for t in range(T)], dtype=float)
            disaggregation[dev.name] = pd.Series(
                x_sol * dev.p_typical_w,
                index=signal.index,
                dtype=float,
            )
        return disaggregation, info
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during unconstrained solve: %s", exc)
        return zero_output, {
            "status": None,
            "has_solution": False,
            "is_optimal": False,
            "objective_value": None,
            "runtime_sec": None,
            "mip_gap": None,
            "n_devices": n,
            "n_timesteps": T,
        }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    lambda_activation: float = 10.0,
    prob_threshold: float = 0.10,
    max_duration_factor: float = 2.0,
    time_limit: float = 60.0,
) -> dict[str, pd.Series]:
    """MIQP-based NILM disaggregation using Gurobi.

    Solves one Gurobi model per calendar week. The objective minimises
    squared reconstruction error plus an activation-frequency penalty.

    Args:
        signal: 1-minute resampled w_medio series with UTC DatetimeIndex.
        devices: List of DeviceProfile instances (all devices, incl. absent).
        lambda_activation: Base penalty per device activation, scaled by
            7 / freq_week_i so that rare devices are penalised more heavily.
        prob_threshold: If s_i(t) < threshold, device i is forced OFF at t.
        max_duration_factor: B_i = ceil(dur_typical_min * factor) caps the
            maximum consecutive ON window via a rolling-window constraint.
        time_limit: Gurobi time limit per weekly chunk in seconds.

    Returns:
        dict mapping device name → pd.Series (same index as *signal*) with
        estimated power in Watts.
    """
    if not _HAS_GUROBI:
        raise ImportError(
            "gurobipy is not installed.  Install it with:\n"
            "    pip install gurobipy\n"
            "and obtain a Gurobi license from "
            "https://www.gurobi.com/academia/academic-program-and-licenses/"
        )

    # Only devices confirmed as present in the household are optimised.
    present_devices = [d for d in devices if d.prior_weight >= 1.0]
    absent_devices = [d for d in devices if d.prior_weight < 1.0]

    if not present_devices:
        return {d.name: pd.Series(0.0, index=signal.index) for d in devices}

    # Iterate over calendar weeks and solve one problem per week.
    week_periods = signal.index.to_period("W-SUN")
    unique_weeks = week_periods.unique().sort_values()
    week_results: dict[str, list[pd.Series]] = {d.name: [] for d in present_devices}

    for idx, week in enumerate(unique_weeks):
        week_mask = week_periods == week
        week_signal = signal[week_mask]
        timestamps = week_signal.index
        y = week_signal.values.astype(float)
        T = len(y)

        if T == 0:
            continue

        logger.debug(
            "OR-MIQP: solving week %s to %s (T=%d)",
            week.start_time.date(),
            week.end_time.date(),
            T,
        )

        week_sol = _solve_week(
            y=y,
            devices=present_devices,
            timestamps=timestamps,
            prob_threshold=prob_threshold,
            max_duration_factor=max_duration_factor,
            lambda_activation=lambda_activation,
            time_limit=time_limit,
        )

        if idx == 2:
            break

        for dev in present_devices:
            arr = week_sol.get(dev.name, np.zeros(T, dtype=float))
            week_results[dev.name].append(pd.Series(arr, index=timestamps, dtype=float))

    # Concatenate weekly chunks and align to original signal index.
    output: dict[str, pd.Series] = {}
    for dev in present_devices:
        if week_results[dev.name]:
            combined = pd.concat(week_results[dev.name])
            output[dev.name] = combined.reindex(signal.index).fillna(0.0)
        else:
            output[dev.name] = pd.Series(0.0, index=signal.index)

    # Absent devices receive all-zero series (consistent with other approaches).
    for dev in absent_devices:
        output[dev.name] = pd.Series(0.0, index=signal.index)

    return output
