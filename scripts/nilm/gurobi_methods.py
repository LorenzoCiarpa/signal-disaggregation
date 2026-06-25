
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


# constrained_v1 adds the first explicit max-consecutive-ON constraint to the daily solver.
def constrained_v1(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    max_consecutive_on_by_device: dict[str, int] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Solve a single-window constrained binary quadratic reconstruction problem.

    v1 introduces explicit transition variables ``u`` and ``dw`` together with
    per-device maximum consecutive ON limits.

    Args:
        signal: Aggregate power series for a single time window.
        devices: Devices to include in the constrained model.
        time_limit: Solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        max_consecutive_on_by_device: Optional mapping of device name to the
            maximum number of consecutive ON minutes allowed in the window.

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
    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            limit = max_consecutive_on_by_device.get(dev.name)
            if limit is None:
                continue
            limit_int = int(limit)
            if limit_int < 1:
                raise ValueError(
                    f"max_consecutive_on_by_device[{dev.name!r}] must be >= 1"
                )
            max_consecutive_limits[i] = limit_int

    try:
        model = gp.Model("nilm_constrained_v1")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0

        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        for i, max_consecutive in max_consecutive_limits.items():
            if max_consecutive >= T:
                continue
            for t in range(T - max_consecutive):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + max_consecutive + 1))
                    <= max_consecutive,
                    name=f"max_consecutive_{i}_{t}",
                )

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
        logger.warning("Gurobi error during constrained_v1 solve: %s", exc)
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


def constrained_v3(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Binary quadratic NILM with minimum ON duration, max consecutive, and time-window constraints.

    Extends v2 with the C5 minimum ON duration constraint:
        sum_{tau=t}^{t+a_i-1} x[i,tau] >= a_i * u[i,t]   for all t
    where a_i = min_on_slots_by_device[device_name] is expressed in timesteps
    (not minutes), so it is granularity-agnostic — the caller converts from minutes.

    Args:
        signal: Aggregate power series (any regular DatetimeIndex frequency).
        devices: Device profiles to disaggregate.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Maps device name to minimum consecutive ON timesteps.
            Values of 0 or 1 are silently skipped (no constraint needed).
        max_consecutive_on_by_device: Maps device name to maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Maps device name to allowed daily windows as
            list of (start_min, end_min) in minutes-of-day.

    Returns:
        (disaggregation, info) where disaggregation maps device name to power series.
    """
    if not _HAS_GUROBI:
        raise ImportError("gurobipy is not installed.")

    T = len(signal)
    n = len(devices)
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

    # --- Pre-process constraint dicts into index-keyed form -----------------
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

    allowed_on_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                allowed_on_masks[i] = _build_allowed_window_mask(signal.index, windows)

    try:
        model = gp.Model("nilm_constrained_v3")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        # Fix variables at NaN timesteps
        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0

        # Fix x=0 outside allowed time windows
        for i, mask in allowed_on_masks.items():
            for t in range(T):
                if not mask[t]:
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0

        # C3/C4: state transition constraints
        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        # C5: minimum ON duration — if device turns on it must stay on for a_i slots
        for i, a_i in min_on_slots.items():
            for t in range(T - a_i + 1):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + a_i)) >= a_i * u[i, t],
                    name=f"mindur_{i}_{t}",
                )

        # C6: maximum consecutive ON (rolling window)
        for i, b_i in max_consecutive_limits.items():
            if b_i >= T:
                continue
            for t in range(T - b_i):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + b_i + 1)) <= b_i,
                    name=f"maxdur_{i}_{t}",
                )

        # Objective: squared reconstruction error
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
                x_sol * dev.p_typical_w, index=signal.index, dtype=float
            )
        return disaggregation, info

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during constrained_v3 solve: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": n, "n_timesteps": T,
        }


def _is_in_daily_window(
    minute_of_day: np.ndarray,
    start_min: int,
    end_min: int,
) -> np.ndarray:
    if end_min <= start_min:
        return (minute_of_day >= start_min) | (minute_of_day < end_min)
    return (minute_of_day >= start_min) & (minute_of_day < end_min)


def _build_allowed_window_mask(
    index: pd.DatetimeIndex,
    allowed_windows: list[tuple[int, int]],
) -> np.ndarray:
    if not allowed_windows:
        return np.ones(len(index), dtype=bool)

    minute_of_day = index.hour.to_numpy(dtype=int) * 60 + index.minute.to_numpy(dtype=int)
    allowed_mask = np.zeros(len(index), dtype=bool)
    for start_min, end_min in allowed_windows:
        allowed_mask |= _is_in_daily_window(minute_of_day, int(start_min), int(end_min))
    return allowed_mask


# constrained_v2 extends v1 with per-device daily hour windows to keep selected
# appliances OFF outside plausible activation periods.
def constrained_v2(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Solve a single-window constrained binary quadratic reconstruction problem.

    v2 adds per-device daily hour windows on top of the v1 transition and
    maximum consecutive ON constraints.

    Args:
        signal: Aggregate power series for a single time window.
        devices: Devices to include in the constrained model.
        time_limit: Solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        max_consecutive_on_by_device: Optional mapping of device name to the
            maximum number of consecutive ON minutes allowed in the window.
        allowed_on_windows_by_device: Optional mapping of device name to a
            list of allowed daily windows expressed as ``(start_min, end_min)``.

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
    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            limit = max_consecutive_on_by_device.get(dev.name)
            if limit is None:
                continue
            limit_int = int(limit)
            if limit_int < 1:
                raise ValueError(
                    f"max_consecutive_on_by_device[{dev.name!r}] must be >= 1"
                )
            max_consecutive_limits[i] = limit_int

    allowed_on_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            allowed_windows = allowed_on_windows_by_device.get(dev.name)
            if not allowed_windows:
                continue
            allowed_on_masks[i] = _build_allowed_window_mask(signal.index, allowed_windows)

    try:
        model = gp.Model("nilm_constrained_v2")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0

        for i, allowed_mask in allowed_on_masks.items():
            for t in range(T):
                if not allowed_mask[t]:
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0

        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        for i, max_consecutive in max_consecutive_limits.items():
            if max_consecutive >= T:
                continue
            for t in range(T - max_consecutive):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + max_consecutive + 1))
                    <= max_consecutive,
                    name=f"max_consecutive_{i}_{t}",
                )

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
        logger.warning("Gurobi error during constrained_v2 solve: %s", exc)
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


def constrained_v4(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """v3 with soft time-window penalties instead of hard exclusion.

    Devices active outside their allowed window incur a penalty proportional to
    ``time_window_penalty * p_typical²`` per slot, keeping the penalty in the
    same unit as the squared reconstruction error.  Hard UB=0 is NOT applied
    for time windows — only for NaN timesteps.

    With ``time_window_penalty = 1.0`` the cost of one out-of-window slot equals
    the squared power of the device (break-even when the signal perfectly matches
    the device and nothing else is on).  Use values > 1.0 for a stronger deterrent.

    Args:
        signal: Aggregate power series (any regular DatetimeIndex frequency).
        devices: Device profiles to disaggregate.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Maps device name to minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Maps device name to maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Maps device name to allowed daily windows as
            list of (start_min, end_min) in minutes-of-day (used as soft penalty).
        time_window_penalty: Dimensionless penalty factor; actual penalty per slot =
            time_window_penalty * p_typical_w² (default 1.0).

    Returns:
        (disaggregation, info) where disaggregation maps device name to power series.
    """
    if not _HAS_GUROBI:
        raise ImportError("gurobipy is not installed.")

    T = len(signal)
    n = len(devices)
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

    # Build out-of-window masks for penalty (True = slot is OUTSIDE allowed window)
    out_of_window_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                in_window = _build_allowed_window_mask(signal.index, windows)
                out_of_window_masks[i] = ~in_window

    try:
        model = gp.Model("nilm_constrained_v4")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0

        # C3/C4: state transition constraints
        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        # C5: minimum ON duration
        for i, a_i in min_on_slots.items():
            for t in range(T - a_i + 1):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + a_i)) >= a_i * u[i, t],
                    name=f"mindur_{i}_{t}",
                )

        # C6: maximum consecutive ON
        for i, b_i in max_consecutive_limits.items():
            if b_i >= T:
                continue
            for t in range(T - b_i):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + b_i + 1)) <= b_i,
                    name=f"maxdur_{i}_{t}",
                )

        # Objective: squared reconstruction error + soft time-window penalty
        objective = gp.QuadExpr()
        for t in range(T):
            if not valid[t]:
                continue
            power_sum = gp.quicksum(devices[i].p_typical_w * x[i, t] for i in range(n))
            residual = power_sum - y_safe[t]
            objective += residual * residual

        # Add soft penalty for activations outside allowed windows.
        # Scale by p_typical² so the penalty is commensurate with squared reconstruction error.
        for i, mask in out_of_window_masks.items():
            p_sq = devices[i].p_typical_w ** 2
            for t in range(T):
                if mask[t] and valid[t]:
                    objective += (time_window_penalty * p_sq) * x[i, t]

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
                x_sol * dev.p_typical_w, index=signal.index, dtype=float
            )
        return disaggregation, info

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during constrained_v4 solve: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": n, "n_timesteps": T,
        }


def constrained_v5(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """v4 extended with 3 power levels per device instead of binary ON/OFF.

    Each device can be in one of 4 states at any timestep:
      OFF   = 0 W
      low   = p_typical * (1 - power_level_variation)
      mid   = p_typical
      high  = p_typical * (1 + power_level_variation)

    The ON/OFF aggregate variable x[i,t] = sum_k z[i,t,k] is used for all
    transition and duration constraints, identical to v4.

    Args:
        signal: Aggregate power series (any regular DatetimeIndex frequency).
        devices: Device profiles to disaggregate.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Maps device name to minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Maps device name to maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Maps device name to allowed daily windows as
            list of (start_min, end_min) in minutes-of-day (used as soft penalty).
        time_window_penalty: Penalty per slot for activating outside the allowed window.
        power_level_variation: Fractional deviation for low/high levels (default 0.15 = ±15%).

    Returns:
        (disaggregation, info) where disaggregation maps device name to power series
        with actual estimated wattage (not just p_typical).
    """
    if not _HAS_GUROBI:
        raise ImportError("gurobipy is not installed.")

    T = len(signal)
    n = len(devices)
    K = 3  # number of ON power levels
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

    # Power levels per device: [low, mid, high]
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

    out_of_window_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                in_window = _build_allowed_window_mask(signal.index, windows)
                out_of_window_masks[i] = ~in_window

    try:
        model = gp.Model("nilm_constrained_v5")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        # z[i,t,k]: device i at timestep t is in power level k (k=0 low, 1 mid, 2 high)
        z = model.addVars(n, T, K, vtype=GRB.BINARY, name="z")
        # x[i,t]: aggregate ON indicator (= sum_k z[i,t,k]), used for transitions
        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
        dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        # Link x to z and enforce at-most-one level active
        for i in range(n):
            for t in range(T):
                model.addConstr(
                    x[i, t] == gp.quicksum(z[i, t, k] for k in range(K)),
                    name=f"link_{i}_{t}",
                )

        # Fix all variables to 0 at NaN timesteps
        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    u[i, t].UB = 0.0
                    dw[i, t].UB = 0.0
                    for k in range(K):
                        z[i, t, k].UB = 0.0

        # C3/C4: state transition constraints on aggregate ON indicator
        for i in range(n):
            model.addConstr(u[i, 0] == x[i, 0], name=f"init_u_{i}")
            model.addConstr(dw[i, 0] == 0, name=f"init_dw_{i}")
            for t in range(1, T):
                model.addConstr(
                    x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                    name=f"trans_{i}_{t}",
                )
                model.addConstr(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

        # C5: minimum ON duration
        for i, a_i in min_on_slots.items():
            for t in range(T - a_i + 1):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + a_i)) >= a_i * u[i, t],
                    name=f"mindur_{i}_{t}",
                )

        # C6: maximum consecutive ON
        for i, b_i in max_consecutive_limits.items():
            if b_i >= T:
                continue
            for t in range(T - b_i):
                model.addConstr(
                    gp.quicksum(x[i, tau] for tau in range(t, t + b_i + 1)) <= b_i,
                    name=f"maxdur_{i}_{t}",
                )

        # Objective: squared reconstruction error + soft time-window penalty
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

        for i, mask in out_of_window_masks.items():
            p_sq = devices[i].p_typical_w ** 2
            for t in range(T):
                if mask[t] and valid[t]:
                    objective += (time_window_penalty * p_sq) * x[i, t]

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
        logger.warning("Gurobi error during constrained_v5 solve: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": n, "n_timesteps": T,
        }


def constrained_v6(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """v5 extended with a per-activation penalty to discourage rapid ON/OFF cycling.

    Adds a linear term ``activation_penalty * p_typical² * u[i,t]`` to the objective
    for every rising-edge event u[i,t]=1, where u[i,t] is the turn-ON indicator
    already present in the transition constraints.  The penalty is scaled by p_typical²
    so it is commensurate with the squared reconstruction error.

    With ``activation_penalty = 1.0``: one extra activation costs as much as one
    slot where the device is on with zero signal (break-even).  Increase to
    discourage spurious cycling more aggressively.

    All features from v5 are retained:
      - soft time-window penalty (time_window_penalty)
      - 3 power levels per device (power_level_variation)
      - min ON duration, max consecutive ON (hard constraints)

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

    out_of_window_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                in_window = _build_allowed_window_mask(signal.index, windows)
                out_of_window_masks[i] = ~in_window

    try:
        model = gp.Model("nilm_constrained_v6")
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

        # Soft time-window penalty (scaled by p_typical²)
        for i, mask in out_of_window_masks.items():
            p_sq = devices[i].p_typical_w ** 2
            for t in range(T):
                if mask[t] and valid[t]:
                    objective += (time_window_penalty * p_sq) * x[i, t]

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
        logger.warning("Gurobi error during constrained_v6 solve: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": n, "n_timesteps": T,
        }


def constrained_v7(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """v6 extended: always-on devices modeled as variables instead of estimated baseline.

    All devices receive 3 power levels (low/mid/high = p_typical ± power_level_variation).
    The difference from v6 is how always-on devices are treated:

    Always-on devices (dev.always_on = True):
      - Exactly one power level active at every valid timestep: Σ_k z[i,t,k] = 1
      - No activation/deactivation variables (u, dw) — they never turn off
      - No time-window, min-ON, or max-consecutive constraints
      - Their contribution appears directly in the reconstruction error

    Event devices (dev.always_on = False):
      - Same as v6: binary ON/OFF aggregate, 3 levels, activation penalty,
        soft time-window penalty, min-ON and max-consecutive hard constraints

    This replaces the baseline-subtraction step entirely: the solver jointly
    optimises always-on level selection and event-device activation.

    Args:
        signal: Aggregate power series on the full signal (not residual).
        devices: All device profiles, including always-on ones.
        time_limit: Gurobi solver time limit in seconds.
        verbose: If False, suppress Gurobi console output.
        min_on_slots_by_device: Minimum consecutive ON timesteps (event devices only).
        max_consecutive_on_by_device: Maximum consecutive ON timesteps (event devices only).
        allowed_on_windows_by_device: Soft time-window penalty windows (event devices only).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor for per-activation penalty (× p_typical²).

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

    out_of_window_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for j, (_, dev) in enumerate(ev_devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                in_window = _build_allowed_window_mask(signal.index, windows)
                out_of_window_masks[j] = ~in_window

    try:
        model = gp.Model("nilm_constrained_v7")
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

        # ---- Event variables (identical to v6) --------------------------- #
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

        for j, a_j in min_on_slots.items():
            for t in range(T - a_j + 1):
                model.addConstr(
                    gp.quicksum(x_ev[j, tau] for tau in range(t, t + a_j)) >= a_j * u_ev[j, t],
                    name=f"ev_mindur_{j}_{t}",
                )

        for j, b_j in max_consecutive_limits.items():
            if b_j >= T:
                continue
            for t in range(T - b_j):
                model.addConstr(
                    gp.quicksum(x_ev[j, tau] for tau in range(t, t + b_j + 1)) <= b_j,
                    name=f"ev_maxdur_{j}_{t}",
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

        # Soft time-window penalty (event devices only)
        for j, mask in out_of_window_masks.items():
            p_sq = ev_devices[j][1].p_typical_w ** 2
            for t in range(T):
                if mask[t] and valid[t]:
                    objective += (time_window_penalty * p_sq) * x_ev[j, t]

        # Activation penalty (event devices only)
        for j, (_, dev) in enumerate(ev_devices):
            p_sq = dev.p_typical_w ** 2
            for t in range(T):
                if valid[t]:
                    objective += (activation_penalty * p_sq) * u_ev[j, t]

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
        logger.warning("Gurobi error during constrained_v7 solve: %s", exc)
        return zero_output, {
            "status": None, "has_solution": False, "is_optimal": False,
            "objective_value": None, "runtime_sec": None, "mip_gap": None,
            "n_devices": len(devices), "n_timesteps": T,
        }

