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
from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ON_BY_DEVICE = {"Microonde": 5}
ALLOWED_ON_WINDOWS_BY_DEVICE = {
    "Microonde": [(6 * 60, 10 * 60), (12 * 60, 15 * 60), (19 * 60, 22 * 60)],
    "Lavatrice": [(6 * 60, 22 * 60)],
}


def _build_power_levels(device: DeviceProfile, n_power_levels: int) -> np.ndarray:
    if n_power_levels < 3 or n_power_levels % 2 == 0:
        raise ValueError("n_power_levels must be an odd integer >= 3")

    side_count = n_power_levels // 2
    lower_levels = [
        device.p_min_w + (device.p_typical_w - device.p_min_w) * step / (side_count + 1)
        for step in range(1, side_count + 1)
    ]
    upper_levels = [
        device.p_typical_w + (device.p_max_w - device.p_typical_w) * step / (side_count + 1)
        for step in range(1, side_count + 1)
    ]
    return np.asarray([*lower_levels, device.p_typical_w, *upper_levels], dtype=float)


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


def solve_multilevel_window(
    signal: pd.Series,
    devices: list[DeviceProfile],
    n_power_levels: int,
    constraint_version: str = "unconstrained",
    time_limit: float = 60.0,
    verbose: bool = False,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float | int | bool | None]]:
    """Solve a single-window multi-level reconstruction problem.

    Each device can be OFF or select exactly one among ``n_power_levels`` ON
    wattage levels. The central level is ``p_typical_w`` and the remaining
    levels are linearly interpolated toward ``p_min_w`` and ``p_max_w`` without
    using the extremes directly.

    constraint_version values:
        - ``unconstrained``: only level selection and reconstruction error.
        - ``v1``: adds ``u``/``dw`` transition variables and max consecutive ON.
        - ``v2``: adds the v1 constraints plus allowed daily ON windows.
    """
    if constraint_version not in {"unconstrained", "v1", "v2"}:
        raise ValueError("constraint_version must be one of: 'unconstrained', 'v1', 'v2'")

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
            "n_power_levels": n_power_levels,
            "constraint_version": constraint_version,
        }

    valid = signal.notna().to_numpy()
    y = signal.to_numpy(dtype=float)
    y_safe = np.where(valid, y, 0.0)
    power_levels = [_build_power_levels(dev, n_power_levels) for dev in devices]
    has_transitions = constraint_version in {"v1", "v2"}

    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device and has_transitions:
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
    if constraint_version == "v2" and allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            allowed_windows = allowed_on_windows_by_device.get(dev.name)
            if not allowed_windows:
                continue
            allowed_on_masks[i] = _build_allowed_window_mask(signal.index, allowed_windows)

    try:
        model = gp.Model(f"nilm_multilevel_{constraint_version}_{n_power_levels}lvl")
        model.setParam("TimeLimit", time_limit)
        if not verbose:
            model.setParam("OutputFlag", 0)
            model.setParam("LogToConsole", 0)

        x = model.addVars(n, T, vtype=GRB.BINARY, name="x")
        z = model.addVars(n, n_power_levels, T, vtype=GRB.BINARY, name="z")
        u = dw = None
        if has_transitions:
            u = model.addVars(n, T, vtype=GRB.BINARY, name="u")
            dw = model.addVars(n, T, vtype=GRB.BINARY, name="dw")

        for t in range(T):
            if not valid[t]:
                for i in range(n):
                    x[i, t].UB = 0.0
                    for level_idx in range(n_power_levels):
                        z[i, level_idx, t].UB = 0.0
                    if has_transitions:
                        u[i, t].UB = 0.0
                        dw[i, t].UB = 0.0

        for i, allowed_mask in allowed_on_masks.items():
            for t in range(T):
                if not allowed_mask[t]:
                    x[i, t].UB = 0.0
                    for level_idx in range(n_power_levels):
                        z[i, level_idx, t].UB = 0.0
                    if has_transitions:
                        u[i, t].UB = 0.0

        for i in range(n):
            for t in range(T):
                model.addConstr(
                    gp.quicksum(z[i, level_idx, t] for level_idx in range(n_power_levels))
                    == x[i, t],
                    name=f"level_select_{i}_{t}",
                )

        if has_transitions:
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
            power_sum = gp.quicksum(
                power_levels[i][level_idx] * z[i, level_idx, t]
                for i in range(n)
                for level_idx in range(n_power_levels)
            )
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
            "n_power_levels": n_power_levels,
            "constraint_version": constraint_version,
        }

        if not has_solution:
            return zero_output, info

        disaggregation: dict[str, pd.Series] = {}
        for i, dev in enumerate(devices):
            power_sol = np.array(
                [
                    sum(
                        power_levels[i][level_idx] * round(z[i, level_idx, t].X)
                        for level_idx in range(n_power_levels)
                    )
                    for t in range(T)
                ],
                dtype=float,
            )
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)
        return disaggregation, info
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gurobi error during multi-level solve: %s", exc)
        return zero_output, {
            "status": None,
            "has_solution": False,
            "is_optimal": False,
            "objective_value": None,
            "runtime_sec": None,
            "mip_gap": None,
            "n_devices": n,
            "n_timesteps": T,
            "n_power_levels": n_power_levels,
            "constraint_version": constraint_version,
        }


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    baseline_mode: str = "duty_avg",
    verbose: bool = False,
    n_power_levels: int = 3,
    constraint_version: str = "unconstrained",
    baseline_method: str = "cyclic_median",
) -> dict[str, pd.Series]:
    """Day-wise multi-level Gurobi NILM with optional v1/v2 constraints."""
    if baseline_mode not in {"peak", "duty_avg"}:
        raise ValueError("baseline_mode must be one of: 'peak', 'duty_avg'")

    nan_mask = signal.isna()
    result: dict[str, pd.Series] = {}
    for dev in devices:
        series = pd.Series(0.0, index=signal.index, dtype=float)
        series[nan_mask] = np.nan
        result[dev.name] = series

    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

    max_consecutive_on_by_device = {
        dev.name: MAX_CONSECUTIVE_ON_BY_DEVICE[dev.name]
        for dev in present_events
        if dev.name in MAX_CONSECUTIVE_ON_BY_DEVICE
    }
    allowed_on_windows_by_device = {
        dev.name: ALLOWED_ON_WINDOWS_BY_DEVICE[dev.name]
        for dev in present_events
        if dev.name in ALLOWED_ON_WINDOWS_BY_DEVICE
    }

    baseline_total, _ = estimate_always_on_baseline(
        signal=signal,
        devices=present_always_on,
        method=baseline_method,
        baseline_mode=baseline_mode,
    )
    baseline_by_device = split_baseline_by_device(
        baseline_total=baseline_total,
        devices=present_always_on,
        baseline_mode=baseline_mode,
    )
    for dev in present_always_on:
        baseline_series = baseline_by_device.get(dev.name)
        if baseline_series is None:
            baseline_series = pd.Series(0.0, index=signal.index, dtype=float)
            baseline_series[nan_mask] = np.nan
        result[dev.name] = baseline_series

    if not present_events:
        return result

    residual_signal = pd.Series(
        np.clip(
            signal.fillna(0.0).to_numpy(dtype=float)
            - baseline_total.fillna(0.0).to_numpy(dtype=float),
            0.0,
            None,
        ),
        index=signal.index,
        dtype=float,
    )
    residual_signal[nan_mask] = np.nan

    day_periods = residual_signal.index.to_period("D")
    unique_days = day_periods.unique().sort_values()
    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present_events}

    for idx, day in enumerate(unique_days):
        day_mask = day_periods == day
        day_signal = residual_signal[day_mask]
        if day_signal.empty:
            continue

        if idx == 5: 
            break

        day_result, _ = solve_multilevel_window(
            signal=day_signal,
            devices=present_events,
            n_power_levels=n_power_levels,
            constraint_version=constraint_version,
            time_limit=time_limit,
            verbose=verbose,
            max_consecutive_on_by_device=max_consecutive_on_by_device or None,
            allowed_on_windows_by_device=allowed_on_windows_by_device or None,
        )
        for dev in present_events:
            chunk = day_result.get(dev.name)
            if chunk is None:
                chunk = pd.Series(0.0, index=day_signal.index, dtype=float)
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present_events:
        if per_device_chunks[dev.name]:
            combined = pd.concat(per_device_chunks[dev.name]).reindex(signal.index).fillna(0.0)
        else:
            combined = pd.Series(0.0, index=signal.index, dtype=float)
        combined[nan_mask] = np.nan
        result[dev.name] = combined

    return result