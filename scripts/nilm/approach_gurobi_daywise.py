"""
Day-wise Gurobi NILM with explicit always-on baseline subtraction.

Workflow:
1. Present always-on devices (typically fridge/freezer) are assigned as a
   constant baseline over the full signal.
2. Their baseline is subtracted from the aggregate signal.
3. The positive residual is solved day by day with the current unconstrained
   Gurobi binary quadratic model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.nilm.approach_gurobi import solve_unconstrained_window
from scripts.nilm.devices import DeviceProfile


def _baseline_power(device: DeviceProfile, baseline_mode: str) -> float:
    if baseline_mode == "duty_avg" and device.duty_cycle is not None:
        return float(device.p_typical_w * device.duty_cycle)
    return float(device.p_typical_w)


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    baseline_mode: str = "peak",
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day after subtracting a fridge/freezer baseline.

    Args:
        signal: Aggregate 1-minute signal with DatetimeIndex.
        devices: Device profiles for the household.
        time_limit: Gurobi time limit per daily chunk.
        baseline_mode:
            "peak" uses p_typical_w for always-on devices.
            "duty_avg" uses p_typical_w * duty_cycle for always-on devices.
        verbose: If True, keep Gurobi console logs enabled.

    Returns:
        dict mapping device name -> power series aligned to the input signal.
    """
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

    baseline_total = 0.0
    for dev in present_always_on:
        power_val = _baseline_power(dev, baseline_mode)
        baseline_total += power_val
        baseline_series = pd.Series(power_val, index=signal.index, dtype=float)
        baseline_series[nan_mask] = np.nan
        result[dev.name] = baseline_series

    if not present_events:
        return result

    residual_signal = pd.Series(
        np.clip(signal.fillna(0.0).to_numpy(dtype=float) - baseline_total, 0.0, None),
        index=signal.index,
        dtype=float,
    )
    residual_signal[nan_mask] = np.nan

    day_periods = residual_signal.index.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present_events}

    for day in unique_days:
        day_mask = day_periods == day
        day_signal = residual_signal[day_mask]
        if day_signal.empty:
            continue

        day_result, _ = solve_unconstrained_window(
            signal=day_signal,
            devices=present_events,
            time_limit=time_limit,
            verbose=verbose,
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