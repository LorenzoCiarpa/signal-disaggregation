"""
Day-wise Gurobi NILM with explicit always-on baseline subtraction.

Workflow:
1. Present always-on devices (typically fridge/freezer) are assigned as a
   constant baseline over the full signal.
2. Their baseline is subtracted from the aggregate signal.
3. The positive residual is solved day by day with the current constrained_v2
    Gurobi binary quadratic model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device
from scripts.nilm.gurobi_methods import solve_unconstrained_window, constrained_v1, constrained_v2
from scripts.nilm.devices import DeviceProfile


MAX_CONSECUTIVE_ON_BY_DEVICE = {"Microonde": 5}
ALLOWED_ON_WINDOWS_BY_DEVICE = {
    "Microonde": [(6 * 60, 10 * 60), (12 * 60, 15 * 60), (19 * 60, 22 * 60)],
    "Lavatrice": [(6 * 60, 22 * 60)],
}


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    baseline_mode: str = "duty_avg",
    baseline_method: str = "cyclic_median",
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day after subtracting an estimated fridge/freezer baseline.

    Args:
        signal: Aggregate 1-minute signal with DatetimeIndex.
        devices: Device profiles for the household.
        time_limit: Gurobi time limit per daily chunk.
        baseline_mode:
            "peak" uses p_typical_w for always-on devices.
            "duty_avg" uses p_typical_w * duty_cycle for always-on devices.
        baseline_method:
            Baseline estimator used to identify the always-on base load.
            Supported methods are implemented in scripts.nilm.baseline_load.
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
        

        day_result, _ = solve_unconstrained_window(
            signal=day_signal,
            devices=present_events,
            time_limit=time_limit,
            verbose=verbose,
            # max_consecutive_on_by_device=max_consecutive_on_by_device or None,
            # allowed_on_windows_by_device=allowed_on_windows_by_device or None,
        )
        
        # day_result, _ = constrained_v1(
        #     signal=day_signal,
        #     devices=present_events,
        #     time_limit=time_limit,
        #     verbose=verbose,
        #     # max_consecutive_on_by_device=max_consecutive_on_by_device or None,
        #     # allowed_on_windows_by_device=allowed_on_windows_by_device or None,
        # )

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