"""
Day-wise Gurobi NILM with soft time-window penalties, 3 power levels, and activation penalty (solve_activation).

Extends approach_gurobi_multistate by penalizing the number of ON/OFF cycles per day
via the rising-edge variable u[i,t].  A high activation_penalty discourages the solver
from switching a device on and off many times within the same day.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device
from scripts.nilm.devices import DeviceProfile
from scripts.nilm.time_windows import (
    WINDOW_PENALTY_MAX_FACTOR,
    WINDOW_PENALTY_RAMP_MIN,
)
from scripts.nilm.gurobi_methods import solve_activation

GRANULARITY_MIN = 15
RESAMPLE_METHODS = ("mean", "max", "min", "median")
_MAX_ON_FACTOR = 1.5


def resample_signal(
    signal: pd.Series,
    granularity_min: int = GRANULARITY_MIN,
    method: str = "mean",
) -> pd.Series:
    if method not in RESAMPLE_METHODS:
        raise ValueError(f"method must be one of {RESAMPLE_METHODS}, got {method!r}")
    return getattr(signal.resample(f"{granularity_min}min"), method)()


def _min_on_slots(dev: DeviceProfile, granularity_min: int) -> int:
    if dev.dur_min_min is None or dev.dur_min_min <= 0:
        return 1
    return max(1, math.ceil(dev.dur_min_min / granularity_min))


def _max_on_slots(dev: DeviceProfile, granularity_min: int) -> int | None:
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return max(1, math.ceil(dev.dur_typical_min * _MAX_ON_FACTOR / granularity_min))


def _allowed_windows(dev: DeviceProfile) -> list[tuple[int, int]] | None:
    if dev.start_window_start_min is None or dev.start_window_end_min is None:
        return None
    return [(dev.start_window_start_min, dev.start_window_end_min)]


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    baseline_mode: str = "duty_avg",
    baseline_method: str = "cyclic_median",
    resample_method: str = "mean",
    granularity_min: int = GRANULARITY_MIN,
    time_window_penalty: float = 1.0,
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day penalizing rapid cycling and out-of-window activations.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: Device profiles for the household.
        time_limit: Gurobi time limit per daily chunk in seconds.
        baseline_mode: 'peak' or 'duty_avg' for always-on devices.
        baseline_method: Baseline estimator (see baseline_load module).
        resample_method: Aggregation method for resampling ('mean','max','min','median').
        granularity_min: Bin size in minutes (default 15).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).  Graduated by
            distance from the nearest allowed window.
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor penalizing each ON transition (× p_typical²).
        verbose: If True, keep Gurobi console output.

    Returns:
        dict mapping device name -> power series (actual estimated W) aligned to signal.index.
    """
    nan_mask = signal.isna()

    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index, dtype=float)
        s[nan_mask] = np.nan
        result[dev.name] = s

    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

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
        s = baseline_by_device.get(dev.name, pd.Series(0.0, index=signal.index))
        s = s.reindex(signal.index, fill_value=0.0)
        s[nan_mask] = np.nan
        result[dev.name] = s

    if not present_events:
        return result

    residual_1min = pd.Series(
        np.clip(
            signal.fillna(0.0).to_numpy(dtype=float)
            - baseline_total.fillna(0.0).to_numpy(dtype=float),
            0.0,
            None,
        ),
        index=signal.index,
        dtype=float,
    )
    residual_1min[nan_mask] = np.nan
    residual = resample_signal(residual_1min, granularity_min=granularity_min, method=resample_method)

    min_on: dict[str, int] = {}
    max_on: dict[str, int] = {}
    allowed_wins: dict[str, list[tuple[int, int]]] = {}

    for dev in present_events:
        a_i = _min_on_slots(dev, granularity_min)
        if a_i >= 2:
            min_on[dev.name] = a_i
        b_i = _max_on_slots(dev, granularity_min)
        if b_i is not None:
            max_on[dev.name] = b_i
        wins = _allowed_windows(dev)
        if wins:
            allowed_wins[dev.name] = wins

    _dp_idx = residual.index.tz_convert(None) if residual.index.tz is not None else residual.index
    day_periods = _dp_idx.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present_events}

    for day in unique_days:
        day_mask = day_periods == day
        day_signal = residual[day_mask]
        if day_signal.dropna().empty:
            continue
        
        day_result, info = solve_activation(
            signal=day_signal,
            devices=present_events,
            time_limit=time_limit,
            verbose=verbose,
            min_on_slots_by_device=min_on or None,
            max_consecutive_on_by_device=max_on or None,
            allowed_on_windows_by_device=allowed_wins or None,
            time_window_penalty=time_window_penalty,
            window_penalty_ramp_min=window_penalty_ramp_min,
            window_penalty_max_factor=window_penalty_max_factor,
            power_level_variation=power_level_variation,
            activation_penalty=activation_penalty,
        )

        if verbose:
            print(
                f"  {day}: status={info['status']} optimal={info['is_optimal']} "
                f"obj={info['objective_value']:.1f} t={info['runtime_sec']:.1f}s"
            )

        for dev in present_events:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present_events:
        chunks = per_device_chunks[dev.name]
        combined_coarse = pd.concat(chunks).sort_index() if chunks else pd.Series(0.0, index=residual.index, dtype=float)
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
