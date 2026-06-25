"""
Day-wise Gurobi NILM with all devices as variables, including always-on (constrained_v7).

Extends approach_gurobi_activation by eliminating the baseline-estimation step:
always-on devices (fridge, freezer, ...) are modeled as variables with exactly one
power level active at each timestep (Σ_k z[i,t,k] = 1), instead of being estimated
upfront and subtracted from the signal.  The solver jointly optimises always-on level
selection and event-device activation against the raw aggregate signal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.gurobi_methods import constrained_v7

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
    resample_method: str = "mean",
    granularity_min: int = GRANULARITY_MIN,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day, modeling always-on devices as multistate variables.

    Always-on devices are NOT estimated via baseline: they enter the solver directly
    with a hard 'always active' constraint (exactly one power level per timestep).
    Event devices keep all v6 features: 3 levels, activation penalty, soft time window.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: All device profiles, including always-on ones.
        time_limit: Gurobi time limit per daily chunk in seconds.
        resample_method: Aggregation method for resampling ('mean','max','min','median').
        granularity_min: Bin size in minutes (default 15).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor penalizing each ON transition (× p_typical²).
        verbose: If True, keep Gurobi console output.

    Returns:
        dict mapping every device name -> power series aligned to signal.index (1-min).
    """
    nan_mask = signal.isna()

    present = [d for d in devices if d.prior_weight >= 1.0]

    result: dict[str, pd.Series] = {}
    for dev in present:
        s = pd.Series(0.0, index=signal.index, dtype=float)
        s[nan_mask] = np.nan
        result[dev.name] = s

    if not present:
        return result

    # Resample raw signal directly (no baseline subtraction)
    signal_clean = signal.copy()
    resampled = resample_signal(signal_clean, granularity_min=granularity_min, method=resample_method)

    event_devices = [d for d in present if not d.always_on]

    min_on: dict[str, int] = {}
    max_on: dict[str, int] = {}
    allowed_wins: dict[str, list[tuple[int, int]]] = {}

    for dev in event_devices:
        a_i = _min_on_slots(dev, granularity_min)
        if a_i >= 2:
            min_on[dev.name] = a_i
        b_i = _max_on_slots(dev, granularity_min)
        if b_i is not None:
            max_on[dev.name] = b_i
        wins = _allowed_windows(dev)
        if wins:
            allowed_wins[dev.name] = wins

    day_periods = resampled.index.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present}

    for day_idx, day in enumerate(unique_days):
        day_mask = day_periods == day
        day_signal = resampled[day_mask]
        if day_signal.dropna().empty:
            continue
        
        if day_idx == 5:
            break 
        
        day_result, info = constrained_v7(
            signal=day_signal,
            devices=present,
            time_limit=time_limit,
            verbose=verbose,
            min_on_slots_by_device=min_on or None,
            max_consecutive_on_by_device=max_on or None,
            allowed_on_windows_by_device=allowed_wins or None,
            time_window_penalty=time_window_penalty,
            power_level_variation=power_level_variation,
            activation_penalty=activation_penalty,
        )

        if verbose:
            print(
                f"  {day}: status={info['status']} optimal={info['is_optimal']} "
                f"obj={info['objective_value']:.1f} t={info['runtime_sec']:.1f}s"
            )

        for dev in present:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present:
        chunks = per_device_chunks[dev.name]
        combined_coarse = pd.concat(chunks).sort_index() if chunks else pd.Series(0.0, index=resampled.index, dtype=float)
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
