"""
approach_cvxpy_activation — Day-wise NILM with 3 power levels and activation
penalty via HiGHS L1-MILP (free, no license, no crash risk).

Constraint structure identical to approach_gurobi_activation.py (constrained_v6):
  - 3 power levels per device (low/mid/high = p_typical ± power_level_variation)
  - Soft time-window penalty
  - Per-activation penalty
  - Hard min-ON and max-consecutive-ON constraints

Objective: L1 (sum of absolute errors) instead of L2 (Gurobi).
HiGHS handles only LP/MILP — no quadratic objective, no SCIP, no crash risk.
Penalty scaling: × p_typical (not × p_typical² as in Gurobi), consistent with L1.

Installation:
    pip install highspy
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device
from scripts.nilm.devices import DeviceProfile
from scripts.nilm.highs_methods import constrained_highs_multistate

GRANULARITY_MIN = 15
RESAMPLE_METHODS = ("mean", "max", "min", "median")
_MAX_ON_FACTOR = 1.5


def resample_signal(signal: pd.Series, granularity_min: int = GRANULARITY_MIN, method: str = "mean") -> pd.Series:
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
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day-by-day with 3 power levels and activation penalty (HiGHS L1-MILP).

    Same constraint structure as approach_gurobi_activation (3 levels, activation penalty,
    soft time-window, min/max ON). Objective is L1 instead of L2 — avoids quadratic
    terms and any crash risk. Penalties scale by p_typical (consistent with L1).

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: Device profiles for the household.
        time_limit: HiGHS time limit per daily chunk in seconds.
        baseline_mode: 'peak' or 'duty_avg' for always-on devices.
        baseline_method: Baseline estimator.
        resample_method: Aggregation method for resampling.
        granularity_min: Bin size in minutes (default 15).
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot.
        power_level_variation: Fractional spread of low/high power levels (default ±15%).
        activation_penalty: Penalty factor (× p_typical) per ON transition.
        verbose: Show HiGHS console output.

    Returns:
        dict mapping device name → power series with actual estimated wattage,
        aligned to signal.index (1-min).
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
        signal=signal, devices=present_always_on,
        method=baseline_method, baseline_mode=baseline_mode,
    )
    baseline_by_device = split_baseline_by_device(
        baseline_total=baseline_total, devices=present_always_on, baseline_mode=baseline_mode,
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
            0.0, None,
        ),
        index=signal.index, dtype=float,
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

    _idx = residual.index.tz_convert(None) if residual.index.tz is not None else residual.index
    day_periods = _idx.to_period("D")
    unique_days = day_periods.unique().sort_values()
    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present_events}

    for day in unique_days:
        day_mask = day_periods == day
        day_signal = residual[day_mask]
        if day_signal.dropna().empty:
            continue

        day_result, info = constrained_highs_multistate(
            signal=day_signal,
            devices=present_events,
            time_limit=time_limit,
            verbose=verbose,
            min_on_slots_by_device=min_on or None,
            max_consecutive_on_by_device=max_on or None,
            allowed_on_windows_by_device=allowed_wins or None,
            time_window_penalty=time_window_penalty,
            power_level_variation=power_level_variation,
            activation_penalty=activation_penalty,
        )

        _obj = f"{info['objective_value']:.1f}" if info['objective_value'] is not None else "N/A"
        print(f"  cvxpy_activation [{day}] status={info['status']} obj={_obj} t={info['runtime_sec']:.1f}s")

        for dev in present_events:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present_events:
        chunks = per_device_chunks[dev.name]
        combined_coarse = (
            pd.concat(chunks).sort_index() if chunks
            else pd.Series(0.0, index=residual.index, dtype=float)
        )
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
