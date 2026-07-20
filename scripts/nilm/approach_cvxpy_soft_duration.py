"""
Day-wise HiGHS L1-MILP NILM with soft duration limits.

Free-solver mirror of approach_gurobi_soft_duration: same model as approach_cvxpy_full
(all devices as variables, always-on included) but the min-ON and max-consecutive
limits are penalised in the objective instead of enforced, so a device whose power
matches a plateau can cover it even if the plateau runs slightly past the declared
maximum duration.

Penalties are in slots and scaled by p_typical (the L1 objective scale), so
lambda = 1.0 is the break-even against one slot of unexplained power.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.time_windows import (
    WINDOW_PENALTY_MAX_FACTOR,
    WINDOW_PENALTY_RAMP_MIN,
)
from scripts.nilm.highs_methods import constrained_highs_full

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
    """Soft upper bound in slots, preferring the survey-declared maximum."""
    if dev.duration_minutes_max is not None and dev.duration_minutes_max > 0:
        return max(1, math.ceil(dev.duration_minutes_max / granularity_min))
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return max(1, math.ceil(dev.dur_typical_min * _MAX_ON_FACTOR / granularity_min))


def _expected_daily_on_slots(dev: DeviceProfile, granularity_min: int) -> float | None:
    """Expected ON slots per day from the survey usage frequency and duration."""
    if dev.frequency_per_week is None or dev.frequency_per_week <= 0:
        return None
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return (dev.frequency_per_week / 7.0) * dev.dur_typical_min / granularity_min


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
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    activation_penalty: float = 1.0,
    duration_penalty_block: float = 0.5,
    duration_penalty_daily: float = 0.5,
    power_level_variation: float = 0.15,
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day-by-day with always-on devices as multistate variables (HiGHS L1-MILP).

    Identical constraint structure to gurobi_full (solve_full): always-on devices
    are variables in the solver (not baseline-subtracted), each forced to have exactly
    one power level active per valid timestep.  Objective is L1 instead of L2.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: ALL device profiles, including always-on ones.
        time_limit: HiGHS time limit per daily chunk in seconds.
        resample_method: Aggregation method for resampling ('mean','max','min','median').
        granularity_min: Bin size in minutes (default 15).
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot.  Graduated by
            distance from the nearest allowed window.
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        activation_penalty: Penalty factor (× p_typical) per ON transition (event devices).
        power_level_variation: Fractional spread of low/high levels around p_typical (default ±15%).
        verbose: Show HiGHS console output.

    Returns:
        dict mapping every device name → power series with actual estimated wattage,
        aligned to signal.index (1-min).
    """
    nan_mask = signal.isna()

    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index, dtype=float)
        s[nan_mask] = np.nan
        result[dev.name] = s

    present = [d for d in devices if d.prior_weight >= 1.0]
    if not present:
        return result

    # Resample raw signal (no baseline subtraction)
    signal_resampled = resample_signal(signal, granularity_min=granularity_min, method=resample_method)

    # Build per-event-device constraint dicts
    ev_devices = [d for d in present if not d.always_on]
    min_on: dict[str, int] = {}
    max_on: dict[str, int] = {}
    allowed_wins: dict[str, list[tuple[int, int]]] = {}
    expected_daily: dict[str, float] = {}
    for dev in ev_devices:
        a_i = _min_on_slots(dev, granularity_min)
        if a_i >= 2:
            min_on[dev.name] = a_i
        b_i = _max_on_slots(dev, granularity_min)
        if b_i is not None:
            max_on[dev.name] = b_i
        wins = _allowed_windows(dev)
        if wins:
            allowed_wins[dev.name] = wins
        expected = _expected_daily_on_slots(dev, granularity_min)
        if expected is not None:
            expected_daily[dev.name] = expected

    _idx = signal_resampled.index.tz_convert(None) if signal_resampled.index.tz is not None else signal_resampled.index
    day_periods = _idx.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present}

    for day in unique_days:
        day_mask = day_periods == day
        day_signal = signal_resampled[day_mask]
        if day_signal.dropna().empty:
            continue

        day_result, info = constrained_highs_full(
            signal=day_signal,
            devices=present,
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
            duration_penalty_block=duration_penalty_block,
            duration_penalty_daily=duration_penalty_daily,
            expected_daily_on_slots_by_device=expected_daily or None,
        )

        _obj = f"{info['objective_value']:.1f}" if info['objective_value'] is not None else "N/A"
        print(f"  cvxpy_full [{day}] status={info['status']} obj={_obj} t={info['runtime_sec']:.1f}s")

        for dev in present:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present:
        chunks = per_device_chunks[dev.name]
        combined_coarse = (
            pd.concat(chunks).sort_index() if chunks
            else pd.Series(0.0, index=signal_resampled.index, dtype=float)
        )
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
