"""
Day-wise HiGHS L1-MILP NILM with a weekly activation quota carried across days.

Free-solver mirror of approach_gurobi_weekly_quota.  Builds on
approach_cvxpy_soft_duration (soft duration limits, always-on as variables)
and adds the survey's weekly usage frequency as a running budget rather than a
per-day prior: if the questionnaire says the washing machine runs twice a week and
the solver has already used it twice, a third activation costs substantially more.

How the counter works
---------------------
Each day is solved as an independent model, so the weekly count cannot be a variable
inside it — it is state carried between solves:

    1. days are processed in chronological order
    2. the counter resets when the ISO week changes (i.e. every Monday)
    3. the remaining quota enters day d's model as a constant
    4. after solving, the activations realised that day are counted from the
       solution and added to the counter

Inside a day the model only needs an excess variable e_i >= 0 with

    Σ_t u[i,t] − e_i  ≤  remaining_i

so the first `remaining_i` activations pay the usual activation_penalty and each one
past the quota also pays the surcharge.

Because days are solved one at a time, the allocation is greedy: an early day spends
the quota freely and later days in the same week pay for it.  Enforcing the quota
optimally over a whole week would require solving the seven days jointly, which is far
more expensive; the greedy version still prevents the implausible "washing machine ran
every day" outcome, which is the point.

Calibration
-----------
The surcharge is scaled per device: explaining one activation of a device that runs for
`s` slots gains roughly `s · p`, so the surcharge must exceed that to bite.  It is set
to ``over_activation_factor × s_i`` with s_i the device's typical ON length in slots,
which keeps the same effective strength for a 15-minute microwave cycle and a 2-hour
washing machine cycle.
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
    """Soft upper bound in slots, preferring the survey-declared maximum."""
    if dev.duration_minutes_max is not None and dev.duration_minutes_max > 0:
        return max(1, math.ceil(dev.duration_minutes_max / granularity_min))
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return max(1, math.ceil(dev.dur_typical_min * _MAX_ON_FACTOR / granularity_min))


def _typical_on_slots(dev: DeviceProfile, granularity_min: int) -> int:
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return 1
    return max(1, round(dev.dur_typical_min / granularity_min))


def _expected_daily_on_slots(dev: DeviceProfile, granularity_min: int) -> float | None:
    if dev.frequency_per_week is None or dev.frequency_per_week <= 0:
        return None
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return (dev.frequency_per_week / 7.0) * dev.dur_typical_min / granularity_min


def _weekly_quota(dev: DeviceProfile) -> float | None:
    """Activations allowed per week, preferring the survey's declared upper bound."""
    if dev.frequency_per_week_max is not None and dev.frequency_per_week_max > 0:
        return float(dev.frequency_per_week_max)
    if dev.frequency_per_week is not None and dev.frequency_per_week > 0:
        return float(dev.frequency_per_week)
    return None


def _allowed_windows(dev: DeviceProfile) -> list[tuple[int, int]] | None:
    if dev.start_window_start_min is None or dev.start_window_end_min is None:
        return None
    return [(dev.start_window_start_min, dev.start_window_end_min)]


def _count_activations(series: pd.Series) -> int:
    """Number of rising edges (OFF→ON) in a device's power series."""
    on = (series.to_numpy(dtype=float) > 0.0)
    if on.size == 0:
        return 0
    return int(on[0]) + int(((~on[:-1]) & on[1:]).sum())


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    resample_method: str = "mean",
    granularity_min: int = GRANULARITY_MIN,
    time_window_penalty: float = 1.0,
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
    duration_penalty_block: float = 0.5,
    duration_penalty_daily: float = 0.0,
    over_activation_factor: float = 1.5,
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day, charging activations past the weekly survey quota.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: All device profiles, including always-on ones.
        time_limit: HiGHS time limit per daily chunk in seconds.
        resample_method: Aggregation method for resampling ('mean','max','min','median').
        granularity_min: Bin size in minutes (default 15).
        time_window_penalty: Dimensionless factor for out-of-window penalty (x p_typical).
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        power_level_variation: Fractional spread of low/high levels around p_typical.
        activation_penalty: Dimensionless factor penalizing each ON transition (x p_typical).
        duration_penalty_block: Cost per slot of deviation from the min/max duration bounds.
        duration_penalty_daily: Cost per slot of deviation of daily ON time from the
            survey expectation.  Defaults to 0 here since the weekly quota already
            constrains total usage.
        over_activation_factor: Multiplier on the per-device break-even for the quota
            surcharge.  1.0 is exactly break-even against explaining one typical
            activation; the default 1.5 puts it comfortably above.
        verbose: If True, show HiGHS solver output.

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

    signal_clean = signal.copy()
    resampled = resample_signal(signal_clean, granularity_min=granularity_min, method=resample_method)

    event_devices = [d for d in present if not d.always_on]

    min_on: dict[str, int] = {}
    max_on: dict[str, int] = {}
    allowed_wins: dict[str, list[tuple[int, int]]] = {}
    expected_daily: dict[str, float] = {}
    weekly_quota: dict[str, float] = {}
    over_penalty: dict[str, float] = {}

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
        expected = _expected_daily_on_slots(dev, granularity_min)
        if expected is not None:
            expected_daily[dev.name] = expected
        quota = _weekly_quota(dev)
        if quota is not None:
            weekly_quota[dev.name] = quota
            # Break-even is the typical ON length in slots (explaining one activation
            # of s slots gains ~s·p), so scale the surcharge by it.
            over_penalty[dev.name] = over_activation_factor * _typical_on_slots(dev, granularity_min)

    _dp_idx = resampled.index.tz_convert(None) if resampled.index.tz is not None else resampled.index
    day_periods = _dp_idx.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present}

    used_this_week: dict[str, float] = {name: 0.0 for name in weekly_quota}
    current_week: tuple[int, int] | None = None

    for day in unique_days:
        day_mask = day_periods == day
        day_signal = resampled[day_mask]
        if day_signal.dropna().empty:
            continue

        # Reset the counter whenever the ISO week changes (i.e. every Monday).
        day_ts = day.to_timestamp()
        iso = day_ts.isocalendar()
        week_key = (iso[0], iso[1])
        if week_key != current_week:
            current_week = week_key
            used_this_week = {name: 0.0 for name in weekly_quota}

        remaining = {
            name: max(0.0, quota - used_this_week.get(name, 0.0))
            for name, quota in weekly_quota.items()
        }

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
            over_activation_penalty_by_device=over_penalty or None,
            remaining_activations_by_device=remaining or None,
        )

        for name in weekly_quota:
            chunk = day_result.get(name)
            if chunk is not None:
                used_this_week[name] = used_this_week.get(name, 0.0) + _count_activations(chunk)

        if verbose:
            spent = ", ".join(
                f"{name}={used_this_week[name]:.0f}/{weekly_quota[name]:.0f}"
                for name in sorted(weekly_quota)
            )
            print(
                f"  {day} [week {week_key[1]}]: status={info['status']} "
                f"obj={info['objective_value']:.1f} t={info['runtime_sec']:.1f}s | {spent}"
            )

        for dev in present:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    for dev in present:
        chunks = per_device_chunks[dev.name]
        combined_coarse = (
            pd.concat(chunks).sort_index() if chunks
            else pd.Series(0.0, index=resampled.index, dtype=float)
        )
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
