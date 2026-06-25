"""
Day-wise Gurobi NILM on 15-minute aggregated signal with minimum ON duration.

Workflow
--------
1. Resample the 1-minute aggregate signal to 15-minute bins
   (configurable method: mean, max, min, median).
2. Estimate and subtract always-on baseline (fridge/freezer).
3. For each day solve with constrained_v3:
   - C5: minimum ON duration derived from dev.dur_min_min / GRANULARITY_MIN.
   - C6: maximum consecutive ON from dev.dur_typical_min * 1.5 / GRANULARITY_MIN.
   - Time-window mask from dev.start_window_start_min / start_window_end_min.
4. Forward-fill results back to 1-minute granularity for compatibility.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device
from scripts.nilm.devices import DeviceProfile
from scripts.nilm.gurobi_methods import constrained_v3

GRANULARITY_MIN = 15
RESAMPLE_METHODS = ("mean", "max", "min", "median")

# Maximum ON slots factor applied to dur_typical_min
_MAX_ON_FACTOR = 1.5


def resample_signal(
    signal: pd.Series,
    granularity_min: int = GRANULARITY_MIN,
    method: str = "mean",
) -> pd.Series:
    """Resample a 1-minute power series to coarser bins.

    Args:
        signal: 1-minute power series with DatetimeIndex.
        granularity_min: Target bin size in minutes (default 15).
        method: Aggregation method — 'mean', 'max', 'min', 'median'.

    Returns:
        Resampled series. NaN-only bins remain NaN.
    """
    if method not in RESAMPLE_METHODS:
        raise ValueError(f"method must be one of {RESAMPLE_METHODS}, got {method!r}")
    rule = f"{granularity_min}min"
    resampled = getattr(signal.resample(rule), method)()
    return resampled


def _min_on_slots(dev: DeviceProfile, granularity_min: int) -> int:
    """Minimum ON timesteps for device, rounded up from dur_min_min."""
    if dev.dur_min_min is None or dev.dur_min_min <= 0:
        return 1
    return max(1, math.ceil(dev.dur_min_min / granularity_min))


def _max_on_slots(dev: DeviceProfile, granularity_min: int) -> int | None:
    """Maximum consecutive ON timesteps, derived from dur_typical_min * factor."""
    if dev.dur_typical_min is None or dev.dur_typical_min <= 0:
        return None
    return max(1, math.ceil(dev.dur_typical_min * _MAX_ON_FACTOR / granularity_min))


def _allowed_windows(dev: DeviceProfile) -> list[tuple[int, int]] | None:
    """Return (start_min, end_min) windows from the device profile, or None."""
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
    verbose: bool = False,
) -> dict[str, pd.Series]:
    """Disaggregate day by day on a 15-minute resampled signal.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: Device profiles for the household.
        time_limit: Gurobi time limit per daily chunk in seconds.
        baseline_mode: 'peak' or 'duty_avg' for always-on devices.
        baseline_method: Baseline estimator (see baseline_load module).
        resample_method: Aggregation method for 15-min resampling.
        granularity_min: Bin size in minutes (default 15).
        verbose: If True, keep Gurobi console output.

    Returns:
        dict mapping device name -> power series aligned to signal.index (1-min).
    """
    nan_mask = signal.isna()

    # Initialise output at 1-min granularity (zero, NaN where signal is NaN)
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index, dtype=float)
        s[nan_mask] = np.nan
        result[dev.name] = s

    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

    # --- Always-on baseline (computed on 1-min signal, applied at 1-min) ----
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

    # --- Residual at 1-min then resample to coarser granularity -------------
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

    # --- Build per-device constraint dicts ----------------------------------
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

    # --- Daywise solve on resampled residual --------------------------------
    day_periods = residual.index.to_period("D")
    unique_days = day_periods.unique().sort_values()

    per_device_chunks: dict[str, list[pd.Series]] = {dev.name: [] for dev in present_events}

    for day_idx, day in enumerate(unique_days):
        day_mask = day_periods == day
        day_signal = residual[day_mask]
        if day_signal.dropna().empty:
            continue

        if day_idx == 5:
            break

        day_result, info = constrained_v3(
            signal=day_signal,
            devices=present_events,
            time_limit=time_limit,
            verbose=verbose,
            min_on_slots_by_device=min_on or None,
            max_consecutive_on_by_device=max_on or None,
            allowed_on_windows_by_device=allowed_wins or None,
        )

        if verbose:
            print(
                f"  {day}: status={info['status']} optimal={info['is_optimal']} "
                f"obj={info['objective_value']:.1f} t={info['runtime_sec']:.1f}s"
            )

        for dev in present_events:
            chunk = day_result.get(dev.name, pd.Series(0.0, index=day_signal.index))
            per_device_chunks[dev.name].append(chunk.astype(float))

    # --- Combine chunks and upsample back to 1-min --------------------------
    for dev in present_events:
        chunks = per_device_chunks[dev.name]
        if chunks:
            combined_coarse = pd.concat(chunks).sort_index()
        else:
            combined_coarse = pd.Series(0.0, index=residual.index, dtype=float)

        # Forward-fill each coarse bin value across its 1-min sub-slots
        combined_1min = combined_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        combined_1min[nan_mask] = np.nan
        result[dev.name] = combined_1min

    return result
