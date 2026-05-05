"""fhmm_1_survey - FHMM-1 variant with questionnaire-aware temporal priors.

Extends fhmm_1 with lightweight heuristics from the V2 questionnaire:
1. Present always-on devices (main fridge, secondary fridge, freezer) remain on.
2. Preferred start windows and active months bias the activation threshold.
3. Survey duration guides the commitment window and optional max block length.
4. Weekly frequency caps the number of activation blocks over the observed horizon.
5. Daily hours caps total on-minutes per day when that survey field exists.
"""

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile


_BURSTY_DUR_THRESHOLD_MIN = 5.0
_INSIDE_WINDOW_THRESHOLD_FACTOR = 0.85
_OUTSIDE_WINDOW_THRESHOLD_FACTOR = 1.35
_IN_SEASON_THRESHOLD_FACTOR = 0.85
_OUT_OF_SEASON_THRESHOLD_FACTOR = 1.75


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    max_iter: int = 50,
    tol: float = 1e-3,
    baseline_mode: str = "peak",
) -> dict[str, pd.Series]:
    """Survey-aware FHMM-1 disaggregation."""
    nan_mask = signal.isna().values

    result: dict[str, pd.Series] = {}
    for dev in devices:
        series = pd.Series(0.0, index=signal.index)
        series[nan_mask] = np.nan
        result[dev.name] = series

    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

    baseline = 0.0
    for dev in present_always_on:
        if baseline_mode == "duty_avg" and dev.duty_cycle is not None:
            power_val = dev.p_typical_w * dev.duty_cycle
        else:
            power_val = dev.p_typical_w
        baseline += power_val
        series = pd.Series(power_val, index=signal.index)
        series[nan_mask] = np.nan
        result[dev.name] = series

    if not present_events:
        return result

    sig_vals = signal.fillna(0).values.astype(float)
    residual_ca = np.clip(sig_vals - baseline, 0.0, None)
    minute_of_day = (signal.index.hour * 60 + signal.index.minute).to_numpy(dtype=int)
    month_of_year = signal.index.month.to_numpy(dtype=int)

    t_steps = len(signal)
    n_dev = len(present_events)
    p_typicals = np.array([d.p_typical_w for d in present_events], dtype=float)
    states = np.zeros((t_steps, n_dev), dtype=np.float64)
    tol_count = tol * t_steps

    for _iter in range(max_iter):
        total_changes = 0
        total_power = states @ p_typicals
        for i, dev in enumerate(present_events):
            others = total_power - states[:, i] * p_typicals[i]
            dev_residual = residual_ca - others
            threshold = (p_typicals[i] / 2.0) * _activation_threshold_modifiers(
                dev,
                minute_of_day,
                month_of_year,
            )
            new_state_i = (dev_residual > threshold).astype(np.float64)
            changes = int(np.sum(new_state_i != states[:, i]))
            total_changes += changes
            total_power += (new_state_i - states[:, i]) * p_typicals[i]
            states[:, i] = new_state_i
        if total_changes < tol_count:
            break

    for i, dev in enumerate(present_events):
        device_state = _apply_commitment_window(states[:, i], _commitment_length(dev))

        max_duration = _max_block_duration(dev)
        if max_duration is not None:
            device_state = _cap_block_duration(device_state, max_duration)

        max_blocks = _max_blocks_for_horizon(dev, signal.index)
        if max_blocks is not None:
            device_state = _limit_on_blocks(device_state, max_blocks, residual_ca)

        daily_cap = _daily_minutes_cap(dev)
        if daily_cap is not None:
            device_state = _cap_daily_on_minutes(
                device_state,
                signal.index,
                daily_cap,
                residual_ca,
            )

        states[:, i] = device_state

    remaining = np.maximum(0.0, sig_vals - baseline)
    sort_order = np.argsort(p_typicals)[::-1]
    assigned = np.zeros((t_steps, n_dev), dtype=np.float64)

    for i in sort_order:
        dev = present_events[i]
        on_mask = states[:, i] == 1.0
        if not np.any(on_mask):
            continue

        bursty = dev.dur_typical_min <= _BURSTY_DUR_THRESHOLD_MIN
        if bursty:
            assigned[on_mask, i] = dev.p_typical_w
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - dev.p_typical_w)
        else:
            p_assigned = np.clip(remaining, 0.0, dev.p_max_w)
            assigned[on_mask, i] = p_assigned[on_mask]
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - p_assigned[on_mask])

    for i, dev in enumerate(present_events):
        series = pd.Series(assigned[:, i], index=signal.index)
        series[nan_mask] = np.nan
        result[dev.name] = series

    return result


def _activation_threshold_modifiers(
    dev: DeviceProfile,
    minute_of_day: np.ndarray,
    month_of_year: np.ndarray,
) -> np.ndarray:
    factors = np.ones_like(minute_of_day, dtype=float)

    if dev.start_window_start_min is not None and dev.start_window_end_min is not None:
        in_window = _is_in_window(
            minute_of_day,
            dev.start_window_start_min,
            dev.start_window_end_min,
            dev.start_window_wraps_next_day,
        )
        factors[in_window] *= _INSIDE_WINDOW_THRESHOLD_FACTOR
        factors[~in_window] *= _OUTSIDE_WINDOW_THRESHOLD_FACTOR

    if dev.active_months:
        active_mask = np.isin(month_of_year, np.asarray(dev.active_months, dtype=int))
        factors[active_mask] *= _IN_SEASON_THRESHOLD_FACTOR
        factors[~active_mask] *= _OUT_OF_SEASON_THRESHOLD_FACTOR

    return factors


def _is_in_window(
    minute_of_day: np.ndarray,
    start_min: int,
    end_min: int,
    wraps_next_day: bool,
) -> np.ndarray:
    if wraps_next_day or end_min <= start_min:
        return (minute_of_day >= start_min) | (minute_of_day < end_min)
    return (minute_of_day >= start_min) & (minute_of_day < end_min)


def _commitment_length(dev: DeviceProfile) -> int:
    return max(1, int(round(dev.dur_typical_min)))


def _max_block_duration(dev: DeviceProfile) -> int | None:
    if dev.duration_minutes_max is None:
        return None
    return max(1, int(round(dev.duration_minutes_max)))


def _max_blocks_for_horizon(dev: DeviceProfile, index: pd.DatetimeIndex) -> int | None:
    if dev.frequency_per_week_max is None:
        return None
    observed_days = max(1, len(index.normalize().unique()))
    return max(0, int(np.ceil(dev.frequency_per_week_max * observed_days / 7.0)))


def _daily_minutes_cap(dev: DeviceProfile) -> int | None:
    if dev.daily_usage_hours_max is None:
        return None
    return max(0, int(round(dev.daily_usage_hours_max * 60.0)))


def _find_on_blocks(arr: np.ndarray) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    idx = 0
    n = len(arr)
    while idx < n:
        if arr[idx] != 1.0:
            idx += 1
            continue
        start = idx
        while idx < n and arr[idx] == 1.0:
            idx += 1
        blocks.append((start, idx))
    return blocks


def _apply_commitment_window(arr: np.ndarray, commitment_len: int) -> np.ndarray:
    if commitment_len <= 1:
        return arr.copy()

    out = np.zeros_like(arr)
    commit_end = -1
    n = len(arr)
    for idx in range(n):
        if arr[idx] == 1.0:
            commit_end = max(commit_end, min(n, idx + commitment_len))
        if idx < commit_end:
            out[idx] = 1.0
    return out


def _cap_block_duration(arr: np.ndarray, max_duration: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for start, end in _find_on_blocks(arr):
        out[start:min(end, start + max_duration)] = 1.0
    return out


def _limit_on_blocks(
    arr: np.ndarray,
    max_blocks: int,
    score_signal: np.ndarray,
) -> np.ndarray:
    if max_blocks < 0:
        return arr.copy()
    blocks = _find_on_blocks(arr)
    if len(blocks) <= max_blocks:
        return arr.copy()
    if max_blocks == 0:
        return np.zeros_like(arr)

    scores = np.array([
        float(np.sum(score_signal[start:end]))
        for start, end in blocks
    ])
    keep_ids = set(np.argsort(scores)[-max_blocks:])
    out = np.zeros_like(arr)
    for idx, (start, end) in enumerate(blocks):
        if idx in keep_ids:
            out[start:end] = 1.0
    return out


def _cap_daily_on_minutes(
    arr: np.ndarray,
    index: pd.DatetimeIndex,
    max_minutes: int,
    score_signal: np.ndarray,
) -> np.ndarray:
    if max_minutes < 0:
        return arr.copy()

    out = arr.copy()
    days = index.normalize()
    for day in days.unique():
        day_positions = np.flatnonzero(days == day)
        day_arr = out[day_positions]
        if int(np.sum(day_arr)) <= max_minutes:
            continue

        blocks = _find_on_blocks(day_arr)
        scored_blocks = []
        for start, end in blocks:
            global_start = day_positions[start]
            global_end = day_positions[end - 1] + 1
            scored_blocks.append(
                (
                    float(np.sum(score_signal[global_start:global_end])),
                    start,
                    end,
                )
            )

        scored_blocks.sort(reverse=True)
        trimmed_day = np.zeros_like(day_arr)
        remaining = max_minutes
        for _score, start, end in scored_blocks:
            if remaining <= 0:
                break
            keep_len = min(end - start, remaining)
            trimmed_day[start:start + keep_len] = 1.0
            remaining -= keep_len

        out[day_positions] = trimmed_day

    return out