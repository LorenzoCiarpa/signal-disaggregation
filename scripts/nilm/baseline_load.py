from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import stumpy

    _HAS_STUMPY = True
except ImportError:
    _HAS_STUMPY = False

from scripts.nilm.devices import DeviceProfile

logger = logging.getLogger(__name__)

BASELINE_METHODS = (
    "constant_expected",
    "signal_quantile",
    "cyclic_median",
    "stumpy_motif",
)


def _always_on_reference_power(device: DeviceProfile, baseline_mode: str) -> float:
    if baseline_mode == "duty_avg" and device.duty_cycle is not None:
        return float(device.p_typical_w * device.duty_cycle)
    return float(device.p_typical_w)


def _present_always_on_devices(devices: list[DeviceProfile]) -> list[DeviceProfile]:
    return [device for device in devices if device.always_on and device.prior_weight >= 1.0]


def _expected_cycle_window_minutes(devices: list[DeviceProfile]) -> int:
    periods: list[float] = []
    for device in devices:
        if device.duty_cycle is not None and device.duty_cycle > 0:
            periods.append(device.dur_typical_min / device.duty_cycle)
        else:
            periods.append(device.dur_typical_min * 3.0)
    if not periods:
        return 90
    return int(np.clip(np.median(periods), 15.0, 240.0))


def _global_quantile_floor(signal: pd.Series, quantile: float) -> float:
    valid = signal.dropna()
    if valid.empty:
        return 0.0
    return float(valid.quantile(quantile))


def _rolling_quantile_floor(
    signal: pd.Series,
    quantile: float,
    window_minutes: int,
) -> pd.Series:
    global_floor = _global_quantile_floor(signal, quantile)
    min_periods = max(5, window_minutes // 4)
    floor = signal.rolling(
        window=window_minutes,
        center=True,
        min_periods=min_periods,
    ).quantile(quantile)
    floor = floor.fillna(global_floor)
    floor = floor.rolling(window=15, center=True, min_periods=1).median()
    return floor.clip(lower=0.0)


def _finalize_baseline(
    signal: pd.Series,
    baseline: pd.Series,
    floor: pd.Series,
    peak_total: float,
) -> pd.Series:
    upper = pd.Series(float(max(peak_total, 0.0)), index=signal.index, dtype=float)
    baseline = baseline.clip(lower=floor, upper=upper)

    signal_cap = signal.fillna(np.inf).astype(float)
    baseline = pd.Series(
        np.minimum(baseline.to_numpy(dtype=float), signal_cap.to_numpy(dtype=float)),
        index=signal.index,
        dtype=float,
    )
    baseline[signal.isna()] = np.nan
    return baseline.clip(lower=0.0)


def _estimate_constant_expected(
    signal: pd.Series,
    devices: list[DeviceProfile],
    baseline_mode: str,
) -> tuple[pd.Series, dict[str, float | int | bool | str | None]]:
    constant_total = sum(_always_on_reference_power(device, baseline_mode) for device in devices)
    baseline = pd.Series(float(constant_total), index=signal.index, dtype=float)
    baseline[signal.isna()] = np.nan
    return baseline, {
        "method": "constant_expected",
        "peak_total_w": float(sum(device.p_typical_w for device in devices)),
        "estimated_mean_w": float(constant_total),
        "fallback_used": False,
    }


def _estimate_signal_quantile(
    signal: pd.Series,
    devices: list[DeviceProfile],
    quantile: float,
) -> tuple[pd.Series, dict[str, float | int | bool | str | None]]:
    peak_total = float(sum(device.p_typical_w for device in devices))
    level = _global_quantile_floor(signal, quantile)
    baseline = pd.Series(min(level, peak_total), index=signal.index, dtype=float)
    baseline[signal.isna()] = np.nan
    return baseline, {
        "method": "signal_quantile",
        "peak_total_w": peak_total,
        "estimated_mean_w": float(baseline.dropna().mean()) if baseline.notna().any() else 0.0,
        "fallback_used": False,
    }


def _estimate_cyclic_median(
    signal: pd.Series,
    devices: list[DeviceProfile],
    quantile: float,
    cycle_window_minutes: int | None,
) -> tuple[pd.Series, dict[str, float | int | bool | str | None]]:
    peak_total = float(sum(device.p_typical_w for device in devices))
    floor = _rolling_quantile_floor(signal, quantile=quantile, window_minutes=6 * 60)
    cycle_window = cycle_window_minutes or _expected_cycle_window_minutes(devices)
    plausible_limit = floor + peak_total * 1.10
    candidate = signal.where(signal <= plausible_limit)

    min_periods = max(5, cycle_window // 3)
    baseline = candidate.rolling(
        window=cycle_window,
        center=True,
        min_periods=min_periods,
    ).median()
    baseline = baseline.interpolate(limit_direction="both")
    baseline = baseline.fillna(floor)
    baseline = baseline.rolling(window=7, center=True, min_periods=1).mean()
    baseline = _finalize_baseline(signal, baseline, floor, peak_total)
    return baseline, {
        "method": "cyclic_median",
        "peak_total_w": peak_total,
        "cycle_window_min": int(cycle_window),
        "estimated_mean_w": float(baseline.dropna().mean()) if baseline.notna().any() else 0.0,
        "fallback_used": False,
    }


def _normalize_matches(matches: object) -> np.ndarray:
    if matches is None:
        return np.empty((0, 2), dtype=float)
    array = np.asarray(matches)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim == 1:
        if array.shape[0] < 2:
            return np.empty((0, 2), dtype=float)
        array = array.reshape(1, -1)
    return array[:, :2].astype(float, copy=False)


def _estimate_stumpy_motif(
    signal: pd.Series,
    devices: list[DeviceProfile],
    quantile: float,
    motif_window_minutes: int | None,
) -> tuple[pd.Series, dict[str, float | int | bool | str | None]]:
    fallback_baseline, fallback_info = _estimate_cyclic_median(
        signal=signal,
        devices=devices,
        quantile=quantile,
        cycle_window_minutes=motif_window_minutes,
    )

    if not _HAS_STUMPY:
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "stumpy_not_installed"
        return fallback_baseline, fallback_info

    peak_total = float(sum(device.p_typical_w for device in devices))
    floor = _rolling_quantile_floor(signal, quantile=quantile, window_minutes=6 * 60)
    motif_window = motif_window_minutes or _expected_cycle_window_minutes(devices)
    if len(signal) < motif_window * 2:
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "signal_too_short"
        return fallback_baseline, fallback_info

    plausible_limit = floor + peak_total * 1.10
    candidate = signal.where(signal <= plausible_limit)
    candidate = candidate.interpolate(limit_direction="both")
    candidate = candidate.fillna(floor)
    cycle_candidate = np.clip(
        candidate.to_numpy(dtype=float) - floor.to_numpy(dtype=float),
        0.0,
        peak_total,
    )
    cycle_candidate = pd.Series(cycle_candidate, index=signal.index, dtype=float)
    cycle_candidate = cycle_candidate.rolling(window=5, center=True, min_periods=1).median()
    cycle_array = cycle_candidate.to_numpy(dtype=np.float64)

    reference_len = min(len(cycle_array), max(motif_window * 4, 3 * 24 * 60))
    reference_array = cycle_array[:reference_len]
    if len(reference_array) <= motif_window:
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "reference_too_short"
        return fallback_baseline, fallback_info

    try:
        profile = stumpy.stump(reference_array, m=motif_window)
    except Exception as exc:  # noqa: BLE001
        logger.warning("STUMPY motif estimation failed during matrix profile: %s", exc)
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "matrix_profile_error"
        return fallback_baseline, fallback_info

    profile_values = np.asarray(profile[:, 0], dtype=float)
    neighbor_indices = np.asarray(profile[:, 1], dtype=float)
    valid = np.isfinite(profile_values) & np.isfinite(neighbor_indices) & (neighbor_indices >= 0)
    if not valid.any():
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "no_valid_motif"
        return fallback_baseline, fallback_info

    valid_indices = np.flatnonzero(valid)
    motif_anchor = int(valid_indices[np.argmin(profile_values[valid])])
    motif_neighbor = int(neighbor_indices[motif_anchor])
    template_segments = [
        reference_array[motif_anchor:motif_anchor + motif_window],
        reference_array[motif_neighbor:motif_neighbor + motif_window],
    ]
    template = np.nanmedian(np.vstack(template_segments), axis=0)

    try:
        matches = _normalize_matches(stumpy.match(template.astype(np.float64), cycle_array))
    except Exception as exc:  # noqa: BLE001
        logger.warning("STUMPY motif estimation failed during matching: %s", exc)
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "match_error"
        return fallback_baseline, fallback_info

    if matches.shape[0] == 0:
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "no_matches"
        return fallback_baseline, fallback_info

    distances = matches[:, 0]
    match_indices = matches[:, 1].astype(int, copy=False)
    finite_mask = np.isfinite(distances)
    if not finite_mask.any():
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "no_finite_match"
        return fallback_baseline, fallback_info

    distances = distances[finite_mask]
    match_indices = match_indices[finite_mask]
    distance_cap = float(np.quantile(distances, 0.25)) if len(distances) > 1 else float(distances[0])
    distance_cap = max(distance_cap, 1e-6)

    motif_component = np.zeros(len(cycle_array), dtype=float)
    occupied = np.zeros(len(cycle_array), dtype=bool)
    match_count = 0
    for order_idx in np.argsort(distances):
        distance = float(distances[order_idx])
        start = int(match_indices[order_idx])
        end = min(len(cycle_array), start + motif_window)
        if distance > distance_cap * 1.5:
            continue
        if occupied[start:end].mean() > 0.5:
            continue
        motif_component[start:end] = np.maximum(
            motif_component[start:end],
            template[: end - start],
        )
        occupied[start:end] = True
        match_count += 1

    if match_count == 0:
        fallback_info = dict(fallback_info)
        fallback_info["method"] = "stumpy_motif"
        fallback_info["fallback_used"] = True
        fallback_info["fallback_reason"] = "all_matches_filtered"
        return fallback_baseline, fallback_info

    fallback_component = np.clip(
        fallback_baseline.fillna(0.0).to_numpy(dtype=float) - floor.fillna(0.0).to_numpy(dtype=float),
        0.0,
        peak_total,
    )
    motif_series = pd.Series(motif_component, index=signal.index, dtype=float)
    motif_series = motif_series.replace(0.0, np.nan)
    motif_series = motif_series.interpolate(limit_direction="both")
    motif_series = motif_series.fillna(pd.Series(fallback_component, index=signal.index, dtype=float))
    baseline = floor + motif_series
    baseline = _finalize_baseline(signal, baseline, floor, peak_total)
    return baseline, {
        "method": "stumpy_motif",
        "peak_total_w": peak_total,
        "motif_window_min": int(motif_window),
        "motif_anchor_idx": motif_anchor,
        "motif_neighbor_idx": motif_neighbor,
        "motif_matches": match_count,
        "estimated_mean_w": float(baseline.dropna().mean()) if baseline.notna().any() else 0.0,
        "fallback_used": False,
    }


def estimate_always_on_baseline(
    signal: pd.Series,
    devices: list[DeviceProfile],
    method: str = "cyclic_median",
    baseline_mode: str = "duty_avg",
    quantile: float = 0.08,
    cycle_window_minutes: int | None = None,
    motif_window_minutes: int | None = None,
) -> tuple[pd.Series, dict[str, float | int | bool | str | None]]:
    """Estimate the aggregate always-on baseline for fridge/freezer loads.

    The returned series represents the combined base load to subtract from the
    aggregate signal before solving the event-driven residual.
    """
    if method not in BASELINE_METHODS:
        raise ValueError(f"method must be one of: {', '.join(BASELINE_METHODS)}")
    if baseline_mode not in {"peak", "duty_avg"}:
        raise ValueError("baseline_mode must be one of: 'peak', 'duty_avg'")

    always_on_devices = _present_always_on_devices(devices)
    zero = pd.Series(0.0, index=signal.index, dtype=float)
    zero[signal.isna()] = np.nan

    if not always_on_devices:
        return zero, {
            "method": method,
            "peak_total_w": 0.0,
            "estimated_mean_w": 0.0,
            "fallback_used": False,
        }

    if method == "constant_expected":
        return _estimate_constant_expected(signal, always_on_devices, baseline_mode)
    if method == "signal_quantile":
        return _estimate_signal_quantile(signal, always_on_devices, quantile)
    if method == "cyclic_median":
        return _estimate_cyclic_median(signal, always_on_devices, quantile, cycle_window_minutes)
    return _estimate_stumpy_motif(signal, always_on_devices, quantile, motif_window_minutes)


def split_baseline_by_device(
    baseline_total: pd.Series,
    devices: list[DeviceProfile],
    baseline_mode: str = "duty_avg",
) -> dict[str, pd.Series]:
    """Distribute an aggregate always-on baseline across present devices."""
    always_on_devices = _present_always_on_devices(devices)
    if not always_on_devices:
        return {}

    weights = np.asarray(
        [_always_on_reference_power(device, baseline_mode) for device in always_on_devices],
        dtype=float,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(always_on_devices), dtype=float)
    weights = weights / weights.sum()

    return {
        device.name: pd.Series(
            baseline_total.to_numpy(dtype=float) * weights[idx],
            index=baseline_total.index,
            dtype=float,
        )
        for idx, device in enumerate(always_on_devices)
    }