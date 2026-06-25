"""
NILM preprocessing module.

Loads, filters, and resamples raw JSON power data for each IMEI.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing
from scipy.signal import medfilt, savgol_filter

USABLE_IMEIS = [
    "86853106211162",
    "86684007269866",
    "86853106211179",
    "86684007269887",
    "86853106211173",
    "86684007269889",
]


def get_usable_imeis() -> list[str]:
    """Return the list of IMEIs with >50k records."""
    return USABLE_IMEIS


def _parse_id(value) -> int:
    """Parse _id field which may be an int or {'$numberLong': '...'} dict."""
    if isinstance(value, dict) and "$numberLong" in value:
        return int(value["$numberLong"])
    return int(value)


def _load_resampled_signal(
    imei: str,
    json_dir: str = "json",
    keep_invalid_as_nan: bool = False,
) -> pd.Series:
    """Load raw signal and resample it to 1 minute resolution."""
    json_path = os.path.join(json_dir, f"iot_devices.{imei}.json")
    with open(json_path, "r") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        epoch = _parse_id(rec["_id"])
        w_medio = rec.get("w_medio")
        is_valid = rec.get("epoch_valid", True) and pd.notna(w_medio) and w_medio <= 10000

        if is_valid:
            rows.append((epoch, float(w_medio)))
            continue

        if keep_invalid_as_nan:
            rows.append((epoch, float("nan")))

    if not rows:
        return pd.Series(dtype=float)

    epochs, values = zip(*rows)
    index = pd.to_datetime(list(epochs), unit="s", utc=True)
    series = pd.Series(list(values), index=index, dtype=float, name="w_medio")
    series = series.sort_index()

    start = series.index.min().floor("1min")
    end = series.index.max().ceil("1min")
    full_index = pd.date_range(start=start, end=end, freq="1min", tz="UTC")

    resampled = series.resample("1min").mean()
    return resampled.reindex(full_index)


def _collect_neighbor_values(
    values: list[float],
    index: int,
    neighbors_per_side: int = 3,
) -> list[float]:
    """Collect the nearest valid samples before and after a missing point."""
    left_values = []
    cursor = index - 1
    while cursor >= 0 and len(left_values) < neighbors_per_side:
        value = values[cursor]
        if pd.notna(value):
            left_values.append(float(value))
        cursor -= 1

    right_values = []
    cursor = index + 1
    while cursor < len(values) and len(right_values) < neighbors_per_side:
        value = values[cursor]
        if pd.notna(value):
            right_values.append(float(value))
        cursor += 1

    left_values.reverse()
    return left_values + right_values


def _get_replacement_value(neighbors: list[float], method: str) -> float:
    """Compute the replacement value from surrounding valid samples."""
    if not neighbors:
        return float("nan")

    if method == "media":
        return float(sum(neighbors) / len(neighbors))

    if method == "moda":
        modes = pd.Series(neighbors, dtype=float).mode(dropna=True)
        if not modes.empty:
            return float(modes.iloc[0])
        return float(pd.Series(neighbors, dtype=float).median())

    raise ValueError("method must be either 'media' or 'moda'")


def _fill_missing_values(
    series: pd.Series,
    method: str,
    neighbors_per_side: int = 3,
) -> pd.Series:
    """Fill NaN gaps using neighboring valid samples."""
    if method not in {"media", "moda"}:
        raise ValueError("method must be either 'media' or 'moda'")

    original_values = series.astype(float).tolist()
    filled = series.copy()

    for idx, value in enumerate(original_values):
        if pd.notna(value):
            continue

        neighbors = _collect_neighbor_values(
            original_values,
            idx,
            neighbors_per_side=neighbors_per_side,
        )
        filled.iloc[idx] = _get_replacement_value(neighbors, method)

    return filled


def _apply_plateau_smoothing(
    series: pd.Series,
    method: str,
    window_size: int = 7,
    polyorder: int = 2,
) -> pd.Series:
    """Flatten local oscillations while preserving the plateau level."""
    if method not in {"savgol", "mediana", "plateau_max"}:
        raise ValueError("method must be either 'savgol', 'mediana', or 'plateau_max'")

    if series.empty:
        return series.copy()

    if method == "plateau_max":
        return _apply_high_plateau_envelope(series, window_size=window_size)

    valid_series = series.dropna()
    if valid_series.empty:
        return series.copy()

    max_window = len(valid_series)
    if max_window % 2 == 0:
        max_window -= 1

    if max_window < 3:
        return series.copy()

    adjusted_window = min(window_size, max_window)
    if adjusted_window % 2 == 0:
        adjusted_window -= 1

    if adjusted_window < 3:
        adjusted_window = 3
    if adjusted_window > max_window:
        adjusted_window = max_window

    values = valid_series.to_numpy(dtype=float)

    if method == "savgol":
        adjusted_polyorder = min(polyorder, adjusted_window - 1)
        smoothed_values = savgol_filter(
            values,
            window_length=adjusted_window,
            polyorder=adjusted_polyorder,
            mode="interp",
        )
    else:
        smoothed_values = medfilt(values, kernel_size=adjusted_window)

    smoothed = series.copy()
    smoothed.loc[valid_series.index] = smoothed_values
    return smoothed


def _apply_high_plateau_envelope(
    series: pd.Series,
    window_size: int = 11,
    activation_threshold_w: float = 800.0,
    min_duration_minutes: int = 20,
    gap_tolerance_minutes: int = 6,
    quantile_level: float = 0.9,
    min_range_w: float = 700.0,
    duty_cycle_threshold: float = 0.25,
) -> pd.Series:
    """Flatten pulsed high-load windows to a robust near-maximum plateau."""
    if series.empty:
        return series.copy()

    valid_series = series.dropna()
    if valid_series.empty:
        return series.copy()

    adjusted_window = max(3, window_size)
    if adjusted_window % 2 == 0:
        adjusted_window += 1

    high_mask = valid_series >= activation_threshold_w
    rolling_high_fraction = (
        high_mask.astype(float)
        .rolling(adjusted_window, center=True, min_periods=1)
        .mean()
    )
    rolling_max = valid_series.rolling(adjusted_window, center=True, min_periods=1).max()

    candidate_mask = (
        (rolling_max >= activation_threshold_w)
        & (rolling_high_fraction >= duty_cycle_threshold)
    )

    closed_mask = binary_closing(
        candidate_mask.to_numpy(dtype=bool),
        structure=np.ones(gap_tolerance_minutes + 1, dtype=bool),
    )
    closed_mask = pd.Series(closed_mask, index=valid_series.index)

    smoothed = series.copy()
    groups = (closed_mask != closed_mask.shift(fill_value=False)).cumsum()

    for _, segment_mask in closed_mask.groupby(groups):
        if not bool(segment_mask.iloc[0]):
            continue

        segment_index = segment_mask.index
        if len(segment_index) < min_duration_minutes:
            continue

        segment_values = valid_series.loc[segment_index]
        if segment_values.max() < activation_threshold_w:
            continue

        if (segment_values.max() - segment_values.min()) < min_range_w:
            continue

        segment_level = float(np.nanquantile(segment_values.to_numpy(dtype=float), quantile_level))
        smoothed.loc[segment_index] = segment_level

    return smoothed


def load_imei(imei: str, json_dir: str = "json") -> pd.Series:
    """
    Load, filter, and resample power data for a given IMEI.

    Parameters
    ----------
    imei:
        The IMEI identifier for the device.
    json_dir:
        Directory containing the JSON files (default: 'json').

    Returns
    -------
    pd.Series
        1-minute resampled w_medio series with DatetimeIndex (UTC).
        NaN gaps are preserved.
    """
    return _load_resampled_signal(imei, json_dir=json_dir, keep_invalid_as_nan=False)


def loade_imei_v2(imei: str, method: str, json_dir: str = "json") -> pd.Series:
    """
    Load and clean power data, imputing invalid samples with mean or mode.

    Parameters
    ----------
    imei:
        The IMEI identifier for the device.
    method:
        Imputation strategy: 'media' or 'moda'.
    json_dir:
        Directory containing the JSON files (default: 'json').

    Returns
    -------
    pd.Series
        1-minute resampled w_medio series with invalid points imputed.
    """
    signal = _load_resampled_signal(imei, json_dir=json_dir, keep_invalid_as_nan=True)
    return _fill_missing_values(signal, method=method)


load_imei_v2 = loade_imei_v2


def load_imei_aggregated(
    imei: str,
    fill_method: str = "media",
    aggregation_method: str = "savgol",
    json_dir: str = "json",
    window_size: int = 7,
    polyorder: int = 2,
) -> pd.Series:
    """
    Load, impute, and flatten oscillating plateaus of the aggregate signal.

    Parameters
    ----------
    imei:
        The IMEI identifier for the device.
    fill_method:
        Missing-value strategy: 'media' or 'moda'.
    aggregation_method:
        Plateau smoothing strategy: 'savgol', 'mediana', or 'plateau_max'.
    json_dir:
        Directory containing the JSON files (default: 'json').
    window_size:
        Odd smoothing window in minutes.
    polyorder:
        Polynomial order used by the Savitzky-Golay method.

    Returns
    -------
    pd.Series
        1-minute series with invalid samples imputed and plateaus flattened.
    """
    signal = loade_imei_v2(imei, method=fill_method, json_dir=json_dir)
    return _apply_plateau_smoothing(
        signal,
        method=aggregation_method,
        window_size=window_size,
        polyorder=polyorder,
    )


def load_imei_plateau_median(
    imei: str,
    fill_method: str = "media",
    json_dir: str = "json",
    plateau_window_size: int = 11,
    median_window_size: int = 7,
) -> pd.Series:
    """Load signal with imputation, plateau flattening, and median cleanup."""
    plateau_signal = load_imei_aggregated(
        imei,
        fill_method=fill_method,
        aggregation_method="plateau_max",
        json_dir=json_dir,
        window_size=plateau_window_size,
    )
    return _apply_plateau_smoothing(
        plateau_signal,
        method="mediana",
        window_size=median_window_size,
    )


def _plot_daily_signal(
    raw_signal: pd.Series,
    cleaned_signal: pd.Series,
    imei: str,
    method: str,
    plot_root: str = "plots",
) -> None:
    """Save one plot per day comparing raw and imputed signal."""
    output_dir = os.path.join(plot_root, method)
    os.makedirs(output_dir, exist_ok=True)

    for day_start, day_signal in cleaned_signal.groupby(pd.Grouper(freq="1D")):
        if day_signal.empty:
            continue

        raw_day = raw_signal.reindex(day_signal.index)
        if day_signal.isna().all() and raw_day.isna().all():
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(raw_day.index, raw_day.values, label="raw_with_nan", alpha=0.5, linewidth=1.0)
        ax.plot(day_signal.index, day_signal.values, label=f"imputed_{method}", linewidth=1.2)
        ax.set_title(f"IMEI {imei} - {day_start.strftime('%Y-%m-%d')} ({method})")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Power [W]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"{imei}_{day_start.strftime('%Y-%m-%d')}.png"),
            dpi=150,
        )
        plt.close(fig)


def _plot_daily_aggregated_signal(
    imputed_signal: pd.Series,
    aggregated_signal: pd.Series,
    imei: str,
    method: str,
    plot_root: str = "plots",
) -> None:
    """Save one plot per day comparing imputed and plateau-smoothed signal."""
    output_dir = os.path.join(plot_root, "aggregated", method)
    os.makedirs(output_dir, exist_ok=True)

    for day_start, day_signal in aggregated_signal.groupby(pd.Grouper(freq="1D")):
        if day_signal.empty:
            continue

        imputed_day = imputed_signal.reindex(day_signal.index)
        if day_signal.isna().all() and imputed_day.isna().all():
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(
            imputed_day.index,
            imputed_day.values,
            label="imputed_media",
            alpha=0.55,
            linewidth=1.0,
        )
        ax.plot(
            day_signal.index,
            day_signal.values,
            label=f"aggregated_{method}",
            linewidth=1.3,
        )
        delta = np.nanmean(np.abs(imputed_day.values - day_signal.values))
        ax.set_title(
            f"IMEI {imei} - {day_start.strftime('%Y-%m-%d')} ({method}, delta={delta:.1f}W)"
        )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Power [W]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"{imei}_{day_start.strftime('%Y-%m-%d')}.png"),
            dpi=150,
        )
        plt.close(fig)


if __name__ == "__main__":
    imeis = get_usable_imeis()
    print(imeis)
    target_imei = imeis[0]

    raw_signal = _load_resampled_signal(target_imei, keep_invalid_as_nan=True)
    print(
        f"Loaded {len(raw_signal)} minutes for IMEI {target_imei}, "
        f"NaN before imputation: {raw_signal.isna().sum()}"
    )

    for method in ("media", "moda"):
        signal = loade_imei_v2(target_imei, method=method)
        print(
            f"Method {method}: NaN after imputation = {signal.isna().sum()}"
        )
        _plot_daily_signal(raw_signal, signal, target_imei, method=method)

    imputed_signal = loade_imei_v2(target_imei, method="media")
    for aggregation_method in ("savgol", "mediana", "plateau_max"):
        aggregated_signal = load_imei_aggregated(
            target_imei,
            fill_method="media",
            aggregation_method=aggregation_method,
        )
        mean_delta = np.nanmean(
            np.abs(imputed_signal.to_numpy(dtype=float) - aggregated_signal.to_numpy(dtype=float))
        )
        print(
            f"Aggregation {aggregation_method}: mean abs delta vs imputed = {mean_delta:.2f}W"
        )
        _plot_daily_aggregated_signal(
            imputed_signal,
            aggregated_signal,
            target_imei,
            method=aggregation_method,
        )
