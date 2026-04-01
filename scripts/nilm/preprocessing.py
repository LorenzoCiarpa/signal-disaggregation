"""
NILM preprocessing module.

Loads, filters, and resamples raw JSON power data for each IMEI.
"""

import json
import os

import pandas as pd

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
    json_path = os.path.join(json_dir, f"iot_devices.{imei}.json")
    with open(json_path, "r") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        # Skip invalid epochs
        if not rec.get("epoch_valid", True):
            continue
        # Skip spikes
        w_medio = rec.get("w_medio", 0)
        if w_medio > 10000:
            continue
        epoch = _parse_id(rec["_id"])
        rows.append((epoch, w_medio))

    if not rows:
        return pd.Series(dtype=float)

    epochs, values = zip(*rows)
    index = pd.to_datetime(list(epochs), unit="s", utc=True)
    series = pd.Series(list(values), index=index, dtype=float, name="w_medio")

    # Sort by time
    series = series.sort_index()

    # Resample to 1-minute frequency; reindex to fill gaps with NaN
    start = series.index.min().floor("1min")
    end = series.index.max().ceil("1min")
    full_index = pd.date_range(start=start, end=end, freq="1min", tz="UTC")

    resampled = series.resample("1min").mean()
    resampled = resampled.reindex(full_index)

    return resampled


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis, load_imei

    imeis = get_usable_imeis()
    print(imeis)
    s = load_imei(imeis[0])
    print(f"Loaded {len(s)} minutes for IMEI {imeis[0]}, NaN: {s.isna().sum()}")
