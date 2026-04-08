"""
US-006: Approccio D — Template matching.

Finds device activations by cross-correlating the aggregate signal with
rectangular templates built from the knowledge base (power × duration),
without any model fitting.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

from scripts.nilm.devices import DeviceProfile


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    corr_threshold: float = 0.6,
) -> dict[str, pd.Series]:
    """Template matching NILM disaggregation.

    Builds a rectangular template (p_typical_w × dur_typical_min samples) for
    each present device, cross-correlates with the aggregate signal, detects
    peaks above corr_threshold, and marks the corresponding ON windows.

    Args:
        signal: 1-minute resampled w_medio series with DatetimeIndex.
        devices: List of DeviceProfile instances.
        corr_threshold: Normalized correlation threshold for peak detection.

    Returns:
        dict mapping device name -> pd.Series with estimated power (0 or p_typical_w)
        at each timestamp. NaN where signal is NaN.
    """
    # Build zero-filled output for every device (NaN where signal is NaN)
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[signal.isna()] = np.nan
        result[dev.name] = s

    # Working array: fill NaN with 0 for correlation
    sig_values = signal.fillna(0.0).to_numpy(dtype=float)
    n = len(sig_values)

    # Track which timesteps are already assigned (first-come-first-served)
    assigned = np.zeros(n, dtype=bool)

    # Process devices sorted by descending p_typical_w (larger devices first,
    # so their templates produce higher absolute correlation scores)
    present_devices = [d for d in devices if d.prior_weight >= 1.0 and d.dur_typical_min >= 3]
    present_devices_sorted = sorted(present_devices, key=lambda d: d.p_typical_w, reverse=True)

    for dev in present_devices_sorted:
        dur = int(dev.dur_typical_min)
        if dur < 1:
            continue

        # Build rectangular template
        template = np.full(dur, dev.p_typical_w, dtype=float)
        template_norm = np.linalg.norm(template)
        if template_norm == 0:
            continue

        # Normalized cross-correlation (mode='same' keeps output length == n)
        corr = correlate(sig_values, template, mode="same") / template_norm

        # Peak detection: minimum distance between peaks = half the template duration
        min_distance = max(1, int(dur * 0.5))
        peaks, _ = find_peaks(corr, height=corr_threshold, distance=min_distance)

        # Sort peaks by descending correlation score (greedy assignment)
        peaks = sorted(peaks, key=lambda p: corr[p], reverse=True)

        dev_array = result[dev.name].to_numpy(dtype=float).copy()

        for peak_idx in peaks:
            # Center the template window on the peak
            half = dur // 2
            start = peak_idx - half
            end = start + dur
            # Clamp to valid range
            start = max(0, start)
            end = min(n, end)

            window_slice = slice(start, end)

            # Skip if any timestep in this window is already assigned
            if assigned[window_slice].any():
                continue

            # Mark as ON (respect NaN positions)
            nan_mask = signal.isna().to_numpy()
            for t in range(start, end):
                if not nan_mask[t]:
                    dev_array[t] = dev.p_typical_w
            assigned[start:end] = True

        result[dev.name] = pd.Series(dev_array, index=signal.index)

    return result


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis, load_imei
    from scripts.nilm.devices import get_device_profiles

    imei = get_usable_imeis()[0]
    signal = load_imei(imei)
    devices = get_device_profiles(imei)
    disagg = run(signal, devices)
    for name, series in disagg.items():
        on_minutes = (series > 0).sum()
        if on_minutes > 0:
            print(f"  {name}: {on_minutes} minutes ON")
