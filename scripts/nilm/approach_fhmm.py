"""
US-005: Approccio C — Factorial HMM semplificato.

Models all devices simultaneously with independent latent binary chains,
using a greedy coordinate-ascent EM so that simultaneous appliance
combinations are captured without exponential state explosion.
"""

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    max_iter: int = 50,
    tol: float = 1e-3,
) -> dict[str, pd.Series]:
    """Simplified Factorial HMM disaggregation (greedy coordinate-ascent).

    For each timestep, each device's binary state is updated greedily while
    keeping all other device states fixed, iterating until convergence or
    max_iter. A temporal smoothing step removes ON blocks shorter than
    dur_min_min/2 minutes.

    Args:
        signal: 1-minute resampled w_medio series with DatetimeIndex.
        devices: List of DeviceProfile instances.
        max_iter: Maximum EM iterations (default 50).
        tol: Convergence threshold as fraction of signal length (default 1e-3).

    Returns:
        dict mapping device name -> pd.Series with estimated power
        (0 or p_typical_w). NaN where signal is NaN.
    """
    present_devices = [d for d in devices if d.prior_weight >= 1.0]

    # Build full output dict — zero series with NaN where signal is NaN
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[signal.isna()] = np.nan
        result[dev.name] = s

    if not present_devices:
        return result

    T = len(signal)

    # Use signal values; treat NaN as 0 for inference
    sig_vals = signal.fillna(0).values.astype(float)

    n_dev = len(present_devices)
    p_typicals = np.array([d.p_typical_w for d in present_devices], dtype=float)

    # Initialize all device states to 0
    # states shape: (T, n_dev), binary
    states = np.zeros((T, n_dev), dtype=np.float64)

    tol_count = tol * T

    for _iter in range(max_iter):
        total_changes = 0

        for i, dev in enumerate(present_devices):
            # Contribution of all other devices at each timestep
            others_contribution = states @ p_typicals - states[:, i] * p_typicals[i]

            # Residual signal without device i
            residual = sig_vals - others_contribution

            # For each timestep, choose state (0 or 1) that minimises |residual - p_i * x_i|
            # x_i = 1 if |residual - p_i| < |residual - 0|, i.e., if residual > p_i / 2
            new_state_i = (residual > p_typicals[i] / 2.0).astype(np.float64)

            changes = np.sum(new_state_i != states[:, i])
            total_changes += changes
            states[:, i] = new_state_i

        if total_changes < tol_count:
            break

    # Temporal smoothing: remove ON blocks shorter than dur_min_min/2
    for i, dev in enumerate(present_devices):
        min_block = max(1, int(dev.dur_min_min / 2.0))
        col = states[:, i].copy()
        col = _remove_short_blocks(col, min_block)
        states[:, i] = col

    # Build output series from states
    nan_mask = signal.isna().values
    for i, dev in enumerate(present_devices):
        series = pd.Series(states[:, i] * dev.p_typical_w, index=signal.index)
        series[nan_mask] = np.nan
        result[dev.name] = series

    return result


def _remove_short_blocks(arr: np.ndarray, min_len: int) -> np.ndarray:
    """Remove ON (value=1) runs shorter than min_len samples."""
    if min_len <= 1:
        return arr

    out = arr.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 1.0:
            # Find end of this ON block
            j = i
            while j < n and out[j] == 1.0:
                j += 1
            block_len = j - i
            if block_len < min_len:
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    return out


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis, load_imei
    from scripts.nilm.devices import get_device_profiles

    imei = get_usable_imeis()[0]
    signal = load_imei(imei)
    devices = get_device_profiles(imei)
    disagg = run(signal, devices)
    for name, series in disagg.items():
        on_minutes = int((series > 0).sum())
        if on_minutes > 0:
            print(f"  {name}: {on_minutes} minutes ON")
