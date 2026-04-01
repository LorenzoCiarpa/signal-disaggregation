"""
US-007: Approccio E — Event-based con prior pesato da inventario.

Combines delta_w event detection with a Bayesian prior from the device inventory
(present/absent + frequency from survey), so that known devices get higher probability
and absent devices are de-weighted (not eliminated).
"""

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    threshold_w: float = 50.0,
    sigma_factor: float = 0.3,
) -> dict[str, pd.Series]:
    """Event-based NILM disaggregation with Bayesian prior weighting.

    Detects ON/OFF transitions via delta_w, assigns each ON event to the device
    with the highest score = likelihood * prior_weight * frequency_factor.

    Args:
        signal: 1-minute resampled w_medio series with DatetimeIndex.
        devices: List of DeviceProfile instances (from devices.get_device_profiles).
        threshold_w: Minimum |delta_w| to consider an event (default 50 W).
        sigma_factor: Fraction of p_typical_w used as Gaussian std for likelihood
                      (default 0.3).

    Returns:
        dict mapping device name -> pd.Series with estimated power (0 or p_typical_w)
        at each timestamp. NaN where signal is NaN.
    """
    # Build zero-filled output series for each device (NaN where signal is NaN)
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[signal.isna()] = np.nan
        result[dev.name] = s

    # Compute delta_w
    delta_w = signal.diff()

    # Positive events (turn-on): delta_w > threshold
    on_mask = delta_w > threshold_w
    on_indices = signal.index[on_mask].tolist()

    # Negative events (turn-off): delta_w < -threshold
    off_mask = delta_w < -threshold_w
    off_indices = signal.index[off_mask].tolist()

    def score_device(dev: DeviceProfile, abs_delta: float) -> float:
        """Compute Bayesian score for a device given the observed |delta_w|."""
        sigma = sigma_factor * dev.p_typical_w
        if sigma <= 0:
            sigma = 1.0
        # Gaussian likelihood: N(|delta_w|; p_typical_w, sigma^2)
        likelihood = np.exp(-0.5 * ((abs_delta - dev.p_typical_w) / sigma) ** 2)
        frequency_factor = min(dev.frequency_per_week / 7.0, 1.0)
        return likelihood * dev.prior_weight * frequency_factor

    def best_device_for_event(abs_delta: float) -> DeviceProfile:
        """Select device with maximum Bayesian score among all devices."""
        return max(devices, key=lambda d: score_device(d, abs_delta))

    # Process each ON event
    for on_idx in on_indices:
        abs_delta = abs(delta_w[on_idx])
        dev = best_device_for_event(abs_delta)
        max_dur = dev.dur_typical_min * 3  # minutes

        # Find matching OFF event: first off event after on_idx within max_dur
        off_idx = None
        for candidate_off in off_indices:
            if candidate_off <= on_idx:
                continue
            minutes_apart = (candidate_off - on_idx).total_seconds() / 60.0
            if minutes_apart <= max_dur:
                off_idx = candidate_off
                break

        if off_idx is None:
            # No matching off event: use dur_typical_min as default duration
            off_idx = on_idx + pd.Timedelta(minutes=dev.dur_typical_min)

        # Mark the device as ON between on_idx and off_idx
        on_period = (signal.index >= on_idx) & (signal.index < off_idx)
        result[dev.name][on_period & ~signal.isna()] = dev.p_typical_w

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
