"""
US-004: Approccio B — HMM per singolo dispositivo (unsupervised).

Per-device 2-state (ON/OFF) Gaussian HMM fitted via Baum-Welch,
initialized from the knowledge base. Each device's activation sequence
is estimated independently on the residual signal.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from scripts.nilm.devices import DeviceProfile


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
) -> dict[str, pd.Series]:
    """HMM-based NILM disaggregation (per-device, unsupervised).

    For each present device (prior_weight >= 1.0), fits a 2-state GaussianHMM
    on the residual signal (aggregate minus already-estimated devices), then
    predicts the state sequence and maps the high-power state to ON.

    Args:
        signal: 1-minute resampled w_medio series with DatetimeIndex.
        devices: List of DeviceProfile instances.

    Returns:
        dict mapping device name -> pd.Series with estimated power
        (0 or p_typical_w). NaN where signal is NaN.
    """
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[signal.isna()] = np.nan
        result[dev.name] = s

    # Residual signal updated after each device is estimated
    residual = signal.copy()

    present_devices = [d for d in devices if d.prior_weight >= 1.0]

    for dev in present_devices:
        # Drop NaN from residual for fitting
        valid_mask = residual.notna()
        obs = residual[valid_mask].values.reshape(-1, 1)

        # Skip if all NaN
        if len(obs) == 0:
            continue

        # Initialize GaussianHMM with knowledge-base means and variances
        p_off_mean = 5.0
        p_on_mean = float(dev.p_typical_w)
        var_off = 100.0
        var_on = float((dev.p_typical_w * 0.2) ** 2)

        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=50,
            random_state=42,
            init_params="st",   # init startprob & transmat, but set means/covars manually
            params="stmc",      # learn all params
        )

        # Set initial means and covariances
        model.means_ = np.array([[p_off_mean], [p_on_mean]])
        model.covars_ = np.array([[var_off], [var_on]])

        try:
            model.fit(obs)
            hidden_states = model.predict(obs)
        except Exception:
            # If fitting fails, leave device as zeros
            continue

        # Identify which state index corresponds to ON (higher mean)
        on_state = int(np.argmax(model.means_[:, 0]))

        # Build full-length state series (NaN positions stay NaN/zero)
        device_series = result[dev.name].copy()
        valid_indices = signal.index[valid_mask]
        on_mask_valid = hidden_states == on_state
        device_series.loc[valid_indices[on_mask_valid]] = dev.p_typical_w

        result[dev.name] = device_series

        # Subtract this device's estimate from the residual
        residual = residual - device_series.fillna(0)

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
