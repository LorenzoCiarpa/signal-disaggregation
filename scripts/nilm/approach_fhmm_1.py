"""
fhmm_1 — Variante FHMM con carico always-on e finestra di impegno.

Differenze rispetto al FHMM base:
1. I dispositivi always_on (frigorifero principale, frigorifero secondario,
    congelatore) vengono trattati come
   baseline costante sottratta dal segnale prima del coordinate-ascent.
   Due modalità:
     "peak"     → baseline = Σ p_typical_w  (es. 125W per frigo)
     "duty_avg" → baseline = Σ (p_typical_w × duty_cycle)  (es. 37.5W)
2. Il coordinate-ascent gira solo sui dispositivi "event" (non always_on).
3. Post-processing con finestra di impegno fisso (dur_typical_min) invece di
   rimuovere solo i blocchi brevi. Una volta rilevata l'accensione di un
   dispositivo, rimane acceso per almeno dur_typical_min minuti.
4. Potenza variabile dal residuo per i dispositivi discontinui (non bursty).
   I dispositivi bursty (dur_typical_min ≤ 5, es. microonde) mantengono
   la potenza tipica fissa.
"""

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile


# Threshold: dispositivi con uso breve e potenza stabile → output fisso
_BURSTY_DUR_THRESHOLD_MIN = 5.0


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    max_iter: int = 50,
    tol: float = 1e-3,
    baseline_mode: str = "peak",
) -> dict[str, pd.Series]:
    """FHMM-1 disaggregation con always-on baseline e commitment window.

    Args:
        signal: 1-minute resampled w_medio series with DatetimeIndex.
        devices: List of DeviceProfile instances.
        max_iter: Maximum coordinate-ascent iterations (default 50).
        tol: Convergence threshold as fraction of signal length (default 1e-3).
        baseline_mode:
            "peak"     — baseline = Σ p_typical_w per always-on devices presenti.
            "duty_avg" — baseline = Σ (p_typical_w × duty_cycle) per always-on.

    Returns:
        dict mapping device name -> pd.Series with estimated power.
        NaN where signal is NaN.
    """
    nan_mask = signal.isna().values

    # Build full output dict — zero series with NaN where signal is NaN
    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[nan_mask] = np.nan
        result[dev.name] = s

    # Split devices: always-on (present) vs event (present, not always-on)
    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

    # --- Always-on devices: output costante in tutto il segnale ---
    for dev in present_always_on:
        if baseline_mode == "duty_avg" and dev.duty_cycle is not None:
            power_val = dev.p_typical_w * dev.duty_cycle
        else:
            power_val = dev.p_typical_w
        s = pd.Series(power_val, index=signal.index)
        s[nan_mask] = np.nan
        result[dev.name] = s

    if not present_events:
        return result

    # --- Calcolo baseline totale da sottrarre ---
    baseline = 0.0
    for dev in present_always_on:
        if baseline_mode == "duty_avg" and dev.duty_cycle is not None:
            baseline += dev.p_typical_w * dev.duty_cycle
        else:
            baseline += dev.p_typical_w

    sig_vals = signal.fillna(0).values.astype(float)
    # Residuo per il coordinate-ascent: clip a 0 per evitare residui negativi
    residual_ca = np.clip(sig_vals - baseline, 0.0, None)

    T = len(signal)
    n_dev = len(present_events)
    p_typicals = np.array([d.p_typical_w for d in present_events], dtype=float)

    # --- Greedy coordinate-ascent sul residuo ---
    states = np.zeros((T, n_dev), dtype=np.float64)
    tol_count = tol * T

    for _iter in range(max_iter):
        total_changes = 0
        for i in range(n_dev):
            others = states @ p_typicals - states[:, i] * p_typicals[i]
            dev_residual = residual_ca - others
            new_state_i = (dev_residual > p_typicals[i] / 2.0).astype(np.float64)
            changes = int(np.sum(new_state_i != states[:, i]))
            total_changes += changes
            states[:, i] = new_state_i
        if total_changes < tol_count:
            break

    # --- Post-processing: commitment window ---
    for i, dev in enumerate(present_events):
        commitment_len = max(1, int(dev.dur_typical_min))
        states[:, i] = _apply_commitment_window(states[:, i], commitment_len)

    # --- Assegnazione potenza variabile dal residuo (dopo baseline) ---
    # Usiamo max(0, sig - baseline) come pool di potenza da distribuire
    remaining = np.maximum(0.0, sig_vals - baseline)

    # Ordine di assegnazione: potenza tipica decrescente (grandi consumi prima)
    sort_order = np.argsort(p_typicals)[::-1]

    assigned = np.zeros((T, n_dev), dtype=np.float64)

    for i in sort_order:
        dev = present_events[i]
        on_mask = states[:, i] == 1.0
        if not np.any(on_mask):
            continue

        bursty = dev.dur_typical_min <= _BURSTY_DUR_THRESHOLD_MIN
        if bursty:
            # Dispositivo bursty (es. microonde): potenza fissa p_typical
            assigned[on_mask, i] = dev.p_typical_w
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - dev.p_typical_w)
        else:
            # Dispositivo discontinuo: potenza variabile dal residuo rimasto,
            # clampata tra 0 e p_max. Il lower bound è 0 (non p_min) per
            # rispettare la conservazione dell'energia nelle fasi silenziose.
            p_assigned = np.clip(remaining, 0.0, dev.p_max_w)
            assigned[on_mask, i] = p_assigned[on_mask]
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - p_assigned[on_mask])

    # Build output series
    for i, dev in enumerate(present_events):
        series = pd.Series(assigned[:, i], index=signal.index)
        series[nan_mask] = np.nan
        result[dev.name] = series

    return result


def _apply_commitment_window(arr: np.ndarray, commitment_len: int) -> np.ndarray:
    """Tieni il dispositivo ON per commitment_len campioni dopo ogni rising edge.

    Ogni volta che arr[i]=1 (rilevamento originale), estende la finestra di
    impegno fino a min(n-1, i + commitment_len - 1). Tutti i campioni
    all'interno della finestra attiva vengono impostati a 1.

    Questo previene lo spegnimento prematuro durante le fasi silenziose di
    dispositivi discontinui (es. lavatrice tra fase riscaldamento e risciacquo).
    """
    if commitment_len <= 1:
        return arr.copy()

    out = np.zeros_like(arr)
    n = len(arr)
    commit_end = -1

    for i in range(n):
        if arr[i] == 1.0:
            # Ogni rilevamento estende la finestra di impegno
            new_end = min(n - 1, i + commitment_len - 1)
            if new_end > commit_end:
                commit_end = new_end
        if i <= commit_end:
            out[i] = 1.0

    return out


if __name__ == "__main__":
    from scripts.nilm.preprocessing import get_usable_imeis, load_imei
    from scripts.nilm.devices import get_device_profiles

    for mode in ("peak", "duty_avg"):
        print(f"\n=== baseline_mode={mode} ===")
        imei = get_usable_imeis()[0]
        signal = load_imei(imei)
        devices = get_device_profiles(imei)
        disagg = run(signal, devices, baseline_mode=mode)
        for name, series in disagg.items():
            on_minutes = int((series > 0).sum())
            if on_minutes > 0:
                avg_w = series[series > 0].mean()
                print(f"  {name}: {on_minutes} min ON, avg {avg_w:.0f}W")
