"""
approach_hsmm_survey — Factorial Hidden Semi-Markov Model with survey priors.

Key difference from FHMM (approach_fhmm_survey):
  FHMM assumes state duration ~ Geometric (memoryless).
  HSMM uses an explicit duration distribution: each state has a P(d | state)
  that the Viterbi algorithm enforces natively — no post-processing needed.

Per-device inference:
  - Coordinate-ascent like fhmm_survey: iterate over devices, subtract others,
    run single-device HSMM Viterbi on residual, repeat until convergence.
  - 2 states per device: OFF (state 0) and ON (state 1).

Survey data used:
  - dur_min_min  → hard lower bound on ON duration (d ≥ dur_min)
  - dur_typical_min → mode of ON duration distribution
  - duration_minutes_max → hard upper bound on ON duration (d ≤ dur_max)
  - frequency_per_week → mean OFF duration (time between activations)
  - start_window_start/end_min → log-bias added to ON emissions inside window
  - active_months → log-bias for seasonal devices

No external libraries beyond numpy/scipy (scipy only for special functions).

Installation:
    pip install scipy  # only if not already present
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.baseline_load import estimate_always_on_baseline, split_baseline_by_device

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Fraction of p_typical used as sigma for ON emission: N(p_typical, sigma_on²)
_SIGMA_ON_FACTOR = 0.35

# Fixed sigma for OFF emission: N(0, sigma_off²)
_SIGMA_OFF_W = 80.0

# Log-bias added to ON emission inside / outside the time window
_LOG_BIAS_IN_WINDOW: float = math.log(1.5)   # ×1.5 probability boost
_LOG_BIAS_OUT_WINDOW: float = math.log(0.15)  # ×0.15 penalty

# Log-bias for in-season / out-of-season
_LOG_BIAS_IN_SEASON: float = math.log(1.3)
_LOG_BIAS_OUT_SEASON: float = math.log(0.1)

# Granularity for internal DP (minutes per slot)
_DEFAULT_GRANULARITY_MIN = 15

# Maximum OFF-state duration in slots (caps DP complexity)
_MAX_OFF_SLOTS_FACTOR = 8  # × mean-OFF-duration, at least 2 full days


# ---------------------------------------------------------------------------
# HSMM Viterbi — single 2-state device
# ---------------------------------------------------------------------------


def _build_on_duration_logprob(
    dur_min: int, dur_typ: int, dur_max: int
) -> np.ndarray:
    """Truncated discrete Gaussian log-probability for ON duration.

    Returns array of shape (dur_max - dur_min + 1,) indexed by (d - dur_min).
    Distribution peaks at dur_typ and falls off symmetrically, truncated to
    [dur_min, dur_max].
    """
    d_range = np.arange(dur_min, dur_max + 1, dtype=float)
    sigma_d = max(1.0, (dur_max - dur_min) / 4.0)
    log_p = -0.5 * ((d_range - dur_typ) / sigma_d) ** 2
    # Normalize (log-sum-exp trick)
    log_p -= float(np.logaddexp.reduce(log_p))
    return log_p


def _hsmm_viterbi(
    obs: np.ndarray,
    valid: np.ndarray,
    p_typical: float,
    sigma_on: float,
    sigma_off: float,
    dur_min_on: int,
    dur_max_on: int,
    log_p_on_dur: np.ndarray,
    dur_max_off: int,
    log_geo_p: float,
    log_geo_1mp: float,
    log_bias: np.ndarray,
) -> np.ndarray:
    """HSMM Viterbi decoder for a 2-state (OFF=0, ON=1) device.

    States alternate: every ON segment is preceded and followed by an OFF
    segment.  Duration of ON is distributed according to log_p_on_dur (lookup
    table indexed by d - dur_min_on).  Duration of OFF follows a geometric
    distribution with log-parameter log_geo_p (= log(p)) and log_geo_1mp
    (= log(1-p)), capped at dur_max_off.

    Args:
        obs:           Residual signal, shape (T,), NaN replaced with 0.
        valid:         Boolean mask, shape (T,), True where obs is valid.
        p_typical:     Expected ON-state power (W).
        sigma_on:      Std-dev of ON-state Gaussian emission (W).
        sigma_off:     Std-dev of OFF-state Gaussian emission (W, centred at 0).
        dur_min_on:    Minimum ON duration in slots.
        dur_max_on:    Maximum ON duration in slots.
        log_p_on_dur:  Precomputed log-prob of ON duration, shape (dur_max_on-dur_min_on+1,).
        dur_max_off:   Maximum OFF duration considered in DP (for efficiency).
        log_geo_p:     log(p) where p = geometric prob of ending OFF segment per slot.
        log_geo_1mp:   log(1-p).
        log_bias:      Per-timestep log-factor added to ON emission, shape (T,).

    Returns:
        Binary array of shape (T,) — 1 where device is estimated ON.
    """
    T = len(obs)
    NEG_INF = -1e30

    # Per-timestep log-emissions
    log_e_on = (
        -0.5 * ((obs - p_typical) / sigma_on) ** 2
        - math.log(sigma_on)
        + log_bias
    )
    log_e_off = -0.5 * (obs / sigma_off) ** 2 - math.log(sigma_off)

    # Neutral emission at NaN positions
    log_e_on[~valid] = 0.0
    log_e_off[~valid] = 0.0

    # Prefix sums for O(1) segment log-emission lookup
    # cumsum[t] - cumsum[t-d] = sum of log_e[t-d .. t-1] (d samples ending at t)
    cs_on = np.empty(T + 1)
    cs_off = np.empty(T + 1)
    cs_on[0] = cs_off[0] = 0.0
    cs_on[1:] = np.cumsum(log_e_on)
    cs_off[1:] = np.cumsum(log_e_off)

    # DP tables: best[t, s] = best log-prob of covering obs[0..t-1] with
    # the last segment in state s.  back[t, s] = start of that last segment.
    best = np.full((T + 1, 2), NEG_INF)
    back = np.full((T + 1, 2), -1, dtype=np.int32)

    # At t=0: empty sequence, uniform prior
    best[0, 0] = 0.0
    best[0, 1] = 0.0

    for t in range(1, T + 1):
        # --- Try ON segment of duration d ---
        d_lo = dur_min_on
        d_hi = min(dur_max_on, t)
        for d in range(d_lo, d_hi + 1):
            seg_start = t - d
            emit = cs_on[t] - cs_on[seg_start]
            dur_score = log_p_on_dur[d - dur_min_on]
            prev = best[seg_start, 0] if seg_start > 0 else 0.0  # must follow OFF
            score = prev + emit + dur_score
            if score > best[t, 1]:
                best[t, 1] = score
                back[t, 1] = seg_start

        # --- Try OFF segment of duration d ---
        d_hi = min(dur_max_off, t)
        for d in range(1, d_hi + 1):
            seg_start = t - d
            emit = cs_off[t] - cs_off[seg_start]
            # Geometric log-prob: (d-1)*log(1-p) + log(p)
            dur_score = (d - 1) * log_geo_1mp + log_geo_p
            prev = best[seg_start, 1] if seg_start > 0 else 0.0  # must follow ON
            score = prev + emit + dur_score
            if score > best[t, 0]:
                best[t, 0] = score
                back[t, 0] = seg_start

    # Traceback
    state_seq = np.zeros(T, dtype=np.int8)
    t = T
    s = int(np.argmax(best[T]))
    while t > 0:
        prev_t = back[t, s]
        if prev_t < 0:
            break
        state_seq[prev_t:t] = s
        s = 1 - s
        t = prev_t

    return state_seq.astype(np.float64)


# ---------------------------------------------------------------------------
# Survey helpers
# ---------------------------------------------------------------------------


def _build_log_bias(
    index: pd.DatetimeIndex,
    dev: DeviceProfile,
) -> np.ndarray:
    """Per-slot log-bias for ON emission derived from survey (window + season)."""
    T = len(index)
    bias = np.zeros(T, dtype=float)

    # Time-of-day window
    if dev.start_window_start_min is not None and dev.start_window_end_min is not None:
        mod = index.hour.to_numpy(int) * 60 + index.minute.to_numpy(int)
        s, e = dev.start_window_start_min, dev.start_window_end_min
        if e <= s:
            in_win = (mod >= s) | (mod < e)
        else:
            in_win = (mod >= s) & (mod < e)
        bias[in_win] += _LOG_BIAS_IN_WINDOW
        bias[~in_win] += _LOG_BIAS_OUT_WINDOW

    # Active months (season)
    if dev.active_months:
        month = index.month.to_numpy(int)
        in_season = np.isin(month, np.array(dev.active_months, dtype=int))
        bias[in_season] += _LOG_BIAS_IN_SEASON
        bias[~in_season] += _LOG_BIAS_OUT_SEASON

    return bias


def _device_hsmm_params(
    dev: DeviceProfile,
    granularity_min: int,
    observed_slots: int,
) -> dict:
    """Compute HSMM parameters from a DeviceProfile."""
    sigma_on = max(10.0, _SIGMA_ON_FACTOR * dev.p_typical_w)
    sigma_off = _SIGMA_OFF_W

    # ON duration bounds in slots
    dur_min = max(1, math.ceil(dev.dur_min_min / granularity_min))
    dur_typ = max(dur_min, round(dev.dur_typical_min / granularity_min))
    if dev.duration_minutes_max is not None:
        dur_max = max(dur_typ, math.ceil(dev.duration_minutes_max / granularity_min))
    else:
        dur_max = max(dur_typ, dur_typ * 3)  # fallback: 3× typical
    dur_max = min(dur_max, observed_slots)

    log_p_on_dur = _build_on_duration_logprob(dur_min, dur_typ, dur_max)

    # Geometric parameter for OFF state
    # Mean OFF duration = (slots per day) / (activations per day)
    slots_per_day = int(round(24 * 60 / granularity_min))
    freq_per_day = max(dev.frequency_per_week, 0.5) / 7.0
    mean_off_slots = max(1, int(slots_per_day / freq_per_day - dur_typ))
    geo_p = 1.0 / max(mean_off_slots, 1)
    geo_p = min(geo_p, 0.99)

    dur_max_off = min(observed_slots, max(int(mean_off_slots * _MAX_OFF_SLOTS_FACTOR), 2 * slots_per_day))

    return {
        "sigma_on": sigma_on,
        "sigma_off": sigma_off,
        "dur_min_on": dur_min,
        "dur_max_on": dur_max,
        "log_p_on_dur": log_p_on_dur,
        "dur_max_off": dur_max_off,
        "log_geo_p": math.log(geo_p),
        "log_geo_1mp": math.log(max(1.0 - geo_p, 1e-9)),
    }


# ---------------------------------------------------------------------------
# Power assignment (same strategy as fhmm_survey)
# ---------------------------------------------------------------------------


def _assign_power(
    states: np.ndarray,       # (T, n_dev) binary
    p_typicals: np.ndarray,   # (n_dev,)
    p_maxes: np.ndarray,      # (n_dev,)
    residual: np.ndarray,     # (T,) non-negative residual
    bursty_mask: np.ndarray,  # (n_dev,) True if bursty device
) -> np.ndarray:
    """Assign variable power to each device from the residual pool."""
    T, n_dev = states.shape
    assigned = np.zeros((T, n_dev), dtype=float)
    remaining = residual.copy()

    sort_order = np.argsort(p_typicals)[::-1]  # highest power first

    for i in sort_order:
        on_mask = states[:, i] == 1.0
        if not np.any(on_mask):
            continue
        if bursty_mask[i]:
            assigned[on_mask, i] = p_typicals[i]
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - p_typicals[i])
        else:
            p_assigned = np.clip(remaining, 0.0, p_maxes[i])
            assigned[on_mask, i] = p_assigned[on_mask]
            remaining[on_mask] = np.maximum(0.0, remaining[on_mask] - p_assigned[on_mask])

    return assigned


_BURSTY_DUR_THRESHOLD_MIN = 5.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    max_iter: int = 20,
    tol: float = 1e-3,
    baseline_mode: str = "duty_avg",
    baseline_method: str = "cyclic_median",
    granularity_min: int = _DEFAULT_GRANULARITY_MIN,
) -> dict[str, pd.Series]:
    """Factorial HSMM disaggregation with survey-derived duration priors.

    For each event device the algorithm runs HSMM Viterbi on the residual
    signal (after subtracting the contributions of all other devices) and
    updates the state estimate.  This coordinate-ascent loop repeats until
    convergence or max_iter.  The ON-state duration distribution is derived
    from the survey fields dur_min_min, dur_typical_min, duration_minutes_max.
    The time-of-day window and active months act as a log-bias on the ON
    emission — a principled soft constraint inside the Viterbi inference.

    No external solver needed — pure numpy/scipy.

    Args:
        signal: 1-minute aggregate power series with DatetimeIndex.
        devices: DeviceProfile list (should come from get_device_profiles_v2).
        max_iter: Maximum coordinate-ascent iterations.
        tol: Convergence threshold: fraction of timesteps that must be unchanged.
        baseline_mode: 'peak' or 'duty_avg' for always-on baseline subtraction.
        baseline_method: Method for baseline estimation (see baseline_load).
        granularity_min: Resolution for HSMM inference (default 15 min).
            Lower values are more precise but slower.

    Returns:
        dict mapping device name → estimated power series at 1-min resolution.
    """
    nan_mask = signal.isna().values

    result: dict[str, pd.Series] = {}
    for dev in devices:
        s = pd.Series(0.0, index=signal.index)
        s[nan_mask] = np.nan
        result[dev.name] = s

    present_always_on = [d for d in devices if d.always_on and d.prior_weight >= 1.0]
    present_events = [d for d in devices if not d.always_on and d.prior_weight >= 1.0]

    # ---- Always-on: estimate baseline and subtract --------------------------
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

    # ---- Resample to DP granularity -----------------------------------------
    rule = f"{granularity_min}min"
    sig_coarse = signal.resample(rule).mean()
    base_coarse = baseline_total.reindex(sig_coarse.index, method="nearest").fillna(0.0)

    valid_coarse = sig_coarse.notna().values
    sig_vals = sig_coarse.fillna(0.0).values.astype(float)
    residual_ca = np.clip(sig_vals - base_coarse.values, 0.0, None)

    T = len(sig_coarse)
    n_dev = len(present_events)

    # ---- Precompute HSMM parameters and log-bias per device -----------------
    hsmm_params: list[dict] = []
    log_biases: list[np.ndarray] = []
    for dev in present_events:
        hsmm_params.append(_device_hsmm_params(dev, granularity_min, T))
        log_biases.append(_build_log_bias(sig_coarse.index, dev))

    p_typicals = np.array([d.p_typical_w for d in present_events])
    p_maxes = np.array([d.p_max_w for d in present_events])
    bursty_mask = np.array([d.dur_typical_min <= _BURSTY_DUR_THRESHOLD_MIN for d in present_events])

    # ---- Coordinate-ascent --------------------------------------------------
    states = np.zeros((T, n_dev), dtype=np.float64)
    tol_count = tol * T

    for _iter in range(max_iter):
        total_changes = 0

        for i, dev in enumerate(present_events):
            # Residual = signal - baseline - other event devices
            others = states @ p_typicals - states[:, i] * p_typicals[i]
            dev_residual = np.clip(residual_ca - others, 0.0, None)

            params = hsmm_params[i]
            new_state = _hsmm_viterbi(
                obs=dev_residual,
                valid=valid_coarse,
                p_typical=dev.p_typical_w,
                sigma_on=params["sigma_on"],
                sigma_off=params["sigma_off"],
                dur_min_on=params["dur_min_on"],
                dur_max_on=params["dur_max_on"],
                log_p_on_dur=params["log_p_on_dur"],
                dur_max_off=params["dur_max_off"],
                log_geo_p=params["log_geo_p"],
                log_geo_1mp=params["log_geo_1mp"],
                log_bias=log_biases[i],
            )

            changes = int(np.sum(new_state != states[:, i]))
            total_changes += changes
            states[:, i] = new_state

        if total_changes < tol_count:
            break

    # ---- Variable power assignment ------------------------------------------
    assigned = _assign_power(states, p_typicals, p_maxes, residual_ca, bursty_mask)

    # ---- Build output at coarse resolution, then upsample to 1-min ----------
    for i, dev in enumerate(present_events):
        series_coarse = pd.Series(assigned[:, i], index=sig_coarse.index)
        series_1min = series_coarse.reindex(signal.index, method="ffill").fillna(0.0)
        series_1min[nan_mask] = np.nan
        result[dev.name] = series_1min

    return result
