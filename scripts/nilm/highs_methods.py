"""
highs_methods.py — L1-MILP NILM disaggregation via HiGHS (free, no license).

Builds and solves the day-wise NILM models directly via `highspy`.

Problem class: Mixed-Integer Linear Programming (MILP)
  Objective: L1 reconstruction error (sum of absolute errors) + linear penalties
  Constraints: transitions, min/max ON, time windows

Why L1:
  HiGHS handles only LP/MILP, not MIQP.  L1 avoids a quadratic objective while
  yielding comparable NILM results (often more robust to spike noise).

Required installation (already present in this environment):
    pip install highspy

Usage:
    from scripts.nilm.highs_methods import constrained_highs_full
    result, info = constrained_highs_full(day_signal, devices, ...)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import highspy as hs

    _HAS_HIGHS = True
    _INF = hs.kHighsInf
    _INT = hs.HighsVarType.kInteger
    _CONT = hs.HighsVarType.kContinuous
except ImportError:
    _HAS_HIGHS = False

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.time_windows import (
    WINDOW_PENALTY_MAX_FACTOR,
    WINDOW_PENALTY_RAMP_MIN,
    out_of_season_mask,
    window_penalty_factors,
)

logger = logging.getLogger(__name__)

_INSTALL_MSG = "highspy is not installed. Run: pip install highspy"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_info(n: int, T: int) -> dict:
    return {
        "status": "not_run",
        "has_solution": False,
        "is_optimal": False,
        "objective_value": None,
        "runtime_sec": 0.0,
        "solver": "HiGHS-L1",
        "n_devices": n,
        "n_timesteps": T,
    }


def _row(h, lb: float, ub: float, idx: list[int], val: list[float]) -> None:
    """Add a single row to the HiGHS model."""
    idx_a = np.array(idx, dtype=np.int32)
    val_a = np.array(val, dtype=np.float64)
    h.addRow(lb, ub, len(idx_a), idx_a, val_a)


def constrained_highs_full(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    window_penalty_ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    window_penalty_max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    power_level_variation: float = 0.15,
    activation_penalty: float = 0.0,
    duration_penalty_block: float = 0.0,
    duration_penalty_daily: float = 0.0,
    expected_daily_on_slots_by_device: dict[str, float] | None = None,
    over_activation_penalty_by_device: dict[str, float] | None = None,
    remaining_activations_by_device: dict[str, float] | None = None,
    window_vagueness_weight: float = 0.0,
    season_penalty: float = 0.0,
    active_months_by_device: dict[str, list[int]] | None = None,
) -> tuple[dict[str, pd.Series], dict]:
    """L1-MILP NILM with always-on devices modeled as multistate variables.

    Always-on devices (dev.always_on=True) are NOT estimated via baseline subtraction.
    Instead they enter the solver directly with exactly one power level active at every
    valid timestep.  Event devices get the full multistate treatment (3 levels,
    transitions, min/max ON, activation penalty, soft time-window penalty).

    Variable layout
    ---------------
    z_ao[j,t,k]  →  z_ao_base + j*T*K + t*K + k   binary  always-on devices
    z_ev[i,t,k]  →  z_ev_base + i*T*K + t*K + k   binary  event devices
    x_ev[i,t]    →  x_ev_base + i*T + t            binary  event ON indicator
    u_ev[i,t]    →  u_ev_base + i*T + t            binary  event rising edge
    dw_ev[i,t]   →  dw_ev_base + i*T + t           binary  event falling edge
    ep[k]        →  ep_base + k                    cont≥0  positive error (n_valid entries)
    en[k]        →  en_base + k                    cont≥0  negative error (n_valid entries)

    Constraints for always-on devices
    -----------------------------------
    Σ_k z_ao[j,t,k] = 1   ∀j, ∀t valid   (exactly one level always active)
    z_ao[j,t,k]     = 0   ∀j, ∀t invalid, ∀k

    The event devices get the standard treatment: link x=Σz, transitions,
    min/max ON, and L1 reconstruction.

    Args:
        signal: Aggregate power series for a single time window (raw, not residual).
        devices: ALL device profiles including always-on ones.
        time_limit: HiGHS time limit in seconds.
        verbose: Show HiGHS console output.
        min_on_slots_by_device: Hard minimum consecutive ON timesteps (event devices).
        max_consecutive_on_by_device: Hard maximum consecutive ON timesteps (event devices).
        allowed_on_windows_by_device: Soft time-window penalty windows (event devices).
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot,
            graduated by distance from the nearest allowed window.
        window_penalty_ramp_min: Minutes away from the window adding one unit of penalty.
        window_penalty_max_factor: Upper bound on the distance-graduated factor.
        power_level_variation: Fractional spread around p_typical (default 0.15 = ±15%).
        activation_penalty: Penalty factor (× p_typical) per ON transition (event devices).
        duration_penalty_block: If > 0, min-ON and max-consecutive limits become soft;
            a block pays this factor (× p_typical) per slot of deviation from them.
        duration_penalty_daily: If > 0, penalise the deviation of total daily ON time
            from expected_daily_on_slots_by_device (× p_typical per slot).
        expected_daily_on_slots_by_device: Expected ON slots per day, per device.
        over_activation_penalty_by_device: Extra cost (× p_typical) per activation past
            the remaining weekly quota.  Per device because the break-even scales with
            the device's typical ON length in slots.  The caller carries the counter
            across days, since each day is solved as its own model.
        remaining_activations_by_device: Activations still available this week, per device.
        window_vagueness_weight: Weight on the vagueness floor of the time-window
            penalty, so a device that named a narrow window beats one that named none.
        season_penalty: Cost (× p_typical) per ON slot in a month the device was not
            declared active.
        active_months_by_device: Months (1-12) each device was declared active in.

    Returns:
        (disaggregation, info) — disaggregation maps every device name (always-on and
        event) to an estimated power series with actual wattage.
    """
    if not _HAS_HIGHS:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    K = 3

    ao_devices = [d for d in devices if d.always_on]
    ev_devices = [d for d in devices if not d.always_on]
    n_ao = len(ao_devices)
    n_ev = len(ev_devices)

    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or (n_ao + n_ev) == 0:
        return zero_output, _empty_info(len(devices), T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_list = np.where(valid)[0]
    n_valid = len(valid_list)
    valid_pos = {int(t): k for k, t in enumerate(valid_list)}

    def _levels(dev: DeviceProfile) -> np.ndarray:
        return np.array([
            dev.p_typical_w * (1.0 - power_level_variation),
            dev.p_typical_w,
            dev.p_typical_w * (1.0 + power_level_variation),
        ], dtype=np.float64)

    ao_levels = [_levels(d) for d in ao_devices]
    ev_levels = [_levels(d) for d in ev_devices]

    # ------------------------------------------------------------------
    # Variable index helpers
    # ------------------------------------------------------------------
    z_ao_base = 0
    z_ev_base = n_ao * T * K
    x_ev_base = z_ev_base + n_ev * T * K
    u_ev_base = x_ev_base + n_ev * T
    dw_ev_base = u_ev_base + n_ev * T
    ep_base = dw_ev_base + n_ev * T
    en_base = ep_base + n_valid

    def ZAO(j: int, t: int, k: int) -> int: return z_ao_base + j * T * K + t * K + k
    def ZEV(i: int, t: int, k: int) -> int: return z_ev_base + i * T * K + t * K + k
    def XEV(i: int, t: int) -> int: return x_ev_base + i * T + t
    def UEV(i: int, t: int) -> int: return u_ev_base + i * T + t
    def DWEV(i: int, t: int) -> int: return dw_ev_base + i * T + t
    def EP(t: int) -> int: return ep_base + valid_pos[t]
    def EN(t: int) -> int: return en_base + valid_pos[t]

    n_bin = ep_base

    # Duration-slack columns (continuous, only allocated when the penalties are on,
    # so the default path builds exactly the same model as before).
    soft_duration = duration_penalty_block > 0.0
    use_daily = bool(duration_penalty_daily > 0.0 and expected_daily_on_slots_by_device)
    n_sdur = n_ev * T if soft_duration else 0
    n_daily = n_ev if use_daily else 0

    use_quota = bool(over_activation_penalty_by_device and remaining_activations_by_device)
    n_quota = n_ev if use_quota else 0

    sshort_base = en_base + n_valid
    sover_base = sshort_base + n_sdur
    dover_base = sover_base + n_sdur
    dunder_base = dover_base + n_daily
    eact_base = dunder_base + n_daily
    n_vars = eact_base + n_quota

    def SSHORT(i: int, t: int) -> int: return sshort_base + i * T + t
    def SOVER(i: int, t: int) -> int: return sover_base + i * T + t
    def DOVER(i: int) -> int: return dover_base + i
    def DUNDER(i: int) -> int: return dunder_base + i
    def EACT(i: int) -> int: return eact_base + i

    # ------------------------------------------------------------------
    # Out-of-window masks (event devices only)
    # ------------------------------------------------------------------
    out_factors: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device or window_vagueness_weight > 0.0:
        windows_by_device = allowed_on_windows_by_device or {}
        for i, dev in enumerate(ev_devices):
            wins = windows_by_device.get(dev.name)
            # With a vagueness floor every device is charged, including the ones that
            # declared nothing — that is precisely what the floor is for.
            if not wins and window_vagueness_weight <= 0.0:
                continue
            factors = window_penalty_factors(
                signal.index,
                wins,
                ramp_min=window_penalty_ramp_min,
                max_factor=window_penalty_max_factor,
                vagueness_weight=window_vagueness_weight,
            )
            if factors.any():
                out_factors[i] = factors

    season_masks: dict[int, np.ndarray] = {}
    if season_penalty > 0.0 and active_months_by_device:
        for i, dev in enumerate(ev_devices):
            mask = out_of_season_mask(signal.index, active_months_by_device.get(dev.name))
            if mask.any():
                season_masks[i] = mask

    # ------------------------------------------------------------------
    # Variable bounds and costs
    # ------------------------------------------------------------------
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.ones(n_vars, dtype=np.float64)
    ub[n_bin:] = _INF

    cost = np.zeros(n_vars, dtype=np.float64)
    for t in valid_list.tolist():
        cost[EP(t)] = 1.0
        cost[EN(t)] = 1.0

    for i, factors in out_factors.items():
        scale = time_window_penalty * float(ev_devices[i].p_typical_w)
        for t in valid_list.tolist():
            if factors[t] > 0.0:
                cost[XEV(i, t)] += scale * float(factors[t])

    if soft_duration:
        for i, dev in enumerate(ev_devices):
            scale = duration_penalty_block * float(dev.p_typical_w)
            for t in range(T):
                cost[SSHORT(i, t)] = scale
                cost[SOVER(i, t)] = scale

    if use_daily:
        for i, dev in enumerate(ev_devices):
            scale = duration_penalty_daily * float(dev.p_typical_w)
            cost[DOVER(i)] = scale
            cost[DUNDER(i)] = scale

    for i, mask in season_masks.items():
        scale = season_penalty * float(ev_devices[i].p_typical_w)
        for t in valid_list.tolist():
            if mask[t]:
                cost[XEV(i, t)] += scale

    if use_quota:
        for i, dev in enumerate(ev_devices):
            lam = over_activation_penalty_by_device.get(dev.name)
            if lam and lam > 0:
                cost[EACT(i)] = float(lam) * float(dev.p_typical_w)

    if activation_penalty > 0.0:
        for i, dev in enumerate(ev_devices):
            scale = activation_penalty * float(dev.p_typical_w)
            for t in valid_list.tolist():
                cost[UEV(i, t)] += scale

    # ------------------------------------------------------------------
    # HiGHS model
    # ------------------------------------------------------------------
    h = hs.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("output_flag", verbose)

    h.addVars(n_vars, lb, ub)
    h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int32), cost)
    h.changeColsIntegrality(n_bin, np.arange(n_bin, dtype=np.int32),
                            np.full(n_bin, _INT.value, dtype=np.uint8))

    # Fix NaN: always-on
    for t in range(T):
        if not valid[t]:
            for j in range(n_ao):
                for k in range(K):
                    h.changeColBounds(ZAO(j, t, k), 0.0, 0.0)

    # Fix NaN: event devices + dw_ev[i,0] = 0
    for t in range(T):
        if not valid[t]:
            for i in range(n_ev):
                for k in range(K):
                    h.changeColBounds(ZEV(i, t, k), 0.0, 0.0)
                h.changeColBounds(XEV(i, t), 0.0, 0.0)
                h.changeColBounds(UEV(i, t), 0.0, 0.0)
                h.changeColBounds(DWEV(i, t), 0.0, 0.0)
    for i in range(n_ev):
        h.changeColBounds(DWEV(i, 0), 0.0, 0.0)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # 1. Always-on equality: Σ_k z_ao[j,t,k] = 1  (valid t only)
    for j in range(n_ao):
        for t in valid_list.tolist():
            idx = [ZAO(j, t, k) for k in range(K)]
            _row(h, 1.0, 1.0, idx, [1.0] * K)

    # 2. Event link: x_ev[i,t] = Σ_k z_ev[i,t,k]
    for i in range(n_ev):
        for t in range(T):
            idx = [XEV(i, t)] + [ZEV(i, t, k) for k in range(K)]
            _row(h, 0.0, 0.0, idx, [1.0] + [-1.0] * K)

    # 3. L1 reconstruction: Σ_j Σ_k p_ao z_ao + Σ_i Σ_k p_ev z_ev - ep + en = y
    for t in valid_list.tolist():
        idx = (
            [ZAO(j, t, k) for j in range(n_ao) for k in range(K)]
            + [ZEV(i, t, k) for i in range(n_ev) for k in range(K)]
            + [EP(t), EN(t)]
        )
        val = (
            [float(ao_levels[j][k]) for j in range(n_ao) for k in range(K)]
            + [float(ev_levels[i][k]) for i in range(n_ev) for k in range(K)]
            + [-1.0, 1.0]
        )
        _row(h, y_safe[t], y_safe[t], idx, val)

    # 4. Init u_ev: u_ev[i,0] = x_ev[i,0]
    for i in range(n_ev):
        _row(h, 0.0, 0.0, [UEV(i, 0), XEV(i, 0)], [1.0, -1.0])

    # 5. Transition: x_ev[i,t] - x_ev[i,t-1] - u_ev[i,t] + dw_ev[i,t] = 0
    for i in range(n_ev):
        for t in range(1, T):
            _row(h, 0.0, 0.0,
                 [XEV(i, t), XEV(i, t - 1), UEV(i, t), DWEV(i, t)],
                 [1.0, -1.0, -1.0, 1.0])

    # 6. Mutex: u_ev[i,t] + dw_ev[i,t] ≤ 1
    for i in range(n_ev):
        for t in range(1, T):
            _row(h, -_INF, 1.0, [UEV(i, t), DWEV(i, t)], [1.0, 1.0])

    # 7. Min ON: Σ_{τ=t}^{t+a-1} x_ev[i,τ] - a * u_ev[i,t] ≥ 0
    if min_on_slots_by_device:
        for i, dev in enumerate(ev_devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is None or int(a) < 2:
                continue
            a = int(a)
            for t in range(T - a + 1):
                x_idx = [XEV(i, tt) for tt in range(t, t + a)]
                if soft_duration:
                    _row(h, 0.0, _INF,
                         x_idx + [SSHORT(i, t), UEV(i, t)],
                         [1.0] * a + [1.0, -float(a)])
                else:
                    _row(h, 0.0, _INF, x_idx + [UEV(i, t)], [1.0] * a + [-float(a)])

    # 8. Max consecutive ON: Σ_{τ=t}^{t+b} x_ev[i,τ] ≤ b
    if max_consecutive_on_by_device:
        for i, dev in enumerate(ev_devices):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is None or int(b) >= T:
                continue
            b = int(b)
            for t in range(T - b):
                x_idx = [XEV(i, tt) for tt in range(t, t + b + 1)]
                if soft_duration:
                    _row(h, -_INF, float(b),
                         x_idx + [SOVER(i, t)], [1.0] * (b + 1) + [-1.0])
                else:
                    _row(h, -_INF, float(b), x_idx, [1.0] * (b + 1))

    # 10. Weekly activation quota: Σ_t u_ev[i,t] - excess ≤ remaining.
    # The first `remaining` activations cost only activation_penalty; each one past
    # it also pays over_activation_penalty via the excess column.
    if use_quota:
        for i, dev in enumerate(ev_devices):
            remaining = remaining_activations_by_device.get(dev.name)
            lam = over_activation_penalty_by_device.get(dev.name)
            if remaining is None or not lam or lam <= 0:
                continue
            idx = [UEV(i, t) for t in range(T)] + [EACT(i)]
            _row(h, -_INF, max(0.0, float(remaining)), idx, [1.0] * T + [-1.0])

    # 9. Daily ON-time budget: Σ_t x_ev[i,t] - over + under = expected
    if use_daily:
        for i, dev in enumerate(ev_devices):
            expected = expected_daily_on_slots_by_device.get(dev.name)
            if expected is None or expected < 0:
                continue
            idx = [XEV(i, t) for t in range(T)] + [DOVER(i), DUNDER(i)]
            _row(h, float(expected), float(expected), idx, [1.0] * T + [-1.0, 1.0])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    h.run()

    model_status = h.getModelStatus()
    prim_status = h.getInfoValue("primal_solution_status")[1]
    has_solution = prim_status == 2

    info = {
        "status": str(model_status).replace("HighsModelStatus.", ""),
        "has_solution": has_solution,
        "is_optimal": model_status == hs.HighsModelStatus.kOptimal,
        "objective_value": float(h.getInfoValue("objective_function_value")[1]) if has_solution else None,
        "runtime_sec": float(h.getInfoValue("run_time")[1]),
        "solver": "HiGHS-L1-full",
        "n_devices": len(devices),
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("HiGHS full: no feasible solution (status=%s)", model_status)
        return zero_output, info

    x_sol = np.array(h.getSolution().col_value, dtype=np.float64)

    disaggregation: dict[str, pd.Series] = {}
    for j, dev in enumerate(ao_devices):
        power_sol = np.array(
            [sum(ao_levels[j][k] * round(x_sol[ZAO(j, t, k)]) for k in range(K)) for t in range(T)],
            dtype=float,
        )
        disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    for i, dev in enumerate(ev_devices):
        power_sol = np.array(
            [sum(ev_levels[i][k] * round(x_sol[ZEV(i, t, k)]) for k in range(K)) for t in range(T)],
            dtype=float,
        )
        disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    return disaggregation, info
