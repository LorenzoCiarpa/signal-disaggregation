"""
highs_methods.py — L1-MILP NILM disaggregation via HiGHS (free, no license).

Drop-in replacement for gurobi_methods (constrained_v4/v6) and cvxpy_methods.
Uses HiGHS directly via `highspy` — no CVXPY overhead, no C-level malloc crash.

Problem class: Mixed-Integer Linear Programming (MILP)
  Objective: L1 reconstruction error (sum of absolute errors) + linear penalties
  Constraints: identical to Gurobi v6 — transitions, min/max ON, time windows

Why L1 instead of L2 (unlike Gurobi):
  HiGHS handles only LP/MILP, not MIQP.  L1 avoids the quadratic objective
  while yielding comparable NILM results (often more robust to spike noise).
  L2 would require either SCIP (which malloc-crashed via CVXPY) or explicit
  bilinear linearization (~22k aux vars for n=7, K=3, T=96).

Required installation (already present in this environment):
    pip install highspy

Usage:
    from scripts.nilm.highs_methods import constrained_highs
    result, info = constrained_highs(day_signal, devices, ...)
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

logger = logging.getLogger(__name__)

_INSTALL_MSG = "highspy is not installed. Run: pip install highspy"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_allowed_window_mask(
    index: pd.DatetimeIndex,
    allowed_windows: list[tuple[int, int]],
) -> np.ndarray:
    minute_of_day = index.hour.to_numpy(int) * 60 + index.minute.to_numpy(int)
    mask = np.zeros(len(index), dtype=bool)
    for start_min, end_min in allowed_windows:
        s, e = int(start_min), int(end_min)
        if e <= s:
            mask |= (minute_of_day >= s) | (minute_of_day < e)
        else:
            mask |= (minute_of_day >= s) & (minute_of_day < e)
    return mask


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


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def constrained_highs(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    activation_penalty: float = 0.0,
) -> tuple[dict[str, pd.Series], dict]:
    """L1-MILP NILM disaggregation via HiGHS.

    Replaces constrained_scip (v4) and constrained_scip_activation (v6) with
    a formulation that avoids CVXPY's memory-intensive symbolic expansion.

    Objective: minimize Σ_t |Σ_i p_i x_i(t) - y(t)|
               + Σ_i time_window_penalty * p_i * (out-of-window activations)
               + Σ_i activation_penalty * p_i * (ON transitions)

    All constraints (transitions, min/max ON) are identical to Gurobi v4/v6.

    Args:
        signal: Aggregate power series for a single time window.
        devices: Event device profiles (present devices only).
        time_limit: HiGHS time limit in seconds.
        verbose: Show HiGHS console output.
        min_on_slots_by_device: Hard minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Hard maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Soft time-window penalty windows.
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot.
        activation_penalty: Penalty factor (× p_typical) per ON transition.
            Set to 0 (default) for v4 behaviour; > 0 for v6 behaviour.

    Returns:
        (disaggregation, info) same format as gurobi_methods functions.
    """
    if not _HAS_HIGHS:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    n = len(devices)
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or n == 0:
        return zero_output, _empty_info(n, T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_list = np.where(valid)[0]
    n_valid = len(valid_list)
    valid_pos = {int(t): k for k, t in enumerate(valid_list)}  # t → index in ep/en

    p = np.array([d.p_typical_w for d in devices], dtype=np.float64)

    # ------------------------------------------------------------------
    # Variable layout (flat index)
    #   x [i,t]  →  i*T + t                       binary   [0, n*T)
    #   u [i,t]  →  n*T + i*T + t                 binary   [n*T, 2*n*T)
    #   dw[i,t]  →  2*n*T + i*T + t               binary   [2*n*T, 3*n*T)
    #   ep[k]    →  3*n*T + k                      cont≥0   [3*n*T, 3*n*T+n_valid)
    #   en[k]    →  3*n*T + n_valid + k            cont≥0   [3*n*T+n_valid, end)
    # ------------------------------------------------------------------
    def X(i: int, t: int) -> int: return i * T + t
    def U(i: int, t: int) -> int: return n * T + i * T + t
    def DW(i: int, t: int) -> int: return 2 * n * T + i * T + t
    def EP(t: int) -> int: return 3 * n * T + valid_pos[t]
    def EN(t: int) -> int: return 3 * n * T + n_valid + valid_pos[t]

    n_bin = 3 * n * T
    n_vars = n_bin + 2 * n_valid

    # ------------------------------------------------------------------
    # Pre-compute out-of-window masks (per device, list of valid out-t)
    # ------------------------------------------------------------------
    out_masks: dict[int, list[int]] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = [t for t in valid_list.tolist() if not in_win[t]]
                if out_t:
                    out_masks[i] = out_t

    # ------------------------------------------------------------------
    # Build variable bounds and costs
    # ------------------------------------------------------------------
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.ones(n_vars, dtype=np.float64)
    ub[n_bin:] = _INF  # ep, en are continuous unbounded above

    cost = np.zeros(n_vars, dtype=np.float64)

    # L1 objective: ep[t] + en[t]
    for t in valid_list.tolist():
        cost[EP(t)] = 1.0
        cost[EN(t)] = 1.0

    # Time-window penalty (linear in x)
    for i, out_t in out_masks.items():
        scale = time_window_penalty * float(p[i])
        for t in out_t:
            cost[X(i, t)] += scale

    # Activation penalty (linear in u)
    if activation_penalty > 0.0:
        for i in range(n):
            scale = activation_penalty * float(p[i])
            for t in valid_list.tolist():
                cost[U(i, t)] += scale

    # ------------------------------------------------------------------
    # Initialise HiGHS model
    # ------------------------------------------------------------------
    h = hs.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("output_flag", verbose)

    # Add all variables at once
    h.addVars(n_vars, lb, ub)

    # Set costs
    h.changeColsCost(
        n_vars,
        np.arange(n_vars, dtype=np.int32),
        cost,
    )

    # Set integrality: first n_bin vars are binary (integer with bounds [0,1])
    bin_idx = np.arange(n_bin, dtype=np.int32)
    bin_int = np.full(n_bin, _INT.value, dtype=np.uint8)
    h.changeColsIntegrality(n_bin, bin_idx, bin_int)

    # Fix NaN timesteps: x[i,t] = u[i,t] = dw[i,t] = 0
    for t in range(T):
        if not valid[t]:
            for i in range(n):
                h.changeColBounds(X(i, t), 0.0, 0.0)
                h.changeColBounds(U(i, t), 0.0, 0.0)
                h.changeColBounds(DW(i, t), 0.0, 0.0)

    # dw[i,0] = 0 always (no previous state)
    for i in range(n):
        h.changeColBounds(DW(i, 0), 0.0, 0.0)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # 1. L1: Σ_i p_i x[i,t] - ep[t] + en[t] = y[t]
    for t in valid_list.tolist():
        idx = [X(i, t) for i in range(n)] + [EP(t), EN(t)]
        val = list(p) + [-1.0, 1.0]
        _row(h, y_safe[t], y_safe[t], idx, val)

    # 2. Init u: u[i,0] = x[i,0]
    for i in range(n):
        _row(h, 0.0, 0.0, [U(i, 0), X(i, 0)], [1.0, -1.0])

    # 3. Transition: x[i,t] - x[i,t-1] - u[i,t] + dw[i,t] = 0  (t ≥ 1)
    for i in range(n):
        for t in range(1, T):
            _row(h, 0.0, 0.0,
                 [X(i, t), X(i, t - 1), U(i, t), DW(i, t)],
                 [1.0, -1.0, -1.0, 1.0])

    # 4. Mutex: u[i,t] + dw[i,t] ≤ 1  (t ≥ 1)
    for i in range(n):
        for t in range(1, T):
            _row(h, -_INF, 1.0, [U(i, t), DW(i, t)], [1.0, 1.0])

    # 5. Min ON duration: Σ_{τ=t}^{t+a-1} x[i,τ] ≥ a * u[i,t]
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is None or int(a) < 2:
                continue
            a = int(a)
            for t in range(T - a + 1):
                x_idx = [X(i, tt) for tt in range(t, t + a)]
                _row(h, 0.0, _INF, x_idx + [U(i, t)], [1.0] * a + [-float(a)])

    # 6. Max consecutive ON: Σ_{τ=t}^{t+b} x[i,τ] ≤ b
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is None or int(b) >= T:
                continue
            b = int(b)
            for t in range(T - b):
                x_idx = [X(i, tt) for tt in range(t, t + b + 1)]
                _row(h, -_INF, float(b), x_idx, [1.0] * (b + 1))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    h.run()

    model_status = h.getModelStatus()
    prim_status = h.getInfoValue("primal_solution_status")[1]
    # 2 = Feasible in HiGHS primal_solution_status
    has_solution = prim_status == 2

    info = {
        "status": str(model_status).replace("HighsModelStatus.", ""),
        "has_solution": has_solution,
        "is_optimal": model_status == hs.HighsModelStatus.kOptimal,
        "objective_value": float(h.getInfoValue("objective_function_value")[1]) if has_solution else None,
        "runtime_sec": float(h.getInfoValue("run_time")[1]),
        "solver": "HiGHS-L1",
        "n_devices": n,
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("HiGHS: no feasible solution (status=%s)", model_status)
        return zero_output, info

    x_sol = np.array(h.getSolution().col_value, dtype=np.float64)

    disaggregation: dict[str, pd.Series] = {}
    for i, dev in enumerate(devices):
        x_bin = np.round(np.array([x_sol[X(i, t)] for t in range(T)])).clip(0.0, 1.0)
        disaggregation[dev.name] = pd.Series(
            x_bin * dev.p_typical_w, index=signal.index, dtype=float
        )
    return disaggregation, info


def constrained_highs_multistate(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 0.0,
) -> tuple[dict[str, pd.Series], dict]:
    """L1-MILP NILM with 3 power levels per device via HiGHS.

    Extends constrained_highs with multistate power levels.  Each device can
    operate at low / mid / high power (p_typical ± power_level_variation) instead
    of binary ON/OFF.  The ON/OFF aggregate indicator x[i,t] = Σ_k z[i,t,k] is
    still used for all transition and duration constraints, identical to Gurobi v5/v6.

    Objective: L1 (sum of absolute reconstruction errors) + linear penalties.
    Problem class: MILP (no quadratic terms, no malloc crash risk).

    Variable layout (flat index)
    ----------------------------
    z[i,t,k]  →  z_base + i*T*K + t*K + k    binary   [0, n*T*K)
    x[i,t]    →  x_base + i*T + t             binary   [n*T*K, n*T*K + n*T)
    u[i,t]    →  u_base + i*T + t             binary   [x_base+n*T, x_base+2*n*T)
    dw[i,t]   →  dw_base + i*T + t            binary   [u_base+n*T, u_base+2*n*T)
    ep[k]     →  ep_base + k                  cont≥0   (n_valid entries)
    en[k]     →  en_base + k                  cont≥0   (n_valid entries)

    where K = 3 (number of power levels).

    Args:
        signal: Aggregate power series for a single time window.
        devices: Event device profiles (present devices only).
        time_limit: HiGHS time limit in seconds.
        verbose: Show HiGHS console output.
        min_on_slots_by_device: Hard minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Hard maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Soft time-window penalty windows.
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot.
        power_level_variation: Fractional spread around p_typical (default 0.15 = ±15%).
        activation_penalty: Penalty factor (× p_typical) per ON transition.

    Returns:
        (disaggregation, info) — disaggregation maps device name → estimated power
        series with actual estimated wattage (not just p_typical).
    """
    if not _HAS_HIGHS:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    n = len(devices)
    K = 3
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or n == 0:
        return zero_output, _empty_info(n, T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_list = np.where(valid)[0]
    n_valid = len(valid_list)
    valid_pos = {int(t): k for k, t in enumerate(valid_list)}

    # Power levels per device: [low, mid, high]
    p_levels = np.array(
        [
            [
                d.p_typical_w * (1.0 - power_level_variation),
                d.p_typical_w,
                d.p_typical_w * (1.0 + power_level_variation),
            ]
            for d in devices
        ],
        dtype=np.float64,
    )  # shape (n, K)

    # ------------------------------------------------------------------
    # Variable index helpers
    # ------------------------------------------------------------------
    z_base = 0
    x_base = n * T * K
    u_base = x_base + n * T
    dw_base = u_base + n * T
    ep_base = dw_base + n * T
    en_base = ep_base + n_valid

    def Z(i: int, t: int, k: int) -> int: return z_base + i * T * K + t * K + k
    def X(i: int, t: int) -> int: return x_base + i * T + t
    def U(i: int, t: int) -> int: return u_base + i * T + t
    def DW(i: int, t: int) -> int: return dw_base + i * T + t
    def EP(t: int) -> int: return ep_base + valid_pos[t]
    def EN(t: int) -> int: return en_base + valid_pos[t]

    n_bin = ep_base  # all binary vars
    n_vars = en_base + n_valid

    # ------------------------------------------------------------------
    # Out-of-window masks
    # ------------------------------------------------------------------
    out_masks: dict[int, list[int]] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = [t for t in valid_list.tolist() if not in_win[t]]
                if out_t:
                    out_masks[i] = out_t

    # ------------------------------------------------------------------
    # Variable bounds and costs
    # ------------------------------------------------------------------
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.ones(n_vars, dtype=np.float64)
    ub[n_bin:] = _INF  # ep, en are unbounded above

    cost = np.zeros(n_vars, dtype=np.float64)

    # L1 objective: ep[t] + en[t]
    for t in valid_list.tolist():
        cost[EP(t)] = 1.0
        cost[EN(t)] = 1.0

    # Time-window penalty on x (out-of-window, scaled by p_typical)
    for i, out_t in out_masks.items():
        scale = time_window_penalty * float(devices[i].p_typical_w)
        for t in out_t:
            cost[X(i, t)] += scale

    # Activation penalty on u (scaled by p_typical)
    if activation_penalty > 0.0:
        for i, dev in enumerate(devices):
            scale = activation_penalty * float(dev.p_typical_w)
            for t in valid_list.tolist():
                cost[U(i, t)] += scale

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

    # Set integrality for all binary variables
    bin_idx = np.arange(n_bin, dtype=np.int32)
    h.changeColsIntegrality(n_bin, bin_idx, np.full(n_bin, _INT.value, dtype=np.uint8))

    # Fix NaN timesteps: z, x, u, dw = 0
    for t in range(T):
        if not valid[t]:
            for i in range(n):
                for k in range(K):
                    h.changeColBounds(Z(i, t, k), 0.0, 0.0)
                h.changeColBounds(X(i, t), 0.0, 0.0)
                h.changeColBounds(U(i, t), 0.0, 0.0)
                h.changeColBounds(DW(i, t), 0.0, 0.0)

    # dw[i,0] = 0 always
    for i in range(n):
        h.changeColBounds(DW(i, 0), 0.0, 0.0)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # 1. Link: x[i,t] = Σ_k z[i,t,k]   →   x[i,t] - Σ_k z[i,t,k] = 0
    for i in range(n):
        for t in range(T):
            idx = [X(i, t)] + [Z(i, t, k) for k in range(K)]
            val = [1.0] + [-1.0] * K
            _row(h, 0.0, 0.0, idx, val)

    # 2. L1: Σ_i Σ_k p_ik z[i,t,k] - ep[t] + en[t] = y[t]
    for t in valid_list.tolist():
        idx = [Z(i, t, k) for i in range(n) for k in range(K)] + [EP(t), EN(t)]
        val = [p_levels[i, k] for i in range(n) for k in range(K)] + [-1.0, 1.0]
        _row(h, y_safe[t], y_safe[t], idx, val)

    # 3. Init u: u[i,0] = x[i,0]
    for i in range(n):
        _row(h, 0.0, 0.0, [U(i, 0), X(i, 0)], [1.0, -1.0])

    # 4. Transition: x[i,t] - x[i,t-1] - u[i,t] + dw[i,t] = 0  (t ≥ 1)
    for i in range(n):
        for t in range(1, T):
            _row(h, 0.0, 0.0,
                 [X(i, t), X(i, t - 1), U(i, t), DW(i, t)],
                 [1.0, -1.0, -1.0, 1.0])

    # 5. Mutex: u[i,t] + dw[i,t] ≤ 1  (t ≥ 1)
    for i in range(n):
        for t in range(1, T):
            _row(h, -_INF, 1.0, [U(i, t), DW(i, t)], [1.0, 1.0])

    # 6. Min ON: Σ_{τ=t}^{t+a-1} x[i,τ] - a * u[i,t] ≥ 0
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is None or int(a) < 2:
                continue
            a = int(a)
            for t in range(T - a + 1):
                x_idx = [X(i, tt) for tt in range(t, t + a)]
                _row(h, 0.0, _INF, x_idx + [U(i, t)], [1.0] * a + [-float(a)])

    # 7. Max consecutive ON: Σ_{τ=t}^{t+b} x[i,τ] ≤ b
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is None or int(b) >= T:
                continue
            b = int(b)
            for t in range(T - b):
                x_idx = [X(i, tt) for tt in range(t, t + b + 1)]
                _row(h, -_INF, float(b), x_idx, [1.0] * (b + 1))

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
        "solver": "HiGHS-L1-multistate",
        "n_devices": n,
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("HiGHS multistate: no feasible solution (status=%s)", model_status)
        return zero_output, info

    x_sol = np.array(h.getSolution().col_value, dtype=np.float64)

    disaggregation: dict[str, pd.Series] = {}
    for i, dev in enumerate(devices):
        power_sol = np.array(
            [
                sum(p_levels[i, k] * round(x_sol[Z(i, t, k)]) for k in range(K))
                for t in range(T)
            ],
            dtype=float,
        )
        disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    return disaggregation, info


def constrained_highs_full(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 0.0,
) -> tuple[dict[str, pd.Series], dict]:
    """L1-MILP NILM with always-on devices modeled as multistate variables (mirror of constrained_v7).

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

    Everything else (link x=Σz, transitions, min/max ON, L1 reconstruction) is
    identical to constrained_highs_multistate.

    Args:
        signal: Aggregate power series for a single time window (raw, not residual).
        devices: ALL device profiles including always-on ones.
        time_limit: HiGHS time limit in seconds.
        verbose: Show HiGHS console output.
        min_on_slots_by_device: Hard minimum consecutive ON timesteps (event devices).
        max_consecutive_on_by_device: Hard maximum consecutive ON timesteps (event devices).
        allowed_on_windows_by_device: Soft time-window penalty windows (event devices).
        time_window_penalty: Penalty factor (× p_typical) per out-of-window slot.
        power_level_variation: Fractional spread around p_typical (default 0.15 = ±15%).
        activation_penalty: Penalty factor (× p_typical) per ON transition (event devices).

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
    n_vars = en_base + n_valid

    # ------------------------------------------------------------------
    # Out-of-window masks (event devices only)
    # ------------------------------------------------------------------
    out_masks: dict[int, list[int]] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(ev_devices):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = [t for t in valid_list.tolist() if not in_win[t]]
                if out_t:
                    out_masks[i] = out_t

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

    for i, out_t in out_masks.items():
        scale = time_window_penalty * float(ev_devices[i].p_typical_w)
        for t in out_t:
            cost[XEV(i, t)] += scale

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
                _row(h, -_INF, float(b), x_idx, [1.0] * (b + 1))

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
