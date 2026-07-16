"""
cvxpy_methods.py — Free-solver NILM via CVXPY + SCIP.

Provides three solvers mirroring gurobi_methods constrained_v4/v6/v7:

  constrained_scip            — binary ON/OFF, L2 error (≡ v4)
  constrained_scip_activation — 3 power levels + activation penalty (≡ v6)
  constrained_scip_full       — v6 + always-on devices as variables (≡ v7)

Problem class: Mixed-Integer Quadratic Programming (MIQP)
  min  Σ_t (Σ_i p_i x_i(t) - y(t))²  +  penalties
  s.t. x, u, dw ∈ {0,1}
       transition / min ON / max ON constraints

Required installation (one-time):
    pip install cvxpy pyscipopt

SCIP 9.0+ is Apache-2.0 licensed and free for all uses.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import cvxpy as cp

    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

from scripts.nilm.devices import DeviceProfile

logger = logging.getLogger(__name__)

_INSTALL_MSG = (
    "cvxpy or pyscipopt is not installed. Run:\n"
    "    pip install cvxpy pyscipopt\n"
    "SCIP 9.0+ is Apache-2.0 and free for all uses."
)


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
        "status": None,
        "has_solution": False,
        "is_optimal": False,
        "objective_value": None,
        "runtime_sec": 0.0,
        "solver": "SCIP",
        "n_devices": n,
        "n_timesteps": T,
    }


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def constrained_scip(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict]:
    """MIQP NILM disaggregation via CVXPY + SCIP (free, no license required).

    Mirrors the behaviour of gurobi_methods.constrained_v4 exactly:
    - Squared reconstruction error objective
    - Hard min-ON and max-consecutive constraints
    - Soft time-window penalty (activation outside window ↔ penalty × p_typical²)

    Args:
        signal: Aggregate power series for a single time window (any freq).
        devices: Device profiles for present event devices.
        time_limit: Solver time limit in seconds (passed to SCIP).
        verbose: If True, show SCIP console output.
        min_on_slots_by_device: Device name → minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Device name → maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Device name → [(start_min, end_min)] windows.
            Activations outside these windows incur a soft penalty.
        time_window_penalty: Penalty factor (× p_typical²) per out-of-window slot.

    Returns:
        (disaggregation, info) where disaggregation maps device name → power series
        aligned to signal.index.
    """
    if not _HAS_CVXPY:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    n = len(devices)
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or n == 0:
        return zero_output, _empty_info(n, T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_t = np.where(valid)[0]
    nan_t = np.where(~valid)[0]

    p = np.array([d.p_typical_w for d in devices], dtype=float)  # (n,)

    # Pre-process constraint dicts keyed by device index
    min_on: dict[int, int] = {}
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on[i] = int(a)

    max_on: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is not None and int(b) >= 1:
                max_on[i] = int(b)

    # Out-of-window masks: True = slot is outside the allowed window
    out_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = np.where(~in_win & valid)[0]
                if len(out_t) > 0:
                    out_masks[i] = out_t

    # ---- CVXPY variables --------------------------------------------------
    x = cp.Variable((n, T), boolean=True)   # ON/OFF per device per timestep
    u = cp.Variable((n, T), boolean=True)   # rising edges
    dw = cp.Variable((n, T), boolean=True)  # falling edges

    constraints: list = []

    # Fix NaN timesteps to zero
    if len(nan_t) > 0:
        constraints += [
            x[:, nan_t] == 0,
            u[:, nan_t] == 0,
            dw[:, nan_t] == 0,
        ]

    # State transition: x[i,t] - x[i,t-1] = u[i,t] - dw[i,t]
    constraints += [
        u[:, 0] == x[:, 0],
        dw[:, 0] == 0,
        x[:, 1:] - x[:, :-1] == u[:, 1:] - dw[:, 1:],
        u[:, 1:] + dw[:, 1:] <= 1,
    ]

    # Min ON duration: if device turns on it must stay on for a_i slots
    for i, a_i in min_on.items():
        for t in range(T - a_i + 1):
            constraints.append(cp.sum(x[i, t : t + a_i]) >= a_i * u[i, t])

    # Max consecutive ON (rolling window)
    for i, b_i in max_on.items():
        if b_i >= T:
            continue
        for t in range(T - b_i):
            constraints.append(cp.sum(x[i, t : t + b_i + 1]) <= b_i)

    # ---- Objective --------------------------------------------------------
    # Reconstruction error: Σ_t (Σ_i p_i x_i(t) - y(t))²
    # p @ x has shape (T,) → slice to valid timesteps only
    power = p @ x  # cp expression, shape (T,)
    obj_expr = cp.sum_squares(power[valid_t] - y_safe[valid_t])

    # Soft time-window penalty
    for i, out_t in out_masks.items():
        p_sq = float(devices[i].p_typical_w) ** 2
        obj_expr = obj_expr + time_window_penalty * p_sq * cp.sum(x[i, out_t])

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)

    try:
        prob.solve(
            solver=cp.SCIP,
            verbose=verbose,
            scip_params={"limits/time": time_limit},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SCIP solve error: %s — returning zeros", exc)
        return zero_output, _empty_info(n, T)

    has_solution = prob.status not in (
        cp.INFEASIBLE,
        cp.INFEASIBLE_INACCURATE,
        None,
    ) and x.value is not None

    info = {
        "status": prob.status,
        "has_solution": has_solution,
        "is_optimal": prob.status == cp.OPTIMAL,
        "objective_value": float(prob.value) if has_solution else None,
        "runtime_sec": prob.solver_stats.solve_time if prob.solver_stats else None,
        "solver": "SCIP",
        "n_devices": n,
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("SCIP: no feasible solution (status=%s)", prob.status)
        return zero_output, info

    disaggregation: dict[str, pd.Series] = {}
    x_val = x.value  # (n, T) float array
    for i, dev in enumerate(devices):
        x_bin = np.round(x_val[i]).clip(0, 1)
        disaggregation[dev.name] = pd.Series(
            x_bin * dev.p_typical_w, index=signal.index, dtype=float
        )
    return disaggregation, info


# ---------------------------------------------------------------------------
# constrained_scip_activation  (≡ Gurobi constrained_v6)
# ---------------------------------------------------------------------------


def constrained_scip_activation(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict]:
    """MIQP NILM with 3 power levels and activation penalty via CVXPY + SCIP.

    Extends constrained_scip with:
    - 3 power levels per device: p*(1-var), p, p*(1+var)  (≡ Gurobi v6)
    - Activation penalty: discourages rapid ON/OFF cycling

    Args:
        signal: Aggregate power series for a single time window.
        devices: Event device profiles.
        time_limit: SCIP time limit in seconds.
        verbose: Show SCIP output.
        min_on_slots_by_device: Hard minimum consecutive ON timesteps per device.
        max_consecutive_on_by_device: Hard maximum consecutive ON timesteps per device.
        allowed_on_windows_by_device: Soft time-window penalty windows.
        time_window_penalty: Penalty factor (× p_typical²) per out-of-window slot.
        power_level_variation: Fractional spread of low/high levels (default ±15%).
        activation_penalty: Penalty factor (× p_typical²) per ON transition.
    """
    if not _HAS_CVXPY:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    n = len(devices)
    K = 3  # number of ON power levels
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or n == 0:
        return zero_output, _empty_info(n, T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_t = np.where(valid)[0]
    nan_t = np.where(~valid)[0]

    # Power levels per device: [low, mid, high]
    power_levels = np.array([
        [d.p_typical_w * (1.0 - power_level_variation),
         d.p_typical_w,
         d.p_typical_w * (1.0 + power_level_variation)]
        for d in devices
    ])  # shape (n, K)

    # Preprocess constraint dicts
    min_on: dict[int, int] = {}
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on[i] = int(a)

    max_on: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is not None and int(b) >= 1:
                max_on[i] = int(b)

    out_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = np.where(~in_win & valid)[0]
                if len(out_t) > 0:
                    out_masks[i] = out_t

    # ---- CVXPY variables ---------------------------------------------------
    # z[i, t, k] = 1 if device i is at power level k at timestep t
    # x[i, t]    = sum_k z[i,t,k]  (aggregate ON indicator)
    z = cp.Variable((n, T, K), boolean=True)
    x = cp.Variable((n, T), boolean=True)
    u = cp.Variable((n, T), boolean=True)   # rising edges
    dw = cp.Variable((n, T), boolean=True)  # falling edges

    constraints: list = []

    # Link x to z
    for i in range(n):
        constraints.append(x[i, :] == cp.sum(z[i, :, :], axis=1))

    # Fix NaN timesteps
    if len(nan_t) > 0:
        constraints += [
            x[:, nan_t] == 0,
            u[:, nan_t] == 0,
            dw[:, nan_t] == 0,
            z[:, nan_t, :] == 0,
        ]

    # State transitions on aggregate x
    constraints += [
        u[:, 0] == x[:, 0],
        dw[:, 0] == 0,
        x[:, 1:] - x[:, :-1] == u[:, 1:] - dw[:, 1:],
        u[:, 1:] + dw[:, 1:] <= 1,
    ]

    for i, a_i in min_on.items():
        for t in range(T - a_i + 1):
            constraints.append(cp.sum(x[i, t: t + a_i]) >= a_i * u[i, t])

    for i, b_i in max_on.items():
        if b_i >= T:
            continue
        for t in range(T - b_i):
            constraints.append(cp.sum(x[i, t: t + b_i + 1]) <= b_i)

    # ---- Objective ---------------------------------------------------------
    # Reconstruction: Σ_t (Σ_i Σ_k p[i,k]*z[i,t,k] - y[t])²
    # power_at_t[t] = sum over i,k of power_levels[i,k] * z[i,t,k]
    # Use einsum-style: power = sum over k of (power_levels[:, k] @ z[:, :, k])
    power_sum = sum(
        power_levels[:, k] @ z[:, :, k] for k in range(K)
    )  # shape (T,)
    obj_expr = cp.sum_squares(power_sum[valid_t] - y_safe[valid_t])

    # Soft time-window penalty
    for i, out_t in out_masks.items():
        p_sq = float(devices[i].p_typical_w) ** 2
        obj_expr = obj_expr + time_window_penalty * p_sq * cp.sum(x[i, out_t])

    # Activation penalty: discourage rapid cycling
    for i, dev in enumerate(devices):
        p_sq = float(dev.p_typical_w) ** 2
        obj_expr = obj_expr + activation_penalty * p_sq * cp.sum(u[i, valid_t])

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)

    try:
        prob.solve(
            solver=cp.SCIP,
            verbose=verbose,
            scip_params={"limits/time": time_limit},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SCIP activation solve error: %s — returning zeros", exc)
        return zero_output, _empty_info(n, T)

    has_solution = prob.status not in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE, None) and z.value is not None

    info = {
        "status": prob.status,
        "has_solution": has_solution,
        "is_optimal": prob.status == cp.OPTIMAL,
        "objective_value": float(prob.value) if has_solution else None,
        "runtime_sec": prob.solver_stats.solve_time if prob.solver_stats else None,
        "solver": "SCIP",
        "n_devices": n,
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("SCIP activation: no feasible solution (status=%s)", prob.status)
        return zero_output, info

    z_val = z.value  # (n, T, K)
    disaggregation: dict[str, pd.Series] = {}
    for i, dev in enumerate(devices):
        power_sol = np.array([
            sum(power_levels[i, k] * round(z_val[i, t, k]) for k in range(K))
            for t in range(T)
        ])
        disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)
    return disaggregation, info


# ---------------------------------------------------------------------------
# constrained_scip_full  (≡ Gurobi constrained_v7)
# ---------------------------------------------------------------------------


def constrained_scip_full(
    signal: pd.Series,
    devices: list[DeviceProfile],
    time_limit: float = 60.0,
    verbose: bool = False,
    min_on_slots_by_device: dict[str, int] | None = None,
    max_consecutive_on_by_device: dict[str, int] | None = None,
    allowed_on_windows_by_device: dict[str, list[tuple[int, int]]] | None = None,
    time_window_penalty: float = 1.0,
    power_level_variation: float = 0.15,
    activation_penalty: float = 1.0,
) -> tuple[dict[str, pd.Series], dict]:
    """MIQP NILM with always-on devices as multistate variables via CVXPY + SCIP.

    Extends constrained_scip_activation by modeling always-on devices (fridges,
    freezers) directly as variables — Σ_k z[i,t,k] = 1 at every valid timestep —
    instead of subtracting a pre-estimated baseline.  The solver jointly optimises
    always-on level selection and event-device activation (≡ Gurobi v7).

    Args:
        signal: Raw aggregate power series (NOT residual — no baseline subtraction).
        devices: ALL device profiles, including always-on ones.
        time_limit, verbose, min_on_slots_by_device, max_consecutive_on_by_device,
        allowed_on_windows_by_device, time_window_penalty, power_level_variation,
        activation_penalty: same as constrained_scip_activation.
    """
    if not _HAS_CVXPY:
        raise ImportError(_INSTALL_MSG)

    T = len(signal)
    K = 3
    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}

    if T == 0 or not devices:
        return zero_output, _empty_info(len(devices), T)

    valid = signal.notna().to_numpy()
    y_safe = np.where(valid, signal.to_numpy(dtype=float), 0.0)
    valid_t = np.where(valid)[0]
    nan_t = np.where(~valid)[0]

    ao_devs = [(i, d) for i, d in enumerate(devices) if d.always_on]
    ev_devs = [(i, d) for i, d in enumerate(devices) if not d.always_on]
    n_ao, n_ev = len(ao_devs), len(ev_devs)

    def _levels(dev: DeviceProfile) -> list[float]:
        return [
            dev.p_typical_w * (1.0 - power_level_variation),
            dev.p_typical_w,
            dev.p_typical_w * (1.0 + power_level_variation),
        ]

    ao_levels = np.array([_levels(d) for _, d in ao_devs])  # (n_ao, K)
    ev_levels = np.array([_levels(d) for _, d in ev_devs])  # (n_ev, K)

    # Constraint preprocessing (event devices only)
    min_on: dict[int, int] = {}
    if min_on_slots_by_device:
        for j, (_, dev) in enumerate(ev_devs):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on[j] = int(a)

    max_on: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for j, (_, dev) in enumerate(ev_devs):
            b = max_consecutive_on_by_device.get(dev.name)
            if b is not None and int(b) >= 1:
                max_on[j] = int(b)

    out_masks_ev: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for j, (_, dev) in enumerate(ev_devs):
            wins = allowed_on_windows_by_device.get(dev.name)
            if wins:
                in_win = _build_allowed_window_mask(signal.index, wins)
                out_t = np.where(~in_win & valid)[0]
                if len(out_t) > 0:
                    out_masks_ev[j] = out_t

    # ---- CVXPY variables ---------------------------------------------------
    constraints: list = []

    # Always-on: exactly one level active at every valid timestep
    z_ao = cp.Variable((n_ao, T, K), boolean=True) if n_ao else None
    if z_ao is not None:
        for j in range(n_ao):
            if len(nan_t) > 0:
                constraints.append(z_ao[j, nan_t, :] == 0)
            constraints.append(cp.sum(z_ao[j, valid_t, :], axis=1) == 1)

    # Event devices: 3 levels + transitions
    z_ev = cp.Variable((n_ev, T, K), boolean=True) if n_ev else None
    x_ev = cp.Variable((n_ev, T), boolean=True) if n_ev else None
    u_ev = cp.Variable((n_ev, T), boolean=True) if n_ev else None
    dw_ev = cp.Variable((n_ev, T), boolean=True) if n_ev else None

    if n_ev:
        for j in range(n_ev):
            constraints.append(x_ev[j, :] == cp.sum(z_ev[j, :, :], axis=1))

        if len(nan_t) > 0:
            constraints += [
                x_ev[:, nan_t] == 0,
                u_ev[:, nan_t] == 0,
                dw_ev[:, nan_t] == 0,
                z_ev[:, nan_t, :] == 0,
            ]

        constraints += [
            u_ev[:, 0] == x_ev[:, 0],
            dw_ev[:, 0] == 0,
            x_ev[:, 1:] - x_ev[:, :-1] == u_ev[:, 1:] - dw_ev[:, 1:],
            u_ev[:, 1:] + dw_ev[:, 1:] <= 1,
        ]

        for j, a_j in min_on.items():
            for t in range(T - a_j + 1):
                constraints.append(cp.sum(x_ev[j, t: t + a_j]) >= a_j * u_ev[j, t])

        for j, b_j in max_on.items():
            if b_j >= T:
                continue
            for t in range(T - b_j):
                constraints.append(cp.sum(x_ev[j, t: t + b_j + 1]) <= b_j)

    # ---- Objective ---------------------------------------------------------
    power_ao = (
        sum(ao_levels[:, k] @ z_ao[:, :, k] for k in range(K))
        if n_ao and z_ao is not None
        else 0.0
    )
    power_ev = (
        sum(ev_levels[:, k] @ z_ev[:, :, k] for k in range(K))
        if n_ev and z_ev is not None
        else 0.0
    )
    total_power = power_ao + power_ev  # shape (T,)

    obj_expr = cp.sum_squares(total_power[valid_t] - y_safe[valid_t])

    if n_ev and z_ev is not None:
        for j, out_t in out_masks_ev.items():
            _, dev = ev_devs[j]
            p_sq = float(dev.p_typical_w) ** 2
            obj_expr = obj_expr + time_window_penalty * p_sq * cp.sum(x_ev[j, out_t])

        for j, (_, dev) in enumerate(ev_devs):
            p_sq = float(dev.p_typical_w) ** 2
            obj_expr = obj_expr + activation_penalty * p_sq * cp.sum(u_ev[j, valid_t])

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)

    try:
        prob.solve(
            solver=cp.SCIP,
            verbose=verbose,
            scip_params={"limits/time": time_limit},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SCIP full solve error: %s — returning zeros", exc)
        return zero_output, _empty_info(len(devices), T)

    has_ao = z_ao is not None and z_ao.value is not None
    has_ev = z_ev is not None and z_ev.value is not None
    has_solution = prob.status not in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE, None) and (has_ao or has_ev)

    info = {
        "status": prob.status,
        "has_solution": has_solution,
        "is_optimal": prob.status == cp.OPTIMAL,
        "objective_value": float(prob.value) if has_solution else None,
        "runtime_sec": prob.solver_stats.solve_time if prob.solver_stats else None,
        "solver": "SCIP",
        "n_devices": len(devices),
        "n_timesteps": T,
    }

    if not has_solution:
        logger.warning("SCIP full: no feasible solution (status=%s)", prob.status)
        return zero_output, info

    disaggregation: dict[str, pd.Series] = {}

    if has_ao:
        z_ao_val = z_ao.value
        for j, (_, dev) in enumerate(ao_devs):
            power_sol = np.array([
                sum(ao_levels[j, k] * round(z_ao_val[j, t, k]) for k in range(K))
                for t in range(T)
            ])
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    if has_ev:
        z_ev_val = z_ev.value
        for j, (_, dev) in enumerate(ev_devs):
            power_sol = np.array([
                sum(ev_levels[j, k] * round(z_ev_val[j, t, k]) for k in range(K))
                for t in range(T)
            ])
            disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    # Fill missing devices with zeros
    for dev in devices:
        if dev.name not in disaggregation:
            disaggregation[dev.name] = pd.Series(0.0, index=signal.index, dtype=float)

    return disaggregation, info
