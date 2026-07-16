"""
NILM L2-MIQP solver using pyscipopt directly (no CVXPY).

Implements constrained_scip(), equivalent to gurobi_methods.constrained_v6:
  - L2 objective (sum of squared reconstruction errors)
  - 3 power levels per device (low / mid / high = p_typical ± variation)
  - Soft time-window penalty (× p_typical²)
  - Per-activation penalty (× p_typical²)
  - Hard min-ON and max-consecutive-ON constraints
  - Transition variables u (rising edge) and dw (falling edge)

The L2 objective is expressed via epigraph: min Σ_t q[t]  s.t. q[t] ≥ r[t]²
where r[t] is the residual at timestep t.  This is equivalent to minimising
the squared error but avoids setting a nonlinear objective (which pyscipopt
does not support), using a nonlinear constraint instead (which SCIP does).

Requires: pip install pyscipopt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.gurobi_methods import _build_allowed_window_mask


def constrained_scip(
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
    """L2 MIQP with 3 power levels and activation penalty using pyscipopt directly.

    Identical formulation to gurobi_methods.constrained_v6 but using pyscipopt
    instead of gurobipy.  Bypasses CVXPY entirely to avoid the malloc crash that
    occurred when CVXPY built large symbolic graphs before passing them to SCIP.

    Variables
    ---------
    z[i,t,k]  binary  device i at timestep t is in power level k (k=0 low, 1 mid, 2 high)
    x[i,t]    binary  aggregate ON indicator = Σ_k z[i,t,k]
    u[i,t]    binary  rising edge (device turns ON)
    dw[i,t]   binary  falling edge (device turns OFF)
    r[t]      cont    reconstruction residual at t  = Σ_ik p_ik z_itk - y_t
    q[t]      cont≥0  epigraph variable for r[t]²

    Objective
    ---------
    min  Σ_t q[t]
       + Σ_{i,t out-of-window} time_window_penalty * p_i² * x[i,t]
       + Σ_{i,t valid} activation_penalty * p_i² * u[i,t]

    subject to
      q[t] ≥ r[t]²                    (quadratic constraint, one per valid timestep)
      r[t] = Σ_ik p_ik z_itk - y_t   (linear, one per valid timestep)
      x[i,t] = Σ_k z[i,t,k]          (link, one per device-timestep)
      u[i,0]  = x[i,0]               (init)
      dw[i,0] = 0
      x[i,t] - x[i,t-1] = u[i,t] - dw[i,t]   (transition)
      u[i,t] + dw[i,t]  ≤ 1                    (mutex)
      Σ_{τ=t}^{t+a-1} x[i,τ] ≥ a * u[i,t]     (min ON duration)
      Σ_{τ=t}^{t+b}   x[i,τ] ≤ b               (max consecutive ON)

    Args:
        signal: Aggregate power series (any regular DatetimeIndex frequency).
        devices: Device profiles to disaggregate.
        time_limit: SCIP time limit in seconds.
        verbose: If True, keep SCIP console output.
        min_on_slots_by_device: Maps device name → minimum consecutive ON timesteps.
        max_consecutive_on_by_device: Maps device name → maximum consecutive ON timesteps.
        allowed_on_windows_by_device: Maps device name → allowed daily windows as
            list of (start_min, end_min) in minutes-of-day (soft penalty).
        time_window_penalty: Dimensionless factor for out-of-window penalty (× p_typical²).
        power_level_variation: Fractional spread of low/high levels around p_typical (default 0.15).
        activation_penalty: Dimensionless factor for per-activation penalty (× p_typical²).

    Returns:
        (disaggregation, info) where disaggregation maps device name → power series
        with actual estimated wattage (not just p_typical), and info contains solver
        metadata (status, has_solution, is_optimal, objective_value, runtime_sec, mip_gap).
    """
    try:
        from pyscipopt import Model, quicksum as scip_sum
    except ImportError as exc:
        raise ImportError(
            "pyscipopt is required. Install with: pip install pyscipopt"
        ) from exc

    T = len(signal)
    n = len(devices)
    K = 3

    zero_output = {d.name: pd.Series(0.0, index=signal.index, dtype=float) for d in devices}
    empty_info: dict = {
        "status": None, "has_solution": False, "is_optimal": False,
        "objective_value": None, "runtime_sec": 0.0, "mip_gap": None,
        "n_devices": n, "n_timesteps": T,
    }

    if T == 0 or n == 0:
        return zero_output, empty_info

    valid = signal.notna().to_numpy()
    y = signal.to_numpy(dtype=float)
    y_safe = np.where(valid, y, 0.0)

    # Power levels per device: [low, mid, high]
    power_levels: list[list[float]] = [
        [
            dev.p_typical_w * (1.0 - power_level_variation),
            dev.p_typical_w,
            dev.p_typical_w * (1.0 + power_level_variation),
        ]
        for dev in devices
    ]

    # Pre-process constraint dicts into index-keyed form
    min_on_slots: dict[int, int] = {}
    if min_on_slots_by_device:
        for i, dev in enumerate(devices):
            a = min_on_slots_by_device.get(dev.name)
            if a is not None and int(a) >= 2:
                min_on_slots[i] = int(a)

    max_consecutive_limits: dict[int, int] = {}
    if max_consecutive_on_by_device:
        for i, dev in enumerate(devices):
            lim = max_consecutive_on_by_device.get(dev.name)
            if lim is not None and int(lim) >= 1:
                max_consecutive_limits[i] = int(lim)

    # out_of_window_masks[i] = bool array, True = slot is OUTSIDE allowed window
    out_of_window_masks: dict[int, np.ndarray] = {}
    if allowed_on_windows_by_device:
        for i, dev in enumerate(devices):
            windows = allowed_on_windows_by_device.get(dev.name)
            if windows:
                in_window = _build_allowed_window_mask(signal.index, windows)
                out_of_window_masks[i] = ~in_window

    model = Model("nilm_scip_v6")
    model.setParam("limits/time", time_limit)
    if not verbose:
        model.hideOutput()

    # ---- Decision variables ------------------------------------------------ #
    z: dict[tuple[int, int, int], object] = {}
    x: dict[tuple[int, int], object] = {}
    u: dict[tuple[int, int], object] = {}
    dw: dict[tuple[int, int], object] = {}

    for i in range(n):
        for t in range(T):
            x[i, t] = model.addVar(vtype="B", name=f"x_{i}_{t}")
            u[i, t] = model.addVar(vtype="B", name=f"u_{i}_{t}")
            dw[i, t] = model.addVar(vtype="B", name=f"dw_{i}_{t}")
            for k in range(K):
                z[i, t, k] = model.addVar(vtype="B", name=f"z_{i}_{t}_{k}")

    # Residual and epigraph variables (valid timesteps only)
    r: dict[int, object] = {}
    q: dict[int, object] = {}
    for t in range(T):
        if valid[t]:
            r[t] = model.addVar(vtype="C", lb=None, ub=None, name=f"r_{t}")
            q[t] = model.addVar(vtype="C", lb=0.0, name=f"q_{t}")

    # ---- Fix variables to 0 at NaN timesteps ------------------------------- #
    for t in range(T):
        if not valid[t]:
            for i in range(n):
                model.chgVarUb(x[i, t], 0.0)
                model.chgVarUb(u[i, t], 0.0)
                model.chgVarUb(dw[i, t], 0.0)
                for k in range(K):
                    model.chgVarUb(z[i, t, k], 0.0)

    # ---- Link x to z: x[i,t] = Σ_k z[i,t,k] ------------------------------ #
    for i in range(n):
        for t in range(T):
            model.addCons(
                x[i, t] == scip_sum(z[i, t, k] for k in range(K)),
                name=f"link_{i}_{t}",
            )

    # ---- Residual constraints: r[t] = Σ_ik p_ik z_itk - y_t --------------- #
    for t in r:
        power_sum_t = scip_sum(
            power_levels[i][k] * z[i, t, k]
            for i in range(n)
            for k in range(K)
        )
        model.addCons(r[t] == power_sum_t - y_safe[t], name=f"resid_{t}")

    # ---- Epigraph: q[t] >= r[t]^2 ----------------------------------------- #
    for t in q:
        model.addCons(q[t] >= r[t] * r[t], name=f"epigraph_{t}")

    # ---- Transition constraints -------------------------------------------- #
    for i in range(n):
        model.addCons(u[i, 0] == x[i, 0], name=f"init_u_{i}")
        model.addCons(dw[i, 0] == 0.0, name=f"init_dw_{i}")
        for t in range(1, T):
            model.addCons(
                x[i, t] - x[i, t - 1] == u[i, t] - dw[i, t],
                name=f"trans_{i}_{t}",
            )
            model.addCons(u[i, t] + dw[i, t] <= 1, name=f"mutex_{i}_{t}")

    # ---- Min ON duration ---------------------------------------------------- #
    for i, a_i in min_on_slots.items():
        for t in range(T - a_i + 1):
            model.addCons(
                scip_sum(x[i, tau] for tau in range(t, t + a_i)) >= a_i * u[i, t],
                name=f"mindur_{i}_{t}",
            )

    # ---- Max consecutive ON ------------------------------------------------- #
    for i, b_i in max_consecutive_limits.items():
        if b_i >= T:
            continue
        for t in range(T - b_i):
            model.addCons(
                scip_sum(x[i, tau] for tau in range(t, t + b_i + 1)) <= b_i,
                name=f"maxdur_{i}_{t}",
            )

    # ---- Objective: Σ q[t] + time-window penalty + activation penalty ------- #
    obj_expr = scip_sum(q[t] for t in q)

    # Soft time-window penalty (× p_typical² per out-of-window slot)
    for i, mask in out_of_window_masks.items():
        p_sq = devices[i].p_typical_w ** 2
        for t in range(T):
            if mask[t] and valid[t]:
                obj_expr = obj_expr + (time_window_penalty * p_sq) * x[i, t]

    # Per-activation penalty (× p_typical² per ON transition)
    for i, dev in enumerate(devices):
        p_sq = dev.p_typical_w ** 2
        for t in range(T):
            if valid[t]:
                obj_expr = obj_expr + (activation_penalty * p_sq) * u[i, t]

    model.setObjective(obj_expr, sense="minimize")

    # ---- Solve ------------------------------------------------------------- #
    model.optimize()

    status = model.getStatus()
    has_sol = model.getNSols() > 0
    is_optimal = status == "optimal"

    try:
        gap = model.getGap() if has_sol else None
    except Exception:
        gap = None

    info: dict = {
        "status": status,
        "has_solution": has_sol,
        "is_optimal": is_optimal,
        "objective_value": model.getObjVal() if has_sol else None,
        "runtime_sec": model.getSolvingTime(),
        "mip_gap": gap,
        "n_devices": n,
        "n_timesteps": T,
    }

    if not has_sol:
        return zero_output, info

    # ---- Extract solution -------------------------------------------------- #
    disaggregation: dict[str, pd.Series] = {}
    for i, dev in enumerate(devices):
        power_sol = np.array(
            [
                sum(power_levels[i][k] * round(model.getVal(z[i, t, k])) for k in range(K))
                for t in range(T)
            ],
            dtype=float,
        )
        disaggregation[dev.name] = pd.Series(power_sol, index=signal.index, dtype=float)

    return disaggregation, info
