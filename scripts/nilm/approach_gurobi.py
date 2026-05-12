"""
Approach OR-MIQP — Mixed-Integer Quadratic Programming disaggregation via Gurobi.

Minimises squared reconstruction error between aggregate signal y(t) and the
sum of per-device binary power contributions p_i * x_i(t), subject to:

  - Probability-based availability constraints (s_i(t) < threshold → forced OFF)
  - State-transition tracking: x_i(t) - x_i(t-1) = u_i(t) - d_i(t)
  - Mutual exclusivity of switching events: u_i(t) + d_i(t) ≤ 1
  - Minimum ON-duration per device
  - Maximum ON-duration per device (rolling-window constraint)
  - Activation penalty in the objective to limit start-up frequency

The signal is segmented by calendar week, and an independent problem is
solved per week for tractability.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

from scripts.nilm.devices import DeviceProfile
from scripts.nilm.gurobi_methods import _solve_week

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run(
    signal: pd.Series,
    devices: list[DeviceProfile],
    lambda_activation: float = 10.0,
    prob_threshold: float = 0.10,
    max_duration_factor: float = 2.0,
    time_limit: float = 60.0,
) -> dict[str, pd.Series]:
    """MIQP-based NILM disaggregation using Gurobi.

    Solves one Gurobi model per calendar week. The objective minimises
    squared reconstruction error plus an activation-frequency penalty.

    Args:
        signal: 1-minute resampled w_medio series with UTC DatetimeIndex.
        devices: List of DeviceProfile instances (all devices, incl. absent).
        lambda_activation: Base penalty per device activation, scaled by
            7 / freq_week_i so that rare devices are penalised more heavily.
        prob_threshold: If s_i(t) < threshold, device i is forced OFF at t.
        max_duration_factor: B_i = ceil(dur_typical_min * factor) caps the
            maximum consecutive ON window via a rolling-window constraint.
        time_limit: Gurobi time limit per weekly chunk in seconds.

    Returns:
        dict mapping device name → pd.Series (same index as *signal*) with
        estimated power in Watts.
    """
    if not _HAS_GUROBI:
        raise ImportError(
            "gurobipy is not installed.  Install it with:\n"
            "    pip install gurobipy\n"
            "and obtain a Gurobi license from "
            "https://www.gurobi.com/academia/academic-program-and-licenses/"
        )

    # Only devices confirmed as present in the household are optimised.
    present_devices = [d for d in devices if d.prior_weight >= 1.0]
    absent_devices = [d for d in devices if d.prior_weight < 1.0]

    if not present_devices:
        return {d.name: pd.Series(0.0, index=signal.index) for d in devices}

    # Iterate over calendar weeks and solve one problem per week.
    week_periods = signal.index.to_period("W-SUN")
    unique_weeks = week_periods.unique().sort_values()
    week_results: dict[str, list[pd.Series]] = {d.name: [] for d in present_devices}

    for idx, week in enumerate(unique_weeks):
        week_mask = week_periods == week
        week_signal = signal[week_mask]
        timestamps = week_signal.index
        y = week_signal.values.astype(float)
        T = len(y)

        if T == 0:
            continue

        logger.debug(
            "OR-MIQP: solving week %s to %s (T=%d)",
            week.start_time.date(),
            week.end_time.date(),
            T,
        )

        week_sol = _solve_week(
            y=y,
            devices=present_devices,
            timestamps=timestamps,
            prob_threshold=prob_threshold,
            max_duration_factor=max_duration_factor,
            lambda_activation=lambda_activation,
            time_limit=time_limit,
        )

        if idx == 2:
            break

        for dev in present_devices:
            arr = week_sol.get(dev.name, np.zeros(T, dtype=float))
            week_results[dev.name].append(pd.Series(arr, index=timestamps, dtype=float))

    # Concatenate weekly chunks and align to original signal index.
    output: dict[str, pd.Series] = {}
    for dev in present_devices:
        if week_results[dev.name]:
            combined = pd.concat(week_results[dev.name])
            output[dev.name] = combined.reindex(signal.index).fillna(0.0)
        else:
            output[dev.name] = pd.Series(0.0, index=signal.index)

    # Absent devices receive all-zero series (consistent with other approaches).
    for dev in absent_devices:
        output[dev.name] = pd.Series(0.0, index=signal.index)

    return output
