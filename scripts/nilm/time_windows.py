"""
Daily time-window utilities shared by the solver modules.

Survey windows are expressed as (start_min, end_min) pairs in minutes-of-day and
may wrap past midnight (end <= start).  Two views are provided:

  build_allowed_window_mask — boolean, "is this timestep inside a window?"
  window_penalty_factors    — graduated factor that grows with the distance from
                              the nearest window, for soft out-of-window penalties.
  window_vagueness          — how little a declaration narrows things down, used to
                              stop devices that declared no window from being free.
  out_of_season_mask        — timesteps in a month the device was not declared active.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24.0

# Defaults for the graduated out-of-window penalty
WINDOW_PENALTY_RAMP_MIN = 60.0
WINDOW_PENALTY_MAX_FACTOR = 6.0


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy(int) * 60 + index.minute.to_numpy(int)


def _allowed_minutes(allowed_windows: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Boolean array over the 1440 minutes of a day: True inside any window."""
    allowed = np.zeros(MINUTES_PER_DAY, dtype=bool)
    minute = np.arange(MINUTES_PER_DAY)
    for start_min, end_min in allowed_windows:
        s, e = int(start_min) % MINUTES_PER_DAY, int(end_min) % MINUTES_PER_DAY
        if e <= s:  # window wraps past midnight
            allowed |= (minute >= s) | (minute < e)
        else:
            allowed |= (minute >= s) & (minute < e)
    return allowed


@lru_cache(maxsize=256)
def _distance_to_window(allowed_windows: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Circular distance in minutes from each minute-of-day to the nearest window.

    Zero inside a window, growing outward in both directions and wrapping across
    midnight.  Cached because windows are static per device.
    """
    allowed = _allowed_minutes(allowed_windows)
    if not allowed.any():
        return np.full(MINUTES_PER_DAY, float(MINUTES_PER_DAY), dtype=float)
    if allowed.all():
        return np.zeros(MINUTES_PER_DAY, dtype=float)

    # Tile three days so a forward + backward sweep handles the circular wrap,
    # then keep the middle day.
    tiled = np.tile(allowed, 3)
    dist = np.where(tiled, 0.0, np.inf)
    for i in range(1, len(dist)):
        dist[i] = min(dist[i], dist[i - 1] + 1.0)
    for i in range(len(dist) - 2, -1, -1):
        dist[i] = min(dist[i], dist[i + 1] + 1.0)
    return dist[MINUTES_PER_DAY : 2 * MINUTES_PER_DAY]


def build_allowed_window_mask(
    index: pd.DatetimeIndex,
    allowed_windows: list[tuple[int, int]],
) -> np.ndarray:
    """Return a boolean mask, True where the timestep falls inside a window."""
    if not allowed_windows:
        return np.ones(len(index), dtype=bool)
    allowed = _allowed_minutes(tuple((int(s), int(e)) for s, e in allowed_windows))
    return allowed[_minute_of_day(index)]


def window_penalty_factors(
    index: pd.DatetimeIndex,
    allowed_windows: list[tuple[int, int]] | None,
    ramp_min: float = WINDOW_PENALTY_RAMP_MIN,
    max_factor: float = WINDOW_PENALTY_MAX_FACTOR,
    vagueness_weight: float = 0.0,
) -> np.ndarray:
    """Graduated out-of-window penalty factor per timestep.

    Returns 0.0 inside a window; outside, a factor that starts at 1.0 on the
    window edge and grows linearly with the distance from it:

        factor(d) = min(max_factor, 1 + d / ramp_min)

    So with the defaults an activation just outside the window costs the base
    penalty, one hour out costs double, and the cost saturates at 6× rather than
    diverging.  The factor multiplies the caller's own penalty scale, so it works
    for both the L2 (× p²) and L1 (× p) objectives.

    Args:
        index: Timestamps of the slots being penalised.
        allowed_windows: (start_min, end_min) pairs in minutes-of-day; may wrap.
        ramp_min: Minutes away from the window that add one unit of penalty.
        max_factor: Upper bound on the distance term.
        vagueness_weight: Weight on a constant floor equal to the device's vagueness
            (see window_vagueness), applied everywhere including inside the window.
            At 0 (default) the factor is 0 inside the window and a device that
            declared no window is never charged at all.  Above 0, a device that
            named a narrow slot pays less than one that named nothing, which is what
            breaks ties between them.

    Returns:
        Float array aligned to index.
    """
    if ramp_min <= 0:
        raise ValueError(f"ramp_min must be positive, got {ramp_min!r}")

    floor = vagueness_weight * window_vagueness(allowed_windows) if vagueness_weight else 0.0

    if not allowed_windows:
        # Nothing declared: no window to be outside of, so only the floor applies.
        return np.full(len(index), floor, dtype=float)

    distance = _distance_to_window(tuple((int(s), int(e)) for s, e in allowed_windows))
    d = distance[_minute_of_day(index)]
    graduated = np.minimum(max_factor, 1.0 + d / ramp_min)
    return floor + np.where(d > 0, graduated, 0.0)


def window_total_hours(allowed_windows: list[tuple[int, int]] | None) -> float:
    """Total width of the declared windows in hours.

    A device that declared nothing is treated as declaring the whole day, which is
    literally what "no preferred time" means.
    """
    if not allowed_windows:
        return HOURS_PER_DAY
    total = 0
    for start_min, end_min in allowed_windows:
        span = (int(end_min) - int(start_min)) % MINUTES_PER_DAY
        total += span or MINUTES_PER_DAY
    return min(HOURS_PER_DAY, total / 60.0)


def window_vagueness(allowed_windows: list[tuple[int, int]] | None) -> float:
    """How uninformative a device's declared schedule is, in [0, 1].

    Derived from the log-likelihood of a uniform prior over the declared window: a
    device active uniformly across W hours has density 1/W, so asserting it runs at a
    given time costs log(W).  Normalising by log(24) puts a one-hour window at 0 and
    a device with no declaration (W = 24h) at 1.

    This is what lets a narrow declaration win a tie.  A device that named a 3-hour
    slot scores 0.35 while one that named nothing scores 1.0, so when both could
    explain the same plateau the specific one is preferred — without ever treating
    "no schedule" as evidence of implausibility, only as absence of information.

    Args:
        allowed_windows: (start_min, end_min) pairs, or None/empty if undeclared.

    Returns:
        Vagueness in [0, 1]; 1.0 when nothing was declared.
    """
    hours = window_total_hours(allowed_windows)
    return math.log(max(hours, 1.0)) / math.log(HOURS_PER_DAY)


def out_of_season_mask(
    index: pd.DatetimeIndex,
    active_months: list[int] | None,
) -> np.ndarray:
    """True where the timestep falls in a month the device was not declared active.

    All-False when no months were declared, so an undeclared device is never charged
    a seasonal penalty.
    """
    if not active_months:
        return np.zeros(len(index), dtype=bool)
    months = {int(m) for m in active_months}
    return ~np.isin(index.month.to_numpy(int), list(months))
