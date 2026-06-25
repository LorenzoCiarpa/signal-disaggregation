#!/usr/bin/env python3
"""Validate questionnaire time windows against real power signal.

For each house (IMEI) and each target device (Lavatrice, Lavastoviglie, Forno):
- Detects power events in the aggregate signal using p_min_w excess over baseline.
- Checks what fraction of days show an event inside the declared time window.
- Compares energy consumed in the window against the expected energy (p_min_w * duration).
- Saves one plot per (IMEI, device) pair.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.nilm.preprocessing import get_usable_imeis

SIGNAL_DIR = REPO_ROOT / "json"
QUESTIONNAIRE_DIR = REPO_ROOT / "device_usage_by_imei_v2"
OUTPUT_DIR = REPO_ROOT / "analysis" / "time_window_validation"

# Only devices with a declared time window and explicit cycle duration
TARGET_DEVICES = ["Lavatrice", "Lavastoviglie", "Forno"]

# p_min_w: minimum physically plausible power when the device is running
# dur_typical_min: fallback typical duration if questionnaire gives no bounds
DEVICE_KB: dict[str, dict] = {
    "Lavatrice":    {"p_min_w": 300.0,  "dur_typical_min": 90.0},
    "Lavastoviglie":{"p_min_w": 200.0,  "dur_typical_min": 120.0},
    "Forno":        {"p_min_w": 500.0,  "dur_typical_min": 45.0},
}

# Rolling-minimum window used to estimate "no device" baseline
BASELINE_WINDOW_MIN = 30
# How many minutes of data a day must have to be included
MIN_DAY_COVERAGE_MIN = 60 * 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_id(value: object) -> int:
    if isinstance(value, dict) and "$numberLong" in value:
        return int(value["$numberLong"])
    return int(value)


def load_signal(imei: str) -> pd.Series:
    path = SIGNAL_DIR / f"iot_devices.{imei}.json"
    with open(path) as f:
        records = json.load(f)

    rows = []
    for rec in records:
        epoch = _parse_id(rec["_id"])
        w = rec.get("w_medio")
        if rec.get("epoch_valid", True) and w is not None and 0 <= float(w) <= 15000:
            rows.append((epoch, float(w)))

    if not rows:
        return pd.Series(dtype=float)

    epochs, values = zip(*rows)
    index = pd.to_datetime(list(epochs), unit="s", utc=True)
    series = pd.Series(list(values), index=index, dtype=float, name="w_medio").sort_index()
    full_idx = pd.date_range(series.index.min().floor("1min"),
                             series.index.max().ceil("1min"),
                             freq="1min", tz="UTC")
    return series.resample("1min").mean().reindex(full_idx)


def load_questionnaire(imei: str) -> dict:
    path = QUESTIONNAIRE_DIR / f"{imei}.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def rolling_baseline(signal: pd.Series) -> pd.Series:
    """Estimate always-on baseline with a rolling minimum."""
    return signal.rolling(BASELINE_WINDOW_MIN, min_periods=1, center=True).min()


def find_events(
    excess: pd.Series,
    p_min_w: float,
    dur_min_min: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) pairs for contiguous runs where excess >= p_min_w."""
    active = (excess >= p_min_w).fillna(False)
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None

    for ts, val in active.items():
        if val and start is None:
            start = ts
        elif not val and start is not None:
            duration_min = (ts - start).total_seconds() / 60
            if duration_min >= dur_min_min:
                events.append((start, ts))
            start = None

    if start is not None:
        duration_min = (active.index[-1] - start).total_seconds() / 60
        if duration_min >= dur_min_min:
            events.append((start, active.index[-1]))

    return events


def overlaps_window(
    ev_start: pd.Timestamp,
    ev_end: pd.Timestamp,
    window_start_min: int,
    window_end_min: int,
    wraps: bool,
) -> bool:
    """True when the event overlaps the declared daily time window."""
    s = ev_start.hour * 60 + ev_start.minute
    e = ev_end.hour * 60 + ev_end.minute
    if wraps:
        return s >= window_start_min or e <= window_end_min
    return s < window_end_min and e > window_start_min


def minutes_of_day_mask(index: pd.DatetimeIndex, start_min: int, end_min: int, wraps: bool) -> np.ndarray:
    tod = index.hour * 60 + index.minute
    if wraps:
        return (tod >= start_min) | (tod <= end_min)
    return (tod >= start_min) & (tod <= end_min)


# ---------------------------------------------------------------------------
# Per-IMEI / per-device analysis
# ---------------------------------------------------------------------------

def _typical_duration(device_info: dict, device: str) -> float:
    lo = device_info.get("duration_minutes_min")
    hi = device_info.get("duration_minutes_max")
    if lo is not None and hi is not None:
        return (lo + hi) / 2.0
    if lo is not None:
        return lo
    if hi is not None:
        return hi
    return DEVICE_KB[device]["dur_typical_min"]


def analyze(imei: str, device: str) -> dict | None:
    questionnaire = load_questionnaire(imei)
    device_info = questionnaire["devices"].get(device, {})

    if not device_info.get("present", False):
        return None

    start_min = device_info.get("start_window_start_min")
    end_min = device_info.get("start_window_end_min")
    if start_min is None or end_min is None:
        return None

    wraps = bool(device_info.get("start_window_wraps_next_day", False))
    p_min = DEVICE_KB[device]["p_min_w"]
    dur_typical = _typical_duration(device_info, device)
    expected_energy_wh = p_min * dur_typical / 60.0
    # Detection uses a fixed minimum so we catch the high-power heating segment
    # even though the full cycle lasts longer at lower power.
    dur_min = 15.0

    signal = load_signal(imei)
    if signal.empty:
        return None

    baseline = rolling_baseline(signal)
    excess = (signal - baseline).clip(lower=0)

    events = find_events(excess, p_min, dur_min)

    # local date from UTC timestamps (convert to local naive for readability)
    local_signal = signal.copy()
    local_signal.index = signal.index.tz_localize(None) if signal.index.tz is None else signal.index.tz_convert("Europe/Rome").tz_localize(None)
    local_excess = excess.copy()
    local_excess.index = local_signal.index

    local_events = []
    for ev_s, ev_e in events:
        ls = ev_s.tz_convert("Europe/Rome").tz_localize(None)
        le = ev_e.tz_convert("Europe/Rome").tz_localize(None)
        local_events.append((ls, le))

    days = sorted(set(local_signal.index.date))
    daily: list[dict] = []

    for day in days:
        day_mask = local_signal.index.date == day
        day_signal = local_signal[day_mask]
        if len(day_signal.dropna()) < MIN_DAY_COVERAGE_MIN:
            continue

        day_excess = local_excess[day_mask]

        day_events = [(s, e) for s, e in local_events if s.date() == day]
        in_window = [ev for ev in day_events if overlaps_window(ev[0], ev[1], start_min, end_min, wraps)]
        out_window = [ev for ev in day_events if not overlaps_window(ev[0], ev[1], start_min, end_min, wraps)]

        win_mask = minutes_of_day_mask(day_excess.index, start_min, end_min, wraps)
        energy_window_wh = float(day_excess[win_mask].sum() / 60.0)
        energy_total_wh = float(day_excess.sum() / 60.0)

        daily.append({
            "date": day,
            "has_event_in_window": len(in_window) > 0,
            "has_event_out_window": len(out_window) > 0,
            "n_events_in_window": len(in_window),
            "n_events_out_window": len(out_window),
            "n_events_total": len(day_events),
            "energy_window_wh": energy_window_wh,
            "energy_total_wh": energy_total_wh,
        })

    return {
        "imei": imei,
        "device": device,
        "start_window_start_min": start_min,
        "start_window_end_min": end_min,
        "wraps": wraps,
        "p_min_w": p_min,
        "dur_min_min": dur_min,
        "dur_typical_min": dur_typical,
        "expected_energy_wh": expected_energy_wh,
        "local_signal": local_signal,
        "local_excess": local_excess,
        "local_events": local_events,
        "daily": daily,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _min_to_hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def plot_result(result: dict, output_path: Path) -> None:
    imei = result["imei"]
    device = result["device"]
    daily = result["daily"]
    local_signal = result["local_signal"]
    local_excess = result["local_excess"]
    local_events = result["local_events"]
    start_min = result["start_window_start_min"]
    end_min = result["start_window_end_min"]
    wraps = result["wraps"]
    p_min = result["p_min_w"]
    expected_wh = result["expected_energy_wh"]

    if not daily:
        return

    days = [d["date"] for d in daily]
    n_days = len(days)
    day_index = {d: i for i, d in enumerate(days)}

    # Build heatmap matrix: rows=days, cols=minutes of day (0..1439)
    heatmap = np.full((n_days, 1440), np.nan)
    for day, i in day_index.items():
        mask = local_excess.index.date == day
        day_ex = local_excess[mask]
        for ts, val in day_ex.items():
            col = ts.hour * 60 + ts.minute
            if 0 <= col < 1440 and not np.isnan(val):
                heatmap[i, col] = val

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1.2], hspace=0.45)
    fig.suptitle(f"IMEI {imei} — {device}\n"
                 f"Fascia dichiarata: {_min_to_hhmm(start_min)}–{_min_to_hhmm(end_min)}  |  "
                 f"Soglia evento: ≥{p_min:.0f} W  |  "
                 f"Energia attesa: {expected_wh:.0f} Wh",
                 fontsize=11)

    # --- Panel 1: heatmap ---
    ax1 = fig.add_subplot(gs[0])
    vmax = np.nanpercentile(heatmap, 95) if not np.all(np.isnan(heatmap)) else 1000
    vmax = max(vmax, p_min * 2)
    im = ax1.imshow(heatmap, aspect="auto", origin="upper",
                    vmin=0, vmax=vmax, cmap="YlOrRd",
                    extent=[0, 1440, n_days - 0.5, -0.5])
    plt.colorbar(im, ax=ax1, label="Eccesso potenza (W)", fraction=0.02, pad=0.01)

    # Shade declared time window
    if wraps:
        ax1.axvspan(start_min, 1440, color="steelblue", alpha=0.15, label="Fascia oraria")
        ax1.axvspan(0, end_min, color="steelblue", alpha=0.15)
    else:
        ax1.axvspan(start_min, end_min, color="steelblue", alpha=0.15, label="Fascia oraria")

    # Mark events
    for ev_s, ev_e in local_events:
        day = ev_s.date()
        if day not in day_index:
            continue
        row = day_index[day]
        col_s = ev_s.hour * 60 + ev_s.minute
        col_e = ev_e.hour * 60 + ev_e.minute
        in_win = overlaps_window(ev_s, ev_e, start_min, end_min, wraps)
        color = "green" if in_win else "orange"
        ax1.plot([col_s, col_e], [row, row], color=color, linewidth=3, solid_capstyle="round")

    ax1.set_xlabel("Ora del giorno")
    ax1.set_ylabel("Giorno")
    xticks = list(range(0, 1440, 120))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=7)
    ytick_step = max(1, n_days // 10)
    ax1.set_yticks(range(0, n_days, ytick_step))
    ax1.set_yticklabels([str(days[i]) for i in range(0, n_days, ytick_step)], fontsize=7)

    green_patch = mpatches.Patch(color="green", label="Evento in fascia")
    orange_patch = mpatches.Patch(color="orange", label="Evento fuori fascia")
    blue_patch = mpatches.Patch(color="steelblue", alpha=0.4, label="Fascia dichiarata")
    ax1.legend(handles=[blue_patch, green_patch, orange_patch], fontsize=8, loc="upper right")

    # --- Panel 2: energy per day ---
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(n_days)
    bars_in = [d["energy_window_wh"] for d in daily]
    colors = ["green" if d["has_event_in_window"] else "lightcoral" for d in daily]
    ax2.bar(x, bars_in, color=colors, width=0.7, label="Energia in fascia (eccesso, Wh)")
    ax2.axhline(expected_wh, color="navy", linewidth=1.5, linestyle="--",
                label=f"Energia attesa ({expected_wh:.0f} Wh)")
    ax2.set_ylabel("Energia eccesso in fascia (Wh)")
    ax2.set_xlabel("Giorno")
    ax2.set_xticks(x[::ytick_step])
    ax2.set_xticklabels([str(days[i]) for i in range(0, n_days, ytick_step)], fontsize=7, rotation=30)
    ax2.legend(fontsize=8)

    # --- Panel 3: summary stats ---
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    n = len(daily)
    n_in = sum(d["has_event_in_window"] for d in daily)
    n_out_only = sum(d["has_event_out_window"] and not d["has_event_in_window"] for d in daily)
    n_none = sum(not d["has_event_in_window"] and not d["has_event_out_window"] for d in daily)
    mean_energy = np.mean(bars_in)
    freq_per_week = (result.get("dur_typical_min") or 1)  # placeholder

    lines = [
        f"Giorni analizzati: {n}  |  "
        f"Con evento in fascia: {n_in} ({100*n_in/n:.1f}%)  |  "
        f"Evento solo fuori fascia: {n_out_only} ({100*n_out_only/n:.1f}%)  |  "
        f"Nessun evento: {n_none} ({100*n_none/n:.1f}%)",
        f"Energia media in fascia: {mean_energy:.1f} Wh  |  "
        f"Energia attesa: {expected_wh:.1f} Wh  |  "
        f"Rapporto medio: {mean_energy/expected_wh:.2f}x" if expected_wh > 0 else "",
    ]
    ax3.text(0.01, 0.65, lines[0], transform=ax3.transAxes, fontsize=9, va="top")
    ax3.text(0.01, 0.25, lines[1], transform=ax3.transAxes, fontsize=9, va="top")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_overlap_days(result: dict, output_path: Path) -> None:
    """Overlay plot: one line per day-with-event-in-window, x-axis = hour of day.

    Shows how the actual signal excess aligns with the declared time window
    on the days where the questionnaire prediction was correct.
    A second panel shows the same for days with event *outside* the window,
    for direct comparison.
    """
    imei = result["imei"]
    device = result["device"]
    daily = result["daily"]
    local_excess = result["local_excess"]
    local_events = result["local_events"]
    start_min = result["start_window_start_min"]
    end_min = result["start_window_end_min"]
    wraps = result["wraps"]
    p_min = result["p_min_w"]

    days_in = [d["date"] for d in daily if d["has_event_in_window"]]
    days_out = [d["date"] for d in daily if d["has_event_out_window"] and not d["has_event_in_window"]]

    if not days_in:
        return

    minutes = np.arange(1440)

    def day_to_array(day: date) -> np.ndarray:
        mask = local_excess.index.date == day
        day_ex = local_excess[mask]
        arr = np.full(1440, np.nan)
        for ts, val in day_ex.items():
            col = ts.hour * 60 + ts.minute
            if 0 <= col < 1440:
                arr[col] = val if not np.isnan(val) else 0.0
        return arr

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle(
        f"IMEI {imei} — {device}  |  "
        f"Fascia dichiarata: {_min_to_hhmm(start_min)}–{_min_to_hhmm(end_min)}  |  "
        f"Soglia: ≥{p_min:.0f} W eccesso",
        fontsize=11,
    )

    xticks = list(range(0, 1440, 120))
    xlabels = [f"{h:02d}:00" for h in range(0, 24, 2)]

    def shade_window(ax):
        if wraps:
            ax.axvspan(start_min, 1440, color="steelblue", alpha=0.12, label="Fascia dichiarata")
            ax.axvspan(0, end_min, color="steelblue", alpha=0.12)
        else:
            ax.axvspan(start_min, end_min, color="steelblue", alpha=0.12, label="Fascia dichiarata")
        ax.axhline(p_min, color="red", linewidth=1.0, linestyle="--", alpha=0.6, label=f"Soglia {p_min:.0f} W")

    def mark_events_on_day(ax, day: date, color: str):
        for ev_s, ev_e in local_events:
            if ev_s.date() != day:
                continue
            s = ev_s.hour * 60 + ev_s.minute
            e = ev_e.hour * 60 + ev_e.minute
            ax.axvspan(s, e, color=color, alpha=0.25, linewidth=0)

    # Panel A: days with event in window
    ax = axes[0]
    cmap_in = plt.get_cmap("Blues")
    for i, day in enumerate(days_in):
        arr = day_to_array(day)
        color = cmap_in(0.4 + 0.55 * i / max(len(days_in) - 1, 1))
        ax.plot(minutes, arr, color=color, linewidth=0.8, alpha=0.7)
        mark_events_on_day(ax, day, "green")
    shade_window(ax)
    # Mean line
    stacked = np.vstack([day_to_array(d) for d in days_in])
    mean_line = np.nanmean(stacked, axis=0)
    ax.plot(minutes, mean_line, color="black", linewidth=1.8, label=f"Media ({len(days_in)} giorni)")
    ax.set_title(f"Giorni con evento IN fascia ({len(days_in)})", fontsize=10)
    ax.set_xlabel("Ora del giorno")
    ax.set_ylabel("Eccesso potenza (W)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1439)

    # Panel B: days with event outside window (comparison)
    ax = axes[1]
    if days_out:
        cmap_out = plt.get_cmap("Oranges")
        for i, day in enumerate(days_out):
            arr = day_to_array(day)
            color = cmap_out(0.4 + 0.55 * i / max(len(days_out) - 1, 1))
            ax.plot(minutes, arr, color=color, linewidth=0.8, alpha=0.7)
            mark_events_on_day(ax, day, "orange")
        stacked_out = np.vstack([day_to_array(d) for d in days_out])
        mean_out = np.nanmean(stacked_out, axis=0)
        ax.plot(minutes, mean_out, color="black", linewidth=1.8, label=f"Media ({len(days_out)} giorni)")
        ax.set_title(f"Giorni con evento FUORI fascia ({len(days_out)})", fontsize=10)
    else:
        ax.set_title("Nessun giorno con evento fuori fascia", fontsize=10)
    shade_window(ax)
    ax.set_xlabel("Ora del giorno")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1439)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_daily_overlap(result: dict, out_dir: Path) -> None:
    """One PNG per day: raw signal + excess, with time window and detected events."""
    imei = result["imei"]
    device = result["device"]
    daily = result["daily"]
    local_signal = result["local_signal"]
    local_excess = result["local_excess"]
    local_events = result["local_events"]
    start_min = result["start_window_start_min"]
    end_min = result["start_window_end_min"]
    wraps = result["wraps"]
    p_min = result["p_min_w"]
    expected_wh = result["expected_energy_wh"]

    out_dir.mkdir(parents=True, exist_ok=True)

    xticks = list(range(0, 1440, 120))
    xlabels = [f"{h:02d}:00" for h in range(0, 24, 2)]

    for day_info in daily:
        day = day_info["date"]
        mask = local_signal.index.date == day
        day_raw = local_signal[mask]
        day_exc = local_excess[mask]

        minutes_raw = day_raw.index.hour * 60 + day_raw.index.minute
        minutes_exc = day_exc.index.hour * 60 + day_exc.index.minute

        day_events = [(s, e) for s, e in local_events if s.date() == day]
        in_win = day_info["has_event_in_window"]
        out_win = day_info["has_event_out_window"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        status = "IN fascia" if in_win else ("FUORI fascia" if out_win else "Nessun evento")
        status_color = "green" if in_win else ("orange" if out_win else "gray")
        fig.suptitle(
            f"IMEI {imei} — {device} — {day}  |  "
            f"Fascia: {_min_to_hhmm(start_min)}–{_min_to_hhmm(end_min)}  "
            f"[{status}]",
            fontsize=10,
            color=status_color,
            fontweight="bold",
        )

        def shade(ax):
            if wraps:
                ax.axvspan(start_min, 1440, color="steelblue", alpha=0.12)
                ax.axvspan(0, end_min, color="steelblue", alpha=0.12, label="Fascia dichiarata")
            else:
                ax.axvspan(start_min, end_min, color="steelblue", alpha=0.12, label="Fascia dichiarata")

        # Raw signal
        ax1.plot(minutes_raw, day_raw.values, color="steelblue", linewidth=1.0, label="Segnale grezzo (W)")
        shade(ax1)
        for ev_s, ev_e in day_events:
            s = ev_s.hour * 60 + ev_s.minute
            e = ev_e.hour * 60 + ev_e.minute
            in_ev = overlaps_window(ev_s, ev_e, start_min, end_min, wraps)
            c = "green" if in_ev else "orange"
            ax1.axvspan(s, e, color=c, alpha=0.3)
        ax1.set_ylabel("Potenza (W)")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_xlim(0, 1439)

        # Excess over baseline
        ax2.fill_between(minutes_exc, 0, day_exc.values, color="tomato", alpha=0.5, label="Eccesso sul baseline (W)")
        ax2.axhline(p_min, color="red", linewidth=1.0, linestyle="--", alpha=0.7, label=f"Soglia {p_min:.0f} W")
        energy_wh = day_info["energy_window_wh"]
        ax2.axhline(0, color="black", linewidth=0.4)
        shade(ax2)
        for ev_s, ev_e in day_events:
            s = ev_s.hour * 60 + ev_s.minute
            e = ev_e.hour * 60 + ev_e.minute
            in_ev = overlaps_window(ev_s, ev_e, start_min, end_min, wraps)
            c = "green" if in_ev else "orange"
            ax2.axvspan(s, e, color=c, alpha=0.3)
        ax2.set_ylabel("Eccesso (W)")
        ax2.set_xlabel("Ora del giorno")
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels, fontsize=8)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_xlim(0, 1439)

        # Annotation box
        txt = (f"Energia in fascia: {energy_wh:.0f} Wh  |  Attesa: {expected_wh:.0f} Wh  |  "
               f"Rapporto: {energy_wh/expected_wh:.2f}x" if expected_wh > 0 else "")
        ax2.annotate(txt, xy=(0.01, 0.97), xycoords="axes fraction",
                     fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        fname = out_dir / f"{day}.png"
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(imeis: list[str] | None = None) -> None:
    if imeis is None:
        imeis = get_usable_imeis()
    print(f"Processing {len(imeis)} IMEIs × {len(TARGET_DEVICES)} devices")

    all_stats: list[dict] = []

    for imei in imeis:
        q_path = QUESTIONNAIRE_DIR / f"{imei}.json"
        if not q_path.exists():
            print(f"  [skip] No questionnaire for {imei}")
            continue

        for device in TARGET_DEVICES:
            print(f"  {imei} / {device} ...", end=" ", flush=True)
            result = analyze(imei, device)
            if result is None:
                print("skipped (not present or no time window)")
                continue

            daily = result["daily"]
            n = len(daily)
            n_in = sum(d["has_event_in_window"] for d in daily)
            print(f"{n} days, {n_in} ({100*n_in/n:.0f}%) with event in window")

            fname = OUTPUT_DIR / f"{imei}_{device.replace(' ', '_')}.png"
            plot_result(result, fname)

            fname_ov = OUTPUT_DIR / f"{imei}_{device.replace(' ', '_')}_overlap.png"
            plot_overlap_days(result, fname_ov)

            daily_dir = OUTPUT_DIR / imei / device.replace(" ", "_")
            plot_daily_overlap(result, daily_dir)
            print(f"    Daily plots -> {daily_dir}")

            mean_energy = float(np.mean([d["energy_window_wh"] for d in daily]))
            all_stats.append({
                "imei": imei,
                "device": device,
                "n_days": n,
                "n_event_in_window": n_in,
                "pct_event_in_window": round(100 * n_in / n, 1) if n else None,
                "n_event_out_only": sum(d["has_event_out_window"] and not d["has_event_in_window"] for d in daily),
                "n_no_event": sum(not d["has_event_in_window"] and not d["has_event_out_window"] for d in daily),
                "mean_energy_window_wh": round(mean_energy, 1),
                "expected_energy_wh": round(result["expected_energy_wh"], 1),
                "ratio_energy": round(mean_energy / result["expected_energy_wh"], 3) if result["expected_energy_wh"] > 0 else None,
                "start_window": _min_to_hhmm(result["start_window_start_min"]),
                "end_window": _min_to_hhmm(result["start_window_end_min"]),
                "p_min_w": result["p_min_w"],
                "dur_typical_min": result["dur_typical_min"],
            })

    if all_stats:
        csv_path = OUTPUT_DIR / "summary.csv"
        pd.DataFrame(all_stats).to_csv(csv_path, index=False)
        print(f"\nSummary CSV: {csv_path}")

    print("Done.")


def _min_to_hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--imeis", nargs="+", help="Run only on these IMEIs (space-separated)")
    args = parser.parse_args()
    main(imeis=args.imeis)
