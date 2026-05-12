"""
US-008: Output module — CSV, weekly temporal plots, and energy bar chart.

Provides save_results() to write disaggregation outputs for any (imei, approach) pair.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def save_results(
    signal: pd.Series,
    disaggregation: dict,
    imei: str,
    approach_name: str,
    output_dir: str = "analysis",
    skip_weekly_plots: bool = False,
    temporal_plot_granularity: str = "weekly",
) -> None:
    """Save disaggregation CSV, temporal plots, and energy report for one (imei, approach) pair.

    Args:
        signal: Aggregate power signal (pd.Series with DatetimeIndex).
        disaggregation: dict[str, pd.Series] — per-device estimated power series.
        imei: IMEI identifier string.
        approach_name: Name of the disaggregation approach.
        output_dir: Root output directory (default: 'analysis').
        skip_weekly_plots: If True, skip saving weekly PNG plots (default: False).
        temporal_plot_granularity: 'weekly' or 'daily' for temporal plot output.
    """
    base_dir = os.path.join(output_dir, imei, approach_name)
    temporal_dir_name = "daily_plots" if temporal_plot_granularity == "daily" else "weekly_plots"
    temporal_dir = os.path.join(base_dir, temporal_dir_name)
    os.makedirs(temporal_dir, exist_ok=True)

    _save_csv(signal, disaggregation, base_dir)
    if not skip_weekly_plots:
        if temporal_plot_granularity == "daily":
            _save_daily_plots(signal, disaggregation, imei, approach_name, temporal_dir)
        else:
            _save_weekly_plots(signal, disaggregation, imei, approach_name, temporal_dir)
    _save_energy_report(signal, disaggregation, imei, approach_name, base_dir)


def _save_csv(signal: pd.Series, disaggregation: dict, base_dir: str) -> None:
    """Save disaggregation.csv with columns [timestamp, w_total, device..., residuo]."""
    df = pd.DataFrame({"w_total": signal})
    for device_name, series in disaggregation.items():
        df[device_name] = series

    device_cols = list(disaggregation.keys())
    device_sum = df[device_cols].fillna(0).sum(axis=1)
    df["residuo"] = df["w_total"].fillna(0) - device_sum

    df.index.name = "timestamp"
    df.index = df.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(os.path.join(base_dir, "disaggregation.csv"))


def _save_weekly_plots(
    signal: pd.Series,
    disaggregation: dict,
    imei: str,
    approach_name: str,
    weekly_dir: str,
) -> None:
    """Save one PNG per calendar week with stacked device layers and aggregate signal."""
    if signal.empty:
        return

    # Determine which devices have meaningful power (mean > 5W ignoring NaN)
    active_devices = {
        name: series
        for name, series in disaggregation.items()
        if series.fillna(0).mean() > 5.0
    }

    # Group by calendar week.
    week_periods = signal.index.to_period("W-SUN")
    weeks = week_periods.unique().sort_values()
    colors = plt.cm.tab20.colors  # up to 20 distinct colors

    for week in weeks:
        week_start = week.start_time.strftime("%Y-%m-%d")
        week_end = week.end_time.strftime("%Y-%m-%d")
        week_label = f"{week_start}_to_{week_end}"
        mask = week_periods == week
        week_signal = signal[mask]

        if week_signal.dropna().empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 5))

        # Aggregate signal in grey
        ax.plot(
            week_signal.index,
            week_signal.values,
            color="grey",
            linewidth=1.5,
            label="Totale misurato",
            zorder=10,
        )

        # Stacked colored layers for active devices
        for idx, (dev_name, dev_series) in enumerate(active_devices.items()):
            week_dev = dev_series[mask].fillna(0)
            color = colors[idx % len(colors)]
            ax.fill_between(
                week_signal.index,
                0,
                week_dev.values,
                alpha=0.5,
                color=color,
                label=dev_name,
            )

        # Residual in dashed black
        device_sum = sum(
            s[mask].fillna(0) for s in active_devices.values()
        ) if active_devices else pd.Series(0, index=week_signal.index)
        residual = week_signal.fillna(0) - device_sum
        ax.plot(
            week_signal.index,
            residual.values,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="Residuo",
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title(f"{imei} — {week_start} to {week_end} — {approach_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Potenza (W)")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_xlim(week_signal.index[0], week_signal.index[-1])

        fig.tight_layout()
        fig.savefig(os.path.join(weekly_dir, f"{week_label}.png"), dpi=100)
        plt.close(fig)


def _save_daily_plots(
    signal: pd.Series,
    disaggregation: dict,
    imei: str,
    approach_name: str,
    daily_dir: str,
) -> None:
    """Save one PNG per calendar day with stacked device layers and aggregate signal."""
    if signal.empty:
        return

    active_devices = {
        name: series
        for name, series in disaggregation.items()
        if series.fillna(0).mean() > 5.0
    }

    day_periods = signal.index.to_period("D")
    days = day_periods.unique().sort_values()
    colors = plt.cm.tab20.colors

    for day in days:
        day_label = day.start_time.strftime("%Y-%m-%d")
        mask = day_periods == day
        day_signal = signal[mask]

        if day_signal.dropna().empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 5))

        ax.plot(
            day_signal.index,
            day_signal.values,
            color="grey",
            linewidth=1.5,
            label="Totale misurato",
            zorder=10,
        )

        for idx, (dev_name, dev_series) in enumerate(active_devices.items()):
            day_dev = dev_series[mask].fillna(0)
            color = colors[idx % len(colors)]
            ax.fill_between(
                day_signal.index,
                0,
                day_dev.values,
                alpha=0.5,
                color=color,
                label=dev_name,
            )

        device_sum = sum(
            s[mask].fillna(0) for s in active_devices.values()
        ) if active_devices else pd.Series(0, index=day_signal.index)
        residual = day_signal.fillna(0) - device_sum
        ax.plot(
            day_signal.index,
            residual.values,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="Residuo",
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_title(f"{imei} — {day_label} — {approach_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Potenza (W)")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_xlim(day_signal.index[0], day_signal.index[-1])

        fig.tight_layout()
        fig.savefig(os.path.join(daily_dir, f"{day_label}.png"), dpi=100)
        plt.close(fig)


def _save_energy_report(
    signal: pd.Series,
    disaggregation: dict,
    imei: str,
    approach_name: str,
    base_dir: str,
) -> None:
    """Save horizontal bar chart of kWh per device + residual for the full period."""
    # Energy in kWh: sum of watts * (1 min / 60) / 1000
    minutes_per_sample = 1.0
    kwh_factor = minutes_per_sample / 60.0 / 1000.0

    total_kwh = signal.fillna(0).sum() * kwh_factor
    if total_kwh == 0:
        return

    energies = {}
    for dev_name, series in disaggregation.items():
        kwh = series.fillna(0).sum() * kwh_factor
        if kwh > 0:
            energies[dev_name] = kwh

    # Residual
    device_kwh_sum = sum(energies.values())
    residual_kwh = max(0.0, total_kwh - device_kwh_sum)
    energies["Residuo"] = residual_kwh

    names = list(energies.keys())
    values = [energies[n] for n in names]
    total = sum(values) if sum(values) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    colors = plt.cm.tab20.colors
    bars = ax.barh(
        names,
        values,
        color=[colors[i % len(colors)] for i in range(len(names))],
        edgecolor="white",
    )

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} kWh ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Energia (kWh)")
    ax.set_title(f"Energia disaggregata — {imei} — {approach_name}")
    ax.set_xlim(0, max(values) * 1.35 if values else 1)
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "energy_report.png"), dpi=100)
    plt.close(fig)
