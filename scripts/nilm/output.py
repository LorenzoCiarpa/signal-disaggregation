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


DEVICE_COLOR_MAP = {
    "Frigorifero": "#4E79A7",
    "Frigorifero principale": "#4E79A7",
    "Frigorifero secondario": "#A0CBE8",
    "Congelatore": "#59A14F",
    "Lavatrice": "#E15759",
    "Asciugatrice": "#FF9D9A",
    "Lavastoviglie": "#F28E2B",
    "Forno": "#9C755F",
    "Piano cottura": "#B07AA1",
    "Microonde": "#EDC948",
    "Boiler": "#B6992D",
    "Climatizzatore": "#76B7B2",
    "Televisori": "#AF7AA1",
    "Computer": "#9D7660",
    "Console": "#D37295",
    "Auto elettrica": "#E15759",
    "Residuo": "#BAB0AC",
}


def _device_color(device_name: str) -> str:
    """Return a stable color for a device name across all plots."""
    if device_name in DEVICE_COLOR_MAP:
        return DEVICE_COLOR_MAP[device_name]

    palette = plt.cm.tab20.colors
    return palette[sum(ord(char) for char in device_name) % len(palette)]


def _build_active_device_layers(
    disaggregation: dict,
    mask: pd.Series | np.ndarray,
) -> tuple[list[tuple[str, pd.Series]], pd.Series | None]:
    """Return active device layers and their cumulative sum for the selected time slice."""
    active_layers = []
    cumulative = None

    for name, series in disaggregation.items():
        sliced = series[mask].fillna(0)
        if sliced.mean() <= 5.0:
            continue

        active_layers.append((name, sliced))
        cumulative = sliced.copy() if cumulative is None else cumulative.add(sliced, fill_value=0)

    return active_layers, cumulative


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
        temporal_plot_granularity: 'weekly', 'daily', or 'both' for temporal plot output.
    """
    base_dir = os.path.join(output_dir, imei, approach_name)
    weekly_dir = os.path.join(base_dir, "weekly_plots")
    daily_dir = os.path.join(base_dir, "daily_plots")

    if temporal_plot_granularity in {"weekly", "both"}:
        os.makedirs(weekly_dir, exist_ok=True)
    if temporal_plot_granularity in {"daily", "both"}:
        os.makedirs(daily_dir, exist_ok=True)

    _save_csv(signal, disaggregation, base_dir)
    if not skip_weekly_plots:
        if temporal_plot_granularity in {"weekly", "both"}:
            _save_weekly_plots(signal, disaggregation, imei, approach_name, weekly_dir)
        if temporal_plot_granularity in {"daily", "both"}:
            _save_daily_plots(signal, disaggregation, imei, approach_name, daily_dir)
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

    # Group by calendar week.
    week_periods = signal.index.to_period("W-SUN")
    weeks = week_periods.unique().sort_values()

    for week in weeks:
        week_start = week.start_time.strftime("%Y-%m-%d")
        week_end = week.end_time.strftime("%Y-%m-%d")
        week_label = f"{week_start}_to_{week_end}"
        mask = week_periods == week
        week_signal = signal[mask]

        if week_signal.dropna().empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 5))

        active_layers, device_sum = _build_active_device_layers(disaggregation, mask)
        stack_base = pd.Series(0.0, index=week_signal.index)

        # Aggregate signal in grey
        ax.plot(
            week_signal.index,
            week_signal.values,
            color="black",
            linewidth=1.4,
            label="Totale misurato",
            zorder=20,
        )

        # Stacked colored layers for active devices
        for idx, (dev_name, week_dev) in enumerate(active_layers):
            color = _device_color(dev_name)
            upper = stack_base.add(week_dev, fill_value=0)
            ax.fill_between(
                week_signal.index,
                stack_base.values,
                upper.values,
                alpha=0.65,
                color=color,
                label=dev_name,
                linewidth=0,
                zorder=5 + idx,
            )
            stack_base = upper

        if device_sum is None:
            device_sum = pd.Series(0.0, index=week_signal.index)

        residual = week_signal.fillna(0) - device_sum
        ax.fill_between(
            week_signal.index,
            device_sum.values,
            week_signal.fillna(0).values,
            where=(residual >= 0).values,
            color="lightgrey",
            alpha=0.35,
            label="Residuo positivo",
            zorder=15,
        )
        ax.plot(
            week_signal.index,
            residual.values,
            color="black",
            linewidth=0.9,
            linestyle="--",
            label="Residuo",
            zorder=18,
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

    day_periods = signal.index.to_period("D")
    days = day_periods.unique().sort_values()

    for day in days:
        day_label = day.start_time.strftime("%Y-%m-%d")
        mask = day_periods == day
        day_signal = signal[mask]

        if day_signal.dropna().empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 5))

        active_layers, device_sum = _build_active_device_layers(disaggregation, mask)
        stack_base = pd.Series(0.0, index=day_signal.index)

        ax.plot(
            day_signal.index,
            day_signal.values,
            color="black",
            linewidth=1.4,
            label="Totale misurato",
            zorder=20,
        )

        for idx, (dev_name, day_dev) in enumerate(active_layers):
            color = _device_color(dev_name)
            upper = stack_base.add(day_dev, fill_value=0)
            ax.fill_between(
                day_signal.index,
                stack_base.values,
                upper.values,
                alpha=0.65,
                color=color,
                label=dev_name,
                linewidth=0,
                zorder=5 + idx,
            )
            stack_base = upper

        if device_sum is None:
            device_sum = pd.Series(0.0, index=day_signal.index)

        residual = day_signal.fillna(0) - device_sum
        ax.fill_between(
            day_signal.index,
            device_sum.values,
            day_signal.fillna(0).values,
            where=(residual >= 0).values,
            color="lightgrey",
            alpha=0.35,
            label="Residuo positivo",
            zorder=15,
        )
        ax.plot(
            day_signal.index,
            residual.values,
            color="black",
            linewidth=0.9,
            linestyle="--",
            label="Residuo",
            zorder=18,
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
    bars = ax.barh(
        names,
        values,
        color=[_device_color(name) for name in names],
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
