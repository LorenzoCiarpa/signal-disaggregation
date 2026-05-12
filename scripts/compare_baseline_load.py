from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nilm.baseline_load import BASELINE_METHODS, estimate_always_on_baseline
from scripts.nilm.devices import get_device_profiles
from scripts.nilm.preprocessing import get_usable_imeis, load_imei


DEFAULT_METHODS = [
    "constant_expected",
    "signal_quantile",
    "cyclic_median",
    "stumpy_motif",
]


def _compute_summary(signal: pd.Series, baseline: pd.Series) -> dict[str, float]:
    valid_mask = signal.notna() & baseline.notna()
    if not valid_mask.any():
        return {
            "baseline_mean_w": 0.0,
            "baseline_std_w": 0.0,
            "negative_minutes_pct": 0.0,
            "baseline_energy_pct": 0.0,
            "residual_mean_w": 0.0,
            "residual_mae_w": 0.0,
        }

    sig = signal[valid_mask].astype(float)
    base = baseline[valid_mask].astype(float)
    residual = sig - base
    total_energy = float(sig.sum())
    baseline_energy = float(base.sum())
    return {
        "baseline_mean_w": float(base.mean()),
        "baseline_std_w": float(base.std(ddof=0)),
        "negative_minutes_pct": float((base > sig).mean() * 100.0),
        "baseline_energy_pct": float((baseline_energy / total_energy) * 100.0) if total_energy > 0 else 0.0,
        "residual_mean_w": float(residual.mean()),
        "residual_mae_w": float(residual.abs().mean()),
    }


def _save_plot(
    signal: pd.Series,
    baselines: dict[str, pd.Series],
    output_path: Path,
    title: str = "Always-on baseline alternatives",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plot_signal = signal.copy()
    plot_baselines = baselines
    if len(signal) > 4 * 24 * 60:
        plot_signal = signal.resample("15min").mean()
        plot_baselines = {
            name: baseline.resample("15min").mean()
            for name, baseline in baselines.items()
        }

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    ax_top.plot(plot_signal.index, plot_signal.values, color="black", linewidth=1.4, label="Signal")
    for name, baseline in plot_baselines.items():
        ax_top.plot(baseline.index, baseline.values, linewidth=1.2, label=name)
    ax_top.set_ylabel("Power (W)")
    ax_top.set_title(title)
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=8, ncol=2)

    for name, baseline in plot_baselines.items():
        residual = (plot_signal - baseline).clip(lower=0.0)
        ax_bottom.plot(residual.index, residual.values, linewidth=1.0, label=f"Residual {name}")
    ax_bottom.set_ylabel("Residual (W)")
    ax_bottom.set_xlabel("Time")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(loc="upper right", fontsize=8, ncol=2)
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_daily_plots(
    signal: pd.Series,
    baselines: dict[str, pd.Series],
    output_dir: Path,
) -> None:
    if signal.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    signal_index = signal.index.tz_localize(None) if signal.index.tz is not None else signal.index
    day_periods = signal_index.to_period("D")

    for day in day_periods.unique().sort_values():
        mask = day_periods == day
        day_signal = signal[mask]
        if day_signal.dropna().empty:
            continue

        day_baselines = {
            name: baseline[mask]
            for name, baseline in baselines.items()
        }
        day_label = day.start_time.strftime("%Y-%m-%d")
        _save_plot(
            signal=day_signal,
            baselines=day_baselines,
            output_path=output_dir / f"{day_label}.png",
            title=f"Always-on baseline alternatives — {day_label}",
        )


def run_comparison(
    imei: str,
    methods: list[str],
    output_dir: Path,
    json_dir: str,
    plot_granularity: str,
) -> None:
    signal = load_imei(imei, json_dir=json_dir)
    devices = get_device_profiles(imei)
    imei_dir = output_dir / imei
    imei_dir.mkdir(parents=True, exist_ok=True)

    baselines: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        baseline, info = estimate_always_on_baseline(
            signal=signal,
            devices=devices,
            method=method,
            baseline_mode="duty_avg",
        )
        baselines[method] = baseline
        summary = _compute_summary(signal, baseline)
        row = {"imei": imei, "method": method}
        row.update(summary)
        row.update(info)
        summary_rows.append(row)

    baseline_frame = pd.DataFrame({"w_total": signal})
    for method, baseline in baselines.items():
        baseline_frame[method] = baseline
        baseline_frame[f"residual_{method}"] = (signal - baseline).clip(lower=0.0)
    baseline_frame.index.name = "timestamp"
    baseline_frame.index = baseline_frame.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    baseline_frame.to_csv(imei_dir / "baseline_methods.csv")

    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with (imei_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    if plot_granularity in {"overall", "both"}:
        _save_plot(signal, baselines, imei_dir / "baseline_comparison.png")
    if plot_granularity in {"daily", "both"}:
        _save_daily_plots(signal, baselines, imei_dir / "daily_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dynamic always-on baseline estimators for fridge/freezer removal.",
    )
    parser.add_argument(
        "--imei",
        default=None,
        help="Single IMEI to analyze. Default: all usable IMEIs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=list(BASELINE_METHODS),
        help="Baseline estimation methods to compare.",
    )
    parser.add_argument(
        "--json-dir",
        default="json",
        help="Directory containing IoT device JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/baseline_load"),
        help="Directory where comparison CSV and PNG files are written.",
    )
    parser.add_argument(
        "--plot-granularity",
        choices=["overall", "daily", "both"],
        default="overall",
        help="Save one overall comparison plot, one plot per day, or both.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    imeis = [args.imei] if args.imei else get_usable_imeis()
    for imei in imeis:
        run_comparison(
            imei=imei,
            methods=args.methods,
            output_dir=args.output_dir,
            json_dir=args.json_dir,
            plot_granularity=args.plot_granularity,
        )
        print(f"Saved baseline comparison for {imei} in {args.output_dir / imei}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())