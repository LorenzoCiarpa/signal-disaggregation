from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nilm.approach_gurobi import solve_unconstrained_window
from scripts.nilm.devices import DeviceProfile, get_device_profiles
from scripts.nilm.preprocessing import load_imei


DEFAULT_OUTPUT_DIR = Path("analysis/real_week_gurobi_unconstrained")
DEFAULT_IMEI = "86684007269866"


def _get_present_devices(imei: str) -> list[DeviceProfile]:
    return [device for device in get_device_profiles(imei) if device.prior_weight >= 1.0]


def _select_week(signal: pd.Series, week_start: str | None) -> tuple[pd.Series, str]:
    week_periods = signal.index.to_period("W-SUN")
    weeks = week_periods.unique().sort_values()
    if len(weeks) == 0:
        raise ValueError("No weekly slices available for the selected IMEI.")

    if week_start is None:
        for week in weeks:
            week_signal = signal[week_periods == week]
            if len(week_signal) == 10080 and int(week_signal.notna().sum()) == 10080:
                label = f"{week.start_time.date()}_to_{week.end_time.date()}"
                return week_signal, label
        week = weeks[0]
        week_signal = signal[week_periods == week]
        label = f"{week.start_time.date()}_to_{week.end_time.date()}"
        return week_signal, label

    week_start_ts = pd.Timestamp(week_start)
    for week in weeks:
        if week.start_time.date() == week_start_ts.date():
            week_signal = signal[week_periods == week]
            label = f"{week.start_time.date()}_to_{week.end_time.date()}"
            return week_signal, label

    available = ", ".join(str(week.start_time.date()) for week in weeks)
    raise ValueError(f"Week start {week_start} not found. Available starts: {available}")


def _build_frame(signal: pd.Series, reconstructed: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame({"w_total": signal})
    for device_name, series in reconstructed.items():
        frame[device_name] = series.reindex(signal.index).fillna(0.0)
    frame["reconstructed_sum"] = frame.drop(columns=["w_total"]).sum(axis=1)
    frame["residual"] = frame["w_total"].fillna(0.0) - frame["reconstructed_sum"]
    frame.index.name = "timestamp"
    return frame


def _build_summary(
    imei: str,
    week_label: str,
    signal: pd.Series,
    reconstructed: dict[str, pd.Series],
    solver_info: dict[str, float | int | bool | None],
) -> dict[str, float | int | bool | None | str]:
    reconstructed_sum = sum(
        (series.reindex(signal.index).fillna(0.0) for series in reconstructed.values()),
        start=pd.Series(0.0, index=signal.index, dtype=float),
    )
    error = signal.fillna(0.0) - reconstructed_sum
    valid_mask = signal.notna()

    summary: dict[str, float | int | bool | None | str] = {
        "imei": imei,
        "week": week_label,
        "mae_w": float(error[valid_mask].abs().mean()) if valid_mask.any() else 0.0,
        "rmse_w": float(np.sqrt(np.mean(np.square(error[valid_mask].values)))) if valid_mask.any() else 0.0,
        "max_abs_error_w": float(error[valid_mask].abs().max()) if valid_mask.any() else 0.0,
        "n_samples": int(len(signal)),
        "n_valid_samples": int(valid_mask.sum()),
        "active_devices_reconstructed": int(
            sum(series.reindex(signal.index).fillna(0.0).gt(0).any() for series in reconstructed.values())
        ),
    }
    summary.update(solver_info)
    return summary


def _save_plot(
    signal: pd.Series,
    reconstructed: dict[str, pd.Series],
    output_path: Path,
    title: str,
) -> None:
    active_devices = [
        (name, series.reindex(signal.index).fillna(0.0))
        for name, series in reconstructed.items()
        if series.reindex(signal.index).fillna(0.0).sum() > 0
    ]
    reconstructed_sum = sum(
        (series for _, series in active_devices),
        start=pd.Series(0.0, index=signal.index, dtype=float),
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.5]},
    )

    ax_top.plot(signal.index, signal.values, color="black", linewidth=1.4, label="Segnale originale")
    ax_top.plot(
        reconstructed_sum.index,
        reconstructed_sum.values,
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
        label="Segnale ricostruito",
    )
    ax_top.set_ylabel("Potenza (W)")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right")
    ax_top.grid(alpha=0.2)

    if active_devices:
        labels = [name for name, _ in active_devices]
        stacked_values = [series.values for _, series in active_devices]
        ax_bottom.stackplot(signal.index, stacked_values, labels=labels, alpha=0.8)
    ax_bottom.plot(signal.index, signal.values, color="black", linewidth=1.0, alpha=0.5)
    ax_bottom.plot(
        reconstructed_sum.index,
        reconstructed_sum.values,
        color="tab:red",
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )
    ax_bottom.set_ylabel("Potenza (W)")
    ax_bottom.set_xlabel("Tempo")
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(loc="upper right", fontsize=8, ncol=2)

    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_real_week(
    imei: str,
    week_start: str | None,
    time_limit: float,
    output_dir: Path,
    verbose_solver: bool,
) -> Path:
    signal = load_imei(imei)
    week_signal, week_label = _select_week(signal, week_start)
    devices = _get_present_devices(imei)

    if not devices:
        raise ValueError(f"No present devices found for IMEI {imei}.")

    reconstructed, solver_info = solve_unconstrained_window(
        signal=week_signal,
        devices=devices,
        time_limit=time_limit,
        verbose=verbose_solver,
    )
    summary = _build_summary(imei, week_label, week_signal, reconstructed, solver_info)

    run_dir = output_dir / imei / week_label
    run_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_frame = _build_frame(week_signal, reconstructed)
    reconstruction_frame.to_csv(run_dir / "reconstruction.csv")

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    title = (
        f"IMEI {imei} | {week_label} | objective={summary.get('objective_value')} | "
        f"mae={summary['mae_w']:.2f}W"
    )
    _save_plot(week_signal, reconstructed, run_dir / "comparison.png", title)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unconstrained Gurobi NILM on one real weekly slice for a selected IMEI.",
    )
    parser.add_argument("--imei", default=DEFAULT_IMEI, help="Target IMEI.")
    parser.add_argument(
        "--week-start",
        default=None,
        help="Week start date in YYYY-MM-DD format. If omitted, pick the first full valid week.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--verbose-solver",
        action="store_true",
        help="Show Gurobi logs during optimization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run_real_week(
        imei=args.imei,
        week_start=args.week_start,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        verbose_solver=args.verbose_solver,
    )
    print(f"Saved outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())