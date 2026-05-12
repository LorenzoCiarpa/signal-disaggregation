from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from scripts.nilm.devices import DEVICE_KNOWLEDGE_BASE, DeviceProfile


MINUTES_PER_DAY = 24 * 60
DEFAULT_OUTPUT_DIR = Path("analysis/synthetic_gurobi_unconstrained")
BASE_DEVICE_NAMES = [
    "Frigorifero principale",
    "Lavatrice",
    "Lavastoviglie",
    "Microonde",
    "Televisori",
    "Computer",
    "Forno",
]
PREFERRED_WINDOWS = {
    "Lavatrice": [(8 * 60, 11 * 60), (17 * 60, 20 * 60)],
    "Lavastoviglie": [(12 * 60, 14 * 60), (20 * 60, 22 * 60)],
    "Microonde": [(7 * 60, 8 * 60), (12 * 60, 14 * 60), (19 * 60, 21 * 60)],
    "Televisori": [(19 * 60, 23 * 60)],
    "Computer": [(9 * 60, 12 * 60), (14 * 60, 18 * 60), (20 * 60, 22 * 60)],
    "Forno": [(12 * 60, 13 * 60), (19 * 60, 20 * 60)],
}


@dataclass
class SyntheticInstance:
    name: str
    signal: pd.Series
    ground_truth: dict[str, pd.Series]
    devices: list[DeviceProfile]
    seed: int


def _build_device(name: str) -> DeviceProfile:
    spec = DEVICE_KNOWLEDGE_BASE[name]
    return DeviceProfile(
        name=name,
        p_min_w=float(spec["p_min_w"]),
        p_typical_w=float(spec["p_typical_w"]),
        p_max_w=float(spec["p_max_w"]),
        dur_min_min=float(spec["dur_min_min"]),
        dur_typical_min=float(spec["dur_typical_min"]),
        duty_cycle=spec.get("duty_cycle"),
        frequency_per_week=float(spec["frequency_per_week"]),
        prior_weight=1.0,
        always_on=bool(spec.get("always_on", False)),
    )


def _choose_devices(rng: np.random.Generator) -> list[DeviceProfile]:
    event_pool = BASE_DEVICE_NAMES[1:]
    chosen_events = rng.choice(event_pool, size=4, replace=False).tolist()
    chosen_names = [BASE_DEVICE_NAMES[0], *chosen_events]
    return [_build_device(name) for name in chosen_names]


def _generate_fridge_profile(
    device: DeviceProfile,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.Series:
    on_minutes = max(1, int(round(device.dur_typical_min)))
    duty_cycle = float(device.duty_cycle or 0.3)
    period = max(on_minutes + 1, int(round(on_minutes / duty_cycle)))
    shift = int(rng.integers(0, period))
    values = np.zeros(len(index), dtype=float)
    for start in range(-shift, len(index), period):
        if start >= len(index):
            break
        lo = max(0, start)
        hi = min(len(index), start + on_minutes)
        if lo < hi:
            values[lo:hi] = device.p_typical_w
    return pd.Series(values, index=index, dtype=float)


def _sample_duration(device: DeviceProfile, rng: np.random.Generator) -> int:
    duration = int(round(device.dur_typical_min * rng.uniform(0.7, 1.3)))
    return max(int(round(device.dur_min_min)), duration)


def _sample_start(device_name: str, duration: int, rng: np.random.Generator) -> int:
    windows = PREFERRED_WINDOWS.get(device_name, [(0, MINUTES_PER_DAY)])
    window_idx = int(rng.integers(0, len(windows)))
    lo, hi = windows[window_idx]
    latest_start = max(lo, hi - duration)
    if latest_start <= lo:
        return max(0, min(lo, MINUTES_PER_DAY - duration))
    return int(rng.integers(lo, latest_start + 1))


def _generate_event_profile(
    device: DeviceProfile,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.Series:
    values = np.zeros(len(index), dtype=float)
    occupied = np.zeros(len(index), dtype=bool)
    expected_daily = max(0.5, device.frequency_per_week / 7.0)
    n_activations = max(1, int(rng.poisson(expected_daily)))

    for _ in range(n_activations):
        duration = _sample_duration(device, rng)
        for _attempt in range(30):
            start = _sample_start(device.name, duration, rng)
            end = min(len(index), start + duration)
            if end <= start:
                continue
            if occupied[start:end].any():
                continue
            occupied[start:end] = True
            values[start:end] = device.p_typical_w
            break

    return pd.Series(values, index=index, dtype=float)


def generate_instance(instance_idx: int, base_seed: int) -> SyntheticInstance:
    seed = base_seed + instance_idx
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=instance_idx)
    index = pd.date_range(start=start, periods=MINUTES_PER_DAY, freq="1min", tz="UTC")
    devices = _choose_devices(rng)

    ground_truth: dict[str, pd.Series] = {}
    for device in devices:
        if device.always_on:
            ground_truth[device.name] = _generate_fridge_profile(device, index, rng)
        else:
            ground_truth[device.name] = _generate_event_profile(device, index, rng)

    signal = sum(ground_truth.values(), start=pd.Series(0.0, index=index, dtype=float))
    return SyntheticInstance(
        name=f"instance_{instance_idx + 1:03d}",
        signal=signal,
        ground_truth=ground_truth,
        devices=devices,
        seed=seed,
    )


def _build_frame(signal: pd.Series, disaggregation: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame({"w_total": signal})
    for device_name, series in disaggregation.items():
        frame[device_name] = series.reindex(signal.index).fillna(0.0)
    frame["reconstructed_sum"] = frame.drop(columns=["w_total"]).sum(axis=1)
    frame.index.name = "timestamp"
    return frame


def _compute_summary(
    signal: pd.Series,
    ground_truth: dict[str, pd.Series],
    reconstructed: dict[str, pd.Series],
    solver_info: dict[str, float | int | bool | None],
) -> dict[str, float | int | bool | None]:
    ground_truth_sum = sum(
        (series.reindex(signal.index).fillna(0.0) for series in ground_truth.values()),
        start=pd.Series(0.0, index=signal.index, dtype=float),
    )
    reconstructed_sum = sum(
        (series.reindex(signal.index).fillna(0.0) for series in reconstructed.values()),
        start=pd.Series(0.0, index=signal.index, dtype=float),
    )
    error = signal - reconstructed_sum

    summary: dict[str, float | int | bool | None] = {
        "mae_w": float(error.abs().mean()),
        "rmse_w": float(np.sqrt(np.mean(np.square(error.values)))),
        "max_abs_error_w": float(error.abs().max()),
        "aggregate_exact_match": bool(np.allclose(signal.values, reconstructed_sum.values)),
        "ground_truth_consistent": bool(np.allclose(signal.values, ground_truth_sum.values)),
        "active_devices_truth": int(sum(series.gt(0).any() for series in ground_truth.values())),
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
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.4]},
    )

    ax_top.plot(signal.index, signal.values, color="black", linewidth=1.8, label="Segnale originale")
    ax_top.plot(
        reconstructed_sum.index,
        reconstructed_sum.values,
        color="tab:red",
        linewidth=1.4,
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
    ax_bottom.plot(signal.index, signal.values, color="black", linewidth=1.2, alpha=0.6)
    ax_bottom.plot(
        reconstructed_sum.index,
        reconstructed_sum.values,
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
    )
    ax_bottom.set_ylabel("Potenza (W)")
    ax_bottom.set_xlabel("Ora")
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(loc="upper right", fontsize=8, ncol=2)

    ax_bottom.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_instance_outputs(
    instance: SyntheticInstance,
    reconstructed: dict[str, pd.Series],
    summary: dict[str, float | int | bool | None],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_frame = _build_frame(instance.signal, instance.ground_truth)
    reconstruction_frame = _build_frame(instance.signal, reconstructed)
    ground_truth_frame.to_csv(output_dir / "ground_truth.csv")
    reconstruction_frame.to_csv(output_dir / "reconstruction.csv")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    title = (
        f"{instance.name} | seed={instance.seed} | "
        f"objective={summary.get('objective_value')} | exact={summary['aggregate_exact_match']}"
    )
    _save_plot(instance.signal, reconstructed, output_dir / "comparison.png", title)


def run_experiment(
    n_instances: int,
    base_seed: int,
    time_limit: float,
    output_dir: Path,
    verbose_solver: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | None | str]] = []

    for instance_idx in range(n_instances):
        instance = generate_instance(instance_idx, base_seed)
        reconstructed, solver_info = solve_unconstrained_window(
            signal=instance.signal,
            devices=instance.devices,
            time_limit=time_limit,
            verbose=verbose_solver,
        )
        summary = _compute_summary(instance.signal, instance.ground_truth, reconstructed, solver_info)
        summary_row = {"instance": instance.name, "seed": instance.seed}
        summary_row.update(summary)
        rows.append(summary_row)
        _save_instance_outputs(instance, reconstructed, summary, output_dir / instance.name)

        print(
            f"{instance.name}: exact={summary['aggregate_exact_match']} "
            f"mae={summary['mae_w']:.4f}W rmse={summary['rmse_w']:.4f}W "
            f"status={summary.get('status')} runtime={summary.get('runtime_sec')}"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 24h aggregate signals and test unconstrained Gurobi NILM.",
    )
    parser.add_argument("--instances", type=int, default=5, help="Number of 24h synthetic instances.")
    parser.add_argument("--seed", type=int, default=20260511, help="Base random seed.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Gurobi time limit in seconds for each daily instance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV, JSON and PNG files are written.",
    )
    parser.add_argument(
        "--verbose-solver",
        action="store_true",
        help="Show Gurobi logs for each solve.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = run_experiment(
        n_instances=args.instances,
        base_seed=args.seed,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        verbose_solver=args.verbose_solver,
    )
    exact_matches = int(summary_df["aggregate_exact_match"].sum()) if not summary_df.empty else 0
    print(f"Saved outputs to {args.output_dir} | exact aggregate matches: {exact_matches}/{len(summary_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())