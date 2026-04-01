"""
US-009: Benchmark proxy senza labels.

Computes proxy quality metrics (reconstruction error, energy balance, temporal consistency)
for each (IMEI, approach) combination and saves a comparison CSV and heatmap.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_metrics(
    signal: pd.Series,
    disaggregation: dict,
    devices: list,
) -> dict:
    """Compute proxy quality metrics for a single (imei, approach) disaggregation result.

    Args:
        signal: Aggregate power signal (pd.Series with DatetimeIndex).
        disaggregation: dict[str, pd.Series] — per-device estimated power series.
        devices: list[DeviceProfile] — device profiles for temporal consistency check.

    Returns:
        dict with keys: mae_recon, rmse_recon, energy_error_pct, residuo_medio_w,
                        n_devices_found, temporal_consistency.
    """
    valid_mask = signal.notna()
    sig_valid = signal[valid_mask]

    # Sum of all device estimates aligned to signal index
    device_names = list(disaggregation.keys())
    if device_names:
        disagg_df = pd.DataFrame(disaggregation, index=signal.index)
        total_disagg = disagg_df.sum(axis=1)
    else:
        total_disagg = pd.Series(0.0, index=signal.index)

    residual = signal - total_disagg

    # Reconstruction error metrics (on valid timesteps only)
    recon_error = (signal - total_disagg)[valid_mask]
    mae_recon = float(recon_error.abs().mean())
    rmse_recon = float(np.sqrt((recon_error ** 2).mean()))

    # Energy balance error
    total_energy = float(sig_valid.sum())
    disagg_energy = float(total_disagg[valid_mask].sum())
    if total_energy > 0:
        energy_error_pct = float(abs(total_energy - disagg_energy) / total_energy * 100.0)
    else:
        energy_error_pct = 0.0

    # Residuo medio
    residuo_medio_w = float(residual[valid_mask].mean()) if valid_mask.any() else 0.0

    # n_devices_found: devices whose estimated energy > 1% of total
    n_devices_found = 0
    if total_energy > 0:
        threshold = 0.01 * total_energy
        for name, series in disaggregation.items():
            dev_energy = float(series[valid_mask].sum())
            if dev_energy > threshold:
                n_devices_found += 1

    # Temporal consistency: % of ON blocks with duration in [dur_min_min/2, dur_typical_min*3]
    device_map = {d.name: d for d in devices}
    consistency_counts = []
    for name, series in disaggregation.items():
        if name not in device_map:
            continue
        dev = device_map[name]
        lo = dev.dur_min_min / 2.0
        hi = dev.dur_typical_min * 3.0

        # Find ON blocks (consecutive 1s when series > 0)
        is_on = (series.fillna(0) > 0).astype(int)
        # Detect block boundaries
        diff = is_on.diff().fillna(is_on.iloc[0] if len(is_on) > 0 else 0)
        starts = is_on.index[diff == 1].tolist()
        ends = is_on.index[diff == -1].tolist()

        # Handle edge cases
        if len(is_on) > 0 and is_on.iloc[0] == 1:
            starts = [is_on.index[0]] + starts
        if len(is_on) > 0 and is_on.iloc[-1] == 1:
            ends = ends + [is_on.index[-1]]

        total_blocks = len(starts)
        if total_blocks == 0:
            continue

        # Pair starts and ends
        valid_blocks = 0
        for i, start in enumerate(starts):
            if i < len(ends):
                end = ends[i]
                # Duration in minutes (index is DatetimeIndex)
                try:
                    dur_min = (end - start).total_seconds() / 60.0
                except TypeError:
                    # Fallback: count samples (1-min resampled)
                    dur_min = float(i + 1)
                if lo <= dur_min <= hi:
                    valid_blocks += 1

        consistency_counts.append(valid_blocks / total_blocks)

    temporal_consistency = float(np.mean(consistency_counts)) if consistency_counts else 0.0

    return {
        "mae_recon": mae_recon,
        "rmse_recon": rmse_recon,
        "energy_error_pct": energy_error_pct,
        "residuo_medio_w": residuo_medio_w,
        "n_devices_found": n_devices_found,
        "temporal_consistency": temporal_consistency,
    }


def run_benchmark(
    results: dict,
    signals: dict,
    devices_by_imei: dict,
    output_dir: str = "analysis",
) -> pd.DataFrame:
    """Run benchmark for all (IMEI, approach) combinations and save CSV + heatmap.

    Args:
        results: dict[imei][approach_name] = disaggregation_dict[str, pd.Series]
        signals: dict[imei] = aggregate pd.Series
        devices_by_imei: dict[imei] = list[DeviceProfile]
        output_dir: Root output directory (default: 'analysis').

    Returns:
        pd.DataFrame with columns [imei, approach, mae_recon, rmse_recon,
                                    energy_error_pct, residuo_medio_w,
                                    n_devices_found, temporal_consistency].
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for imei, approach_results in results.items():
        signal = signals.get(imei)
        devices = devices_by_imei.get(imei, [])
        if signal is None:
            continue
        for approach_name, disaggregation in approach_results.items():
            metrics = compute_metrics(signal, disaggregation, devices)
            row = {"imei": imei, "approach": approach_name}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "imei", "approach", "mae_recon", "rmse_recon",
        "energy_error_pct", "residuo_medio_w", "n_devices_found", "temporal_consistency",
    ])

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    # Build summary: mean metrics per approach across all IMEIs
    metric_cols = ["mae_recon", "rmse_recon", "energy_error_pct",
                   "residuo_medio_w", "n_devices_found", "temporal_consistency"]
    if not df.empty:
        summary = df.groupby("approach")[metric_cols].mean()

        # Save heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.5), max(4, len(summary) * 0.8)))
        data = summary.values
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd_r")

        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_cols, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(summary.index, fontsize=9)

        # Annotate cells
        for i in range(len(summary)):
            for j in range(len(metric_cols)):
                val = data[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax)
        ax.set_title("Benchmark Summary — mean metrics per approach")
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "benchmark_summary.png")
        fig.savefig(summary_path, dpi=100)
        plt.close(fig)

    # Print ranking by mae_recon ascending
    if not df.empty:
        ranking = df.groupby("approach")["mae_recon"].mean().sort_values()
        print("\n=== Benchmark Ranking (mae_recon ascending) ===")
        for rank, (approach, mae) in enumerate(ranking.items(), start=1):
            print(f"  {rank}. {approach}: mae_recon={mae:.2f} W")
        print()

    return df
