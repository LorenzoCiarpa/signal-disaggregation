"""
Main orchestration script for NILM signal disaggregation.

Usage:
    python3 scripts/run_disaggregation.py [--imei IMEI] [--approach APPROACH] [--no-plots]
                                          [--json-dir DIR] [--output-dir DIR]
"""

import argparse
import datetime
import functools
import inspect
import os
import random
import sys
import traceback

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nilm.preprocessing import (
    get_usable_imeis,
    load_imei,
    load_imei_plateau_median,
)
from scripts.nilm.devices import get_device_profiles
from scripts.nilm import approach_highs_survey_prior
from scripts.nilm.output import save_results
from scripts.nilm.benchmark import run_benchmark


class _PartialApproach:
    """Wraps a module's run() with fixed keyword args, preserving the module interface."""

    def __init__(self, module, **kwargs):
        self._run = functools.partial(module.run, **kwargs)
        self._signature = inspect.signature(module.run)
        self._accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in self._signature.parameters.values()
        )

    def _filter_kwargs(self, kwargs):
        if self._accepts_var_kwargs:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if key in self._signature.parameters
        }

    def run(self, signal, devices, **kwargs):
        return self._run(signal, devices, **self._filter_kwargs(kwargs))


def _run_approach(approach_module, signal, devices, **kwargs):
    """Call an approach with only the keyword arguments it actually supports."""
    run_callable = approach_module.run
    signature = inspect.signature(run_callable)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        filtered_kwargs = kwargs
    else:
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }

    return run_callable(signal, devices, **filtered_kwargs)


# Analysis window: three scattered blocks of consecutive days, 15 days per IMEI.
# The blocks are drawn from the days available in EVERY IMEI, so all households are
# compared over identical periods and metric differences reflect the home, not the
# season.  Selection is seeded, so a rerun reproduces the same blocks.
_N_BLOCKS = 3
_BLOCK_DAYS = 5
_BLOCK_SEED = 20251212

# Christmas period, excluded because consumption is not representative.  Wider than
# "Christmas week" alone since New Year and Epiphany are just as atypical.
_HOLIDAY_START = "2025-12-22"
_HOLIDAY_END = "2026-01-06"  # inclusive

# Minimum distance between the starts of two blocks, so the blocks stay scattered
# across the record instead of clumping together.
_MIN_BLOCK_START_GAP_DAYS = 15


def _select_day_blocks(available_days, n_blocks, block_days, seed):
    """Pick n_blocks scattered runs of block_days consecutive available days.

    A day is eligible as a block start when the whole run that follows it is
    available and clear of the holiday period.  Starts are then drawn at random and
    kept only if far enough from the ones already chosen; the spacing requirement is
    relaxed if the record is too tight to satisfy it.

    Args:
        available_days: Sorted normalized timestamps present in every IMEI.
        n_blocks: How many blocks to select.
        block_days: Consecutive days per block.
        seed: Seed for reproducible selection.

    Returns:
        List of (start, end) timestamps, end exclusive, sorted chronologically.

    Raises:
        ValueError: If the record cannot supply the requested blocks at all.
    """
    day_set = set(available_days)
    holiday_start = pd.Timestamp(_HOLIDAY_START)
    holiday_end = pd.Timestamp(_HOLIDAY_END)

    candidates = []
    for day in available_days:
        run = [day + pd.Timedelta(days=k) for k in range(block_days)]
        if any(d not in day_set for d in run):
            continue
        if any(holiday_start <= d <= holiday_end for d in run):
            continue
        candidates.append(day)

    if len(candidates) < n_blocks:
        raise ValueError(
            f"only {len(candidates)} eligible starts for {n_blocks} blocks of "
            f"{block_days} days outside {_HOLIDAY_START}..{_HOLIDAY_END}"
        )

    rng = random.Random(seed)
    for min_gap in range(_MIN_BLOCK_START_GAP_DAYS, block_days - 1, -1):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        chosen = []
        for day in shuffled:
            if all(abs((day - c).days) >= min_gap for c in chosen):
                chosen.append(day)
            if len(chosen) == n_blocks:
                break
        if len(chosen) == n_blocks:
            chosen.sort()
            return [(d, d + pd.Timedelta(days=block_days)) for d in chosen]

    raise ValueError(
        f"could not place {n_blocks} non-overlapping blocks of {block_days} days"
    )

APPROACH_MAP = {
    # ---- HiGHS L1-MILP (free solver, no license) ----
    # highs_survey_prior: adds seasonality (active_months) and window-specificity weighting
    "highs_survey_prior": approach_highs_survey_prior,
    "highs_survey_prior_30min": _PartialApproach(approach_highs_survey_prior, granularity_min=30),
}



def main():
    parser = argparse.ArgumentParser(
        description="NILM Signal Disaggregation — full pipeline runner"
    )
    parser.add_argument(
        "--imei",
        default=None,
        help="Single IMEI to process (default: all 6 usable IMEIs)",
    )
    parser.add_argument(
        "--approach",
        nargs="+",
        default=["all"],
        metavar="APPROACH",
        help="One or more approaches to run, or 'all' (default: all)",
    )
    parser.add_argument(
        "--plots",
        choices=["none", "daily", "weekly", "both"],
        default="daily",
        help=(
            "Which temporal plots to save (default: daily). "
            "'none' is fastest; weekly plots are the slow ones"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show HiGHS solver output per day",
    )
    parser.add_argument(
        "--json-dir",
        default="json",
        help="Directory containing IoT device JSON files (default: json)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Output directory for results (default: analysis)",
    )
    parser.add_argument(
        "--signal-source",
        choices=["raw", "plateau_max_mediana"],
        default="plateau_max_mediana",
        help=(
            "Aggregate signal to disaggregate: raw or preprocessed with "
            "plateau_max followed by mediana (default: plateau_max_mediana)"
        ),
    )
    args = parser.parse_args() 

    # Determine which IMEIs to process
    if args.imei:
        imeis = [args.imei]
    else:
        imeis = get_usable_imeis()

    # Determine which approaches to run
    if args.approach == ["all"]:
        approaches = list(APPROACH_MAP.items())
    else:
        unknown = [a for a in args.approach if a not in APPROACH_MAP and a != "all"]
        if unknown:
            parser.error(f"Unknown approach(es): {', '.join(unknown)}. Valid: {', '.join(APPROACH_MAP)}")
        keys = list(APPROACH_MAP.keys()) if "all" in args.approach else args.approach
        approaches = [(k, APPROACH_MAP[k]) for k in keys]

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "run_log.txt")
    errors = []
    start_time = datetime.datetime.now()

    with open(log_path, "a") as log_f:
        log_f.write(f"\n=== Run started at {start_time.isoformat()} ===\n")
        log_f.write(f"IMEIs: {imeis}\n")
        log_f.write(f"Approaches: {[k for k, _ in approaches]}\n")

    # Load all signals and device profiles up front
    signals = {}
    devices_by_imei = {}

    for imei in imeis:
        try:
            print(f"Loading signal for IMEI {imei}...")
            if args.signal_source == "plateau_max_mediana":
                signals[imei] = load_imei_plateau_median(imei, json_dir=args.json_dir)
            else:
                signals[imei] = load_imei(imei, json_dir=args.json_dir)
            devices_by_imei[imei] = get_device_profiles(imei)
        except Exception as e:
            msg = f"ERROR loading IMEI {imei}: {e}"
            print(msg)
            errors.append(msg)

    if not signals:
        print("No signal could be loaded; nothing to do.")
        return

    # Days present in every loaded IMEI, so all households share the same blocks.
    common_days = None
    for signal in signals.values():
        days = set(signal.index.normalize().tz_localize(None).unique())
        common_days = days if common_days is None else (common_days & days)
    common_days = sorted(common_days)

    blocks = _select_day_blocks(common_days, _N_BLOCKS, _BLOCK_DAYS, _BLOCK_SEED)
    block_desc = ", ".join(
        f"{start.date()}..{(end - pd.Timedelta(days=1)).date()}" for start, end in blocks
    )
    print(f"Analysis blocks ({_N_BLOCKS} x {_BLOCK_DAYS} days): {block_desc}")
    with open(log_path, "a") as log_f:
        log_f.write(f"Blocks: {block_desc}\n")

    # Restrict every signal to the selected blocks.  Blocks stay separate so each is
    # solved on its own continuous span: concatenating first would make resampling
    # invent thousands of empty slots across the gaps between blocks.
    block_signals_by_imei = {}
    for imei in list(signals):
        signal = signals[imei]
        tz = signal.index.tz
        chunks = []
        for start, end in blocks:
            start_ts = pd.Timestamp(start, tz=tz)
            end_ts = pd.Timestamp(end, tz=tz)
            chunk = signal[(signal.index >= start_ts) & (signal.index < end_ts)]
            if not chunk.dropna().empty:
                chunks.append(chunk)
        if not chunks:
            msg = f"ERROR IMEI {imei}: no data in the selected blocks"
            print(msg)
            errors.append(msg)
            del signals[imei]
            continue
        block_signals_by_imei[imei] = chunks
        # Benchmark and plots must see exactly the days that were disaggregated.
        signals[imei] = pd.concat(chunks).sort_index()

    # results[imei][approach_name] = disaggregation_dict
    results = {}

    for imei in imeis:
        if imei not in signals:
            continue
        signal = signals[imei]
        block_signals = block_signals_by_imei[imei]
        results[imei] = {}

        for approach_key, approach_module in approaches:
            approach_name = approach_key
            devices = devices_by_imei[imei]
            
            print(f"Processing IMEI {imei} — approach {approach_name}...")
            print(f"  {len(signal)} samples over {len(block_signals)} blocks, {len(devices)} devices")

            try:
                import time
                initial_time = time.time()
                
                # Each block is disaggregated independently, then stitched back
                # together; a device's weekly activation counter therefore restarts
                # per block, which is what we want since blocks are weeks apart.
                per_block = [
                    _run_approach(
                        approach_module,
                        block_signal,
                        devices,
                        time_limit=60,
                        verbose=args.verbose,
                    )
                    for block_signal in block_signals
                ]
                device_names = {name for block in per_block for name in block}
                disaggregation = {
                    name: pd.concat(
                        [block[name] for block in per_block if name in block]
                    ).sort_index()
                    for name in device_names
                }
                results[imei][approach_name] = disaggregation
                elapsed_time = time.time() - initial_time
                print(f"Approach {approach_name} completed in {elapsed_time:.2f} seconds.")

                save_results(
                    signal=signal,
                    disaggregation=disaggregation,
                    imei=imei,
                    approach_name=approach_name,
                    output_dir=args.output_dir,
                    skip_plots=args.plots == "none",
                    temporal_plot_granularity=(
                        args.plots if args.plots != "none" else "daily"
                    ),
                )
            except Exception as e:
                msg = f"ERROR IMEI {imei}, approach {approach_name}: {e}\n{traceback.format_exc()}"
                print(msg)
                errors.append(f"ERROR IMEI {imei}, approach {approach_name}: {e}")

    # Run benchmark if we have any results
    if results:
        print("\nRunning benchmark...")
        try:
            run_benchmark(
                results=results,
                signals=signals,
                devices_by_imei=devices_by_imei,
                output_dir=args.output_dir,
            )
        except Exception as e:
            msg = f"ERROR in benchmark: {e}\n{traceback.format_exc()}"
            print(msg)
            errors.append(f"ERROR in benchmark: {e}")

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    with open(log_path, "a") as log_f:
        log_f.write(f"=== Run finished at {end_time.isoformat()} (duration: {duration}) ===\n")
        if errors:
            log_f.write(f"Errors ({len(errors)}):\n")
            for err in errors:
                log_f.write(f"  - {err}\n")
        else:
            log_f.write("No errors.\n")

    print(f"\nDone. Duration: {duration}. Log: {log_path}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
