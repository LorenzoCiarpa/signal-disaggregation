"""
Main orchestration script for NILM signal disaggregation.

Usage:
    python3 scripts/run_disaggregation.py [--imei IMEI] [--approach APPROACH] [--no-plots]
                                          [--json-dir DIR] [--output-dir DIR]
"""

import argparse
import datetime
import functools
import os
import sys
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nilm.preprocessing import load_imei, get_usable_imeis
from scripts.nilm.devices import get_device_profiles
from scripts.nilm import approach_event_based
from scripts.nilm import approach_hmm
from scripts.nilm import approach_fhmm
from scripts.nilm import approach_fhmm_1
from scripts.nilm import approach_template
from scripts.nilm import approach_event_prior
from scripts.nilm.output import save_results
from scripts.nilm.benchmark import run_benchmark


class _PartialApproach:
    """Wraps a module's run() with fixed keyword args, preserving the module interface."""

    def __init__(self, module, **kwargs):
        self._run = functools.partial(module.run, **kwargs)

    def run(self, signal, devices):
        return self._run(signal, devices)


APPROACH_MAP = {
    "event": approach_event_based,
    "hmm": approach_hmm,
    "fhmm": approach_fhmm,
    "fhmm_1": _PartialApproach(approach_fhmm_1, baseline_mode="peak"),
    "fhmm_1_dc": _PartialApproach(approach_fhmm_1, baseline_mode="duty_avg"),
    "template": approach_template,
    "event_prior": approach_event_prior,
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
        choices=["event", "hmm", "fhmm", "fhmm_1", "fhmm_1_dc", "template", "event_prior", "all"],
        default="all",
        help="Disaggregation approach to run (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip daily plots (faster execution)",
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
    args = parser.parse_args()

    # Determine which IMEIs to process
    if args.imei:
        imeis = [args.imei]
    else:
        imeis = get_usable_imeis()

    # Determine which approaches to run
    if args.approach == "all":
        approaches = list(APPROACH_MAP.items())
    else:
        approaches = [(args.approach, APPROACH_MAP[args.approach])]

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
            signals[imei] = load_imei(imei, json_dir=args.json_dir)
            devices_by_imei[imei] = get_device_profiles(imei)
        except Exception as e:
            msg = f"ERROR loading IMEI {imei}: {e}"
            print(msg)
            errors.append(msg)

    # results[imei][approach_name] = disaggregation_dict
    results = {}

    for imei in imeis:
        if imei not in signals:
            continue
        signal = signals[imei]
        devices = devices_by_imei[imei]
        results[imei] = {}

        for approach_key, approach_module in approaches:
            approach_name = approach_key
            print(f"Processing IMEI {imei} — approach {approach_name}...")
            try:
                disaggregation = approach_module.run(signal, devices)
                results[imei][approach_name] = disaggregation

                save_results(
                    signal=signal,
                    disaggregation=disaggregation,
                    imei=imei,
                    approach_name=approach_name,
                    output_dir=args.output_dir,
                    skip_daily_plots=args.no_plots,
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
