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
import sys
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nilm.preprocessing import (
    get_usable_imeis,
    load_imei,
    load_imei_plateau_median,
)
from scripts.nilm.devices import get_device_profiles, get_device_profiles_v2
from scripts.nilm import approach_event_based
from scripts.nilm import approach_hmm
from scripts.nilm import approach_fhmm
from scripts.nilm import approach_fhmm_1
from scripts.nilm import approach_fhmm_1_survey
from scripts.nilm import approach_template
from scripts.nilm import approach_event_prior
from scripts.nilm import approach_gurobi
from scripts.nilm import approach_gurobi_daywise
from scripts.nilm import approach_gurobi_15min
from scripts.nilm import approach_gurobi_soft
from scripts.nilm import approach_gurobi_multistate
from scripts.nilm import approach_gurobi_activation
from scripts.nilm import approach_gurobi_full
from scripts.nilm import gurobi_multiple
from scripts.nilm import approach_cvxpy
from scripts.nilm import approach_cvxpy_activation
from scripts.nilm import approach_cvxpy_full
from scripts.nilm import approach_hsmm_survey
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


def _resolve_devices_for_approach(device_bundle, approach_name: str):
    if isinstance(device_bundle, dict):
        return device_bundle.get(approach_name, device_bundle.get("default", []))
    return device_bundle


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


_DAY_START = 40  # skip this many days from the beginning of each signal
_DAY_COUNT = 5    # number of days to process

APPROACH_MAP = {
    "event": approach_event_based,
    "hmm": approach_hmm,
    "fhmm": approach_fhmm,
    # "fhmm_1": _PartialApproach(approach_fhmm_1, baseline_mode="peak"),
    # "fhmm_1_dc": _PartialApproach(approach_fhmm_1, baseline_mode="duty_avg"),
    "fhmm_1_survey": _PartialApproach(approach_fhmm_1_survey, baseline_mode="peak"),
    # "template": approach_template,
    # "event_prior": approach_event_prior,
    # "gurobi": approach_gurobi,
    "gurobi_daywise": approach_gurobi_daywise,
    "gurobi_15min": approach_gurobi_15min,
    "gurobi_15min_max": _PartialApproach(approach_gurobi_15min, resample_method="max"),
    "gurobi_15min_median": _PartialApproach(approach_gurobi_15min, resample_method="median"),
    # v4: soft time-window penalty (same granularity options)
    "gurobi_soft": approach_gurobi_soft,
    "gurobi_soft_max": _PartialApproach(approach_gurobi_soft, resample_method="max"),
    "gurobi_soft_median": _PartialApproach(approach_gurobi_soft, resample_method="median"),
    # v5: soft time-window penalty + 3 power levels per device (15 min)
    "gurobi_multistate": approach_gurobi_multistate,
    "gurobi_multistate_max": _PartialApproach(approach_gurobi_multistate, resample_method="max"),
    "gurobi_multistate_median": _PartialApproach(approach_gurobi_multistate, resample_method="median"),
    # v5: same but at 30-minute granularity
    "gurobi_multistate_30min": _PartialApproach(approach_gurobi_multistate, granularity_min=30),
    "gurobi_multistate_30min_max": _PartialApproach(approach_gurobi_multistate, granularity_min=30, resample_method="max"),
    "gurobi_multistate_30min_median": _PartialApproach(approach_gurobi_multistate, granularity_min=30, resample_method="median"),
    # v6: v5 + activation penalty (15 min)
    "gurobi_activation": approach_gurobi_activation,
    "gurobi_activation_max": _PartialApproach(approach_gurobi_activation, resample_method="max"),
    "gurobi_activation_median": _PartialApproach(approach_gurobi_activation, resample_method="median"),
    # v6: same but at 30-minute granularity
    "gurobi_activation_30min": _PartialApproach(approach_gurobi_activation, granularity_min=30),
    "gurobi_activation_30min_max": _PartialApproach(approach_gurobi_activation, granularity_min=30, resample_method="max"),
    "gurobi_activation_30min_median": _PartialApproach(approach_gurobi_activation, granularity_min=30, resample_method="median"),
    # v7: always-on devices as multistate variables (no baseline subtraction) — 15 min
    "gurobi_full": approach_gurobi_full,
    "gurobi_full_max": _PartialApproach(approach_gurobi_full, resample_method="max"),
    "gurobi_full_median": _PartialApproach(approach_gurobi_full, resample_method="median"),
    # v7: same at 30-minute granularity
    "gurobi_full_30min": _PartialApproach(approach_gurobi_full, granularity_min=30),
    "gurobi_full_30min_max": _PartialApproach(approach_gurobi_full, granularity_min=30, resample_method="max"),
    "gurobi_full_30min_median": _PartialApproach(approach_gurobi_full, granularity_min=30, resample_method="median"),
    # ---- Free-solver equivalents (CVXPY + SCIP, pip install cvxpy pyscipopt) ----
    # cvxpy: binary ON/OFF, L2 error — mirrors gurobi_soft (v4)
    "cvxpy": approach_cvxpy,
    "cvxpy_max": _PartialApproach(approach_cvxpy, resample_method="max"),
    "cvxpy_median": _PartialApproach(approach_cvxpy, resample_method="median"),
    "cvxpy_30min": _PartialApproach(approach_cvxpy, granularity_min=30),
    # cvxpy_activation: 3 power levels + activation penalty — mirrors gurobi_activation (v6)
    "cvxpy_activation": approach_cvxpy_activation,
    "cvxpy_activation_max": _PartialApproach(approach_cvxpy_activation, resample_method="max"),
    "cvxpy_activation_median": _PartialApproach(approach_cvxpy_activation, resample_method="median"),
    "cvxpy_activation_30min": _PartialApproach(approach_cvxpy_activation, granularity_min=30),
    # cvxpy_full: always-on as variables, no baseline — mirrors gurobi_full (v7)
    "cvxpy_full": approach_cvxpy_full,
    "cvxpy_full_max": _PartialApproach(approach_cvxpy_full, resample_method="max"),
    "cvxpy_full_median": _PartialApproach(approach_cvxpy_full, resample_method="median"),
    "cvxpy_full_30min": _PartialApproach(approach_cvxpy_full, granularity_min=30),
    # ---- HSMM with survey priors (pure numpy, no solver needed) ----
    "hsmm_survey": approach_hsmm_survey,
    "hsmm_survey_30min": _PartialApproach(approach_hsmm_survey, granularity_min=30),
    # "gurobi_multiple_3": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=3,
    #     constraint_version="unconstrained",
    # ),
    # "gurobi_multiple_v1_3": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=3,
    #     constraint_version="v1",
    # ),
    # "gurobi_multiple_v2_3": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=3,
    #     constraint_version="v2",
    # ),
    # "gurobi_multiple_5": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=5,
    #     constraint_version="unconstrained",
    # ),
    # "gurobi_multiple_v1_5": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=5,
    #     constraint_version="v1",
    # ),
    # "gurobi_multiple_v2_5": _PartialApproach(
    #     gurobi_multiple,
    #     n_power_levels=5,
    #     constraint_version="v2",
    # ),
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
        "--no-plots",
        action="store_true",
        help="Skip weekly plots (faster execution)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show solver output per day (HiGHS, HSMM, etc.)",
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
            profiles_v1 = get_device_profiles(imei)
            profiles_v2 = get_device_profiles_v2(imei)
            # Survey-aware approaches use v2 profiles (with time windows, duration bounds…)
            _survey_approaches = {
                "fhmm_1_survey",
                "hsmm_survey", "hsmm_survey_30min",
                "cvxpy", "cvxpy_max", "cvxpy_median", "cvxpy_30min",
                "cvxpy_activation", "cvxpy_activation_max",
                "cvxpy_activation_median", "cvxpy_activation_30min",
                "cvxpy_full", "cvxpy_full_max",
                "cvxpy_full_median", "cvxpy_full_30min",
            }
            devices_by_imei[imei] = {
                name: (profiles_v2 if name in _survey_approaches else profiles_v1)
                for name in APPROACH_MAP
            }
            devices_by_imei[imei]["default"] = profiles_v1
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
        _unique_days = signal.index.normalize().unique().sort_values()
        _start = min(_DAY_START, len(_unique_days))
        _end = min(_DAY_START + _DAY_COUNT, len(_unique_days))
        signal = signal[
            (signal.index >= _unique_days[_start]) & (signal.index < _unique_days[_end])
        ] if _start < _end else signal.iloc[0:0]
        results[imei] = {}

        for approach_key, approach_module in approaches:
            approach_name = approach_key
            devices = _resolve_devices_for_approach(devices_by_imei[imei], approach_name)
            
            print(f"Processing IMEI {imei} — approach {approach_name}...")
            print(f"signals: {signal[:50]}, devices: {len(devices)}")

            try:
                import time
                initial_time = time.time()
                
                disaggregation = _run_approach(
                    approach_module,
                    signal,
                    devices,
                    time_limit=60,
                    verbose=args.verbose,
                )
                results[imei][approach_name] = disaggregation
                elapsed_time = time.time() - initial_time
                print(f"Approach {approach_name} completed in {elapsed_time:.2f} seconds.")

                save_results(
                    signal=signal,
                    disaggregation=disaggregation,
                    imei=imei,
                    approach_name=approach_name,
                    output_dir=args.output_dir,
                    skip_weekly_plots=args.no_plots,
                    temporal_plot_granularity="both",
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
