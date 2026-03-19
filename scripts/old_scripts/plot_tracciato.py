#!/usr/bin/env python3
"""Read an IoT JSON export and plot a chosen power signal (e.g. `w_medio`).

Usage example:
  python scripts/plot_tracciato.py --file iot_devices.86853106211173.json \
    --field w_medio --output plot.png --csv out.csv
"""
import argparse
import json
from datetime import datetime
import os
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependency: please install requirements.txt", file=sys.stderr)
    raise


def extract_number(x):
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("$numberLong", "$numberInt", "$numberDouble"):
            if k in x:
                v = x[k]
                try:
                    # some exports use strings
                    if isinstance(v, str):
                        if '.' in v:
                            return float(v)
                        return int(v)
                    return int(v)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None
    return x


def parse_record(rec, time_key, field):
    # time value may be in `_id` as {$numberLong: "..."}
    tval = rec.get(time_key)
    epoch = extract_number(tval)
    if epoch is None:
        # fallback to common timestamp fields
        for alt in ("t_min_absolute", "t_max_absolute"):
            tv = rec.get(alt)
            if isinstance(tv, dict) and "$numberLong" in tv:
                epoch = extract_number(tv)
                break
    val = rec.get(field)
    v = extract_number(val)
    return epoch, v


def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "result" in data and isinstance(data["result"], list):
        data = data["result"]
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array or {'result': [...]} export")
    return data


def build_dataframe(records, time_key, field, only_valid=True):
    rows = []
    for rec in records:
        if only_valid and rec.get("epoch_valid") is False:
            continue
        epoch, value = parse_record(rec, time_key, field)
        if epoch is None or value is None:
            continue
        rows.append((int(epoch), float(value)))
    if not rows:
        return pd.DataFrame(columns=["epoch", "value"]) 
    df = pd.DataFrame(rows, columns=["epoch", "value"]) 
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()
    return df


def plot_series(df, field, output, show, title=None):
    if df.empty:
        print("No data to plot.")
        return
    plt.figure(figsize=(10, 4))
    df["value"].plot()
    plt.ylabel(field)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot power trace from IoT JSON export")
    p.add_argument("--file", "-f", required=True, help="Path to JSON file")
    p.add_argument("--field", default="w_medio", help="Field to plot (default: w_medio)")
    p.add_argument("--time-key", default="_id", help="Timestamp key (default: _id)")
    p.add_argument("--output", "-o", default="plot.png", help="Output PNG file")
    p.add_argument("--csv", help="Optional CSV output file")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    p.add_argument("--no-filter-invalid", dest="only_valid", action="store_false",
                   help="Do not filter out records with epoch_valid==false")
    args = p.parse_args()

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    records = load_json_array(args.file)
    df = build_dataframe(records, args.time_key, args.field, only_valid=args.only_valid)
    if args.csv:
        df_to_save = df[["epoch", "value"]].copy()
        df_to_save["epoch"] = df_to_save.index.astype('int64') // 10**9
        df_to_save.to_csv(args.csv, index=False)
        print(f"Saved CSV to {args.csv}")

    plot_series(df, args.field, args.output, args.show, title=os.path.basename(args.file))


if __name__ == "__main__":
    main()
