#!/usr/bin/env python3
"""Generate one plot per recording day for each iot_devices JSON file.

Creates PNGs under `daily_plots/<basename>/<basename>_YYYY-MM-DD.png`.
"""
import os
import glob
import json
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def extract_number(x):
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("$numberLong", "$numberInt", "$numberDouble"):
            if k in x:
                v = x[k]
                try:
                    if isinstance(v, str):
                        return float(v) if ('.' in v) else int(v)
                    return int(v)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None
    try:
        return float(x)
    except Exception:
        return None


def load_json_array(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
        data = data['result']
    if not isinstance(data, list):
        raise ValueError('Expected top-level JSON array or {"result": [...]} export')
    return data


def build_dataframe(records, time_key='_id', field='w_medio', only_valid=True):
    rows = []
    for rec in records:
        if only_valid and rec.get('epoch_valid') is False:
            continue
        tval = rec.get(time_key)
        epoch = extract_number(tval)
        if epoch is None:
            # fallback
            for alt in ('t_min_absolute', 't_max_absolute'):
                if alt in rec:
                    epoch = extract_number(rec[alt])
                    if epoch is not None:
                        break
        val = extract_number(rec.get(field))
        if epoch is None or val is None:
            continue
        rows.append((int(epoch), float(val)))
    if not rows:
        return pd.DataFrame(columns=['epoch','value'])
    df = pd.DataFrame(rows, columns=['epoch','value'])
    df['datetime'] = pd.to_datetime(df['epoch'], unit='s', utc=True)
    df = df.set_index('datetime').sort_index()
    return df


def plot_daily(df, outdir, basename, field='w_medio', field_label='w_medio'):
    if df.empty:
        return 0
    # group by date (UTC)
    df['date'] = df.index.date
    n = 0
    for date, group in df.groupby('date'):
        date_str = date.isoformat()
        day_outdir = os.path.join(outdir, basename)
        os.makedirs(day_outdir, exist_ok=True)
        # include field in filename to avoid overwriting plots for different fields
        outpath = os.path.join(day_outdir, f"{basename}_{field}_{date_str}.png")
        plt.figure(figsize=(10,4))
        ax = plt.gca()
        group['value'].plot(ax=ax)
        plt.ylabel(field_label)
        # set x-axis to span full 24 hours (00:00 - 23:59:59 UTC)
        day_start = pd.Timestamp(date, tz='UTC')
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        ax.set_xlim(day_start, day_end)
        # show the date as the x-axis label and hours as tick labels
        plt.xlabel(date_str + ' (UTC)')
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.title(f"{basename} — {date_str} (UTC)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description='Plot daily traces for each JSON file')
    p.add_argument('--dir', default='.', help='Directory with iot_devices*.json files')
    p.add_argument('--outdir', default='daily_plots', help='Output directory for daily plots')
    p.add_argument('--field', default='w_medio', help='Field to plot (default: w_medio)')
    p.add_argument('--time-key', default='_id', help='Timestamp key (default: _id)')
    p.add_argument('--only-valid', dest='only_valid', action='store_true', default=True,
                   help='Only include records with epoch_valid != false (default)')
    args = p.parse_args()

    pattern = os.path.join(args.dir, 'iot_devices*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No files found matching', pattern)
        return

    total_plots = 0
    for f in files:
        basename = os.path.basename(f).rsplit('.json', 1)[0]
        try:
            records = load_json_array(f)
            df = build_dataframe(records, time_key=args.time_key, field=args.field, only_valid=args.only_valid)
            n = plot_daily(df, args.outdir, basename, field=args.field, field_label=args.field)
            print(f"{basename}: generated {n} daily plots")
            total_plots += n
        except Exception as e:
            print(f"Failed {f}: {e}")

    print(f"Done — total daily plots: {total_plots}")


if __name__ == '__main__':
    main()
