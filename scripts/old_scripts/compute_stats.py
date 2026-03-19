#!/usr/bin/env python3
"""Compute per-file statistics for iot_devices JSON exports.

Produces a CSV with one row per input file containing counts, time range,
w_medio stats and granularity info. Also prints a short table to stdout.
"""
import json
import os
import glob
import argparse
from statistics import mean, median
from datetime import datetime

def extract_number(x):
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("$numberLong", "$numberInt", "$numberDouble"):
            if k in x:
                v = x[k]
                try:
                    if isinstance(v, str):
                        return int(v) if v.isdigit() else float(v)
                    return int(v)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None
    return x


def load_json_array(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
        data = data['result']
    if not isinstance(data, list):
        raise ValueError('Expected top-level JSON array or {"result": [...]}')
    return data


def epoch_from_record(rec):
    # prefer _id, then t_min_absolute, then t_max_absolute
    for key in ('_id', 't_min_absolute', 't_max_absolute'):
        if key in rec:
            e = extract_number(rec[key])
            if e is not None:
                return int(e)
    return None


def field_number(rec, field):
    return extract_number(rec.get(field))


def process_file(path, top_n=3):
    records = load_json_array(path)
    total = len(records)
    valid = sum(1 for r in records if r.get('epoch_valid') is not False)
    invalid = total - valid

    epochs = [epoch_from_record(r) for r in records]
    epochs = [e for e in epochs if e is not None]
    start = min(epochs) if epochs else None
    end = max(epochs) if epochs else None

    w_medio_vals = [field_number(r, 'w_medio') for r in records if field_number(r, 'w_medio') is not None and r.get('epoch_valid') is not False]
    w_medio_vals = [float(x) for x in w_medio_vals]

    gran_vals = [field_number(r, 'granularity') for r in records if field_number(r, 'granularity') is not None]
    gran_unique = sorted(set(gran_vals))

    w_massimo_list = []
    for r in records:
        wm = field_number(r, 'w_massimo')
        if wm is not None:
            epoch = epoch_from_record(r) or None
            w_massimo_list.append((float(wm), epoch))
    w_massimo_list.sort(reverse=True, key=lambda x: x[0])
    top_spikes = w_massimo_list[:top_n]

    stats = {
        'file': os.path.basename(path),
        'total_records': total,
        'valid_records': valid,
        'invalid_records': invalid,
        'start_utc': datetime.utcfromtimestamp(start).isoformat() + 'Z' if start else '',
        'end_utc': datetime.utcfromtimestamp(end).isoformat() + 'Z' if end else '',
        'w_medio_min': min(w_medio_vals) if w_medio_vals else '',
        'w_medio_mean': mean(w_medio_vals) if w_medio_vals else '',
        'w_medio_median': median(w_medio_vals) if w_medio_vals else '',
        'w_medio_max': max(w_medio_vals) if w_medio_vals else '',
        'unique_granularities': '|'.join(str(int(g)) for g in gran_unique) if gran_unique else '',
        'top_spikes_values': ','.join(str(int(x[0])) for x in top_spikes),
        'top_spikes_epochs_utc': ','.join(datetime.utcfromtimestamp(x[1]).isoformat() + 'Z' if x[1] else '' for x in top_spikes),
    }
    return stats


def main():
    p = argparse.ArgumentParser(description='Compute statistics for iot_devices JSON files')
    p.add_argument('--dir', default='.', help='Directory with iot_devices*.json files')
    p.add_argument('--out', default='summary_stats.csv', help='Output CSV file')
    p.add_argument('--top', type=int, default=3, help='Top N spikes to record')
    args = p.parse_args()

    pattern = os.path.join(args.dir, 'iot_devices*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No files found matching', pattern)
        return

    rows = []
    for f in files:
        try:
            stats = process_file(f, top_n=args.top)
            rows.append(stats)
            print(f"Processed {stats['file']}: total={stats['total_records']} valid={stats['valid_records']} w_medio_mean={stats['w_medio_mean']}")
        except Exception as e:
            print(f"Failed processing {f}: {e}")

    # write CSV
    import csv
    keys = ['file','total_records','valid_records','invalid_records','start_utc','end_utc','w_medio_min','w_medio_mean','w_medio_median','w_medio_max','unique_granularities','top_spikes_values','top_spikes_epochs_utc']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})

    print('\nWrote summary to', args.out)


if __name__ == '__main__':
    main()
