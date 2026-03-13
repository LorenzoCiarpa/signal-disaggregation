#!/usr/bin/env python3
"""Count missing values per file for selected fields and write CSV."""
import glob
import json
import os
import csv
from typing import List


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


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
        data = data['result']
    if not isinstance(data, list):
        raise ValueError('Expected JSON array or {"result": [...]}')
    return data


def epoch_present(rec):
    for k in ('_id', 't_min_absolute', 't_max_absolute'):
        if k in rec and extract_number(rec[k]) is not None:
            return True
    return False


def field_present(rec, field):
    if field not in rec:
        return False
    v = rec.get(field)
    if v is None:
        return False
    # if it's a dict like {$numberLong:...} consider present
    if isinstance(v, dict):
        return any(k in v for k in ("$numberLong", "$numberInt", "$numberDouble")) or bool(v)
    return True


def process_file(path, fields: List[str]):
    records = load_json(path)
    total = len(records)
    counts = {f: 0 for f in fields}
    counts['missing_epoch'] = 0
    counts['missing_any'] = 0

    for r in records:
        missing_any = False
        if not epoch_present(r):
            counts['missing_epoch'] += 1
            missing_any = True
        for f in fields:
            if not field_present(r, f):
                counts[f] += 1
                missing_any = True
        if missing_any:
            counts['missing_any'] += 1

    row = {'file': os.path.basename(path), 'total_records': total}
    row.update(counts)
    return row


def main():
    pattern = 'iot_devices*.json'
    files = sorted(glob.glob(pattern))
    if not files:
        print('No files found matching', pattern)
        return

    fields = ['w_medio', 'w_istantaneo', 'w_massimo', 'granularity']
    rows = []
    for f in files:
        try:
            r = process_file(f, fields)
            rows.append(r)
            print(f"Processed {r['file']}: total={r['total_records']} missing_any={r['missing_any']}")
        except Exception as e:
            print(f"Failed {f}: {e}")

    out = 'missing_counts.csv'
    keys = ['file','total_records','missing_any','missing_epoch'] + fields
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, 0) for k in keys})

    print('\nWrote', out)


if __name__ == '__main__':
    main()
