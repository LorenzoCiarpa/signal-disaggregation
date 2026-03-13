#!/usr/bin/env python3
"""Count zero values per file for selected numeric fields and write CSV."""
import glob
import json
import os
import csv


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
                    return float(v)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None
    try:
        return float(x)
    except Exception:
        return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
        data = data['result']
    if not isinstance(data, list):
        raise ValueError('Expected JSON array or {"result": [...]}')
    return data


def is_valid_record(rec):
    return rec.get('epoch_valid') is not False


def process_file(path, fields):
    records = load_json(path)
    total = len(records)
    valid_records = [r for r in records if is_valid_record(r)]
    valid = len(valid_records)

    zeros = {f: 0 for f in fields}
    for r in valid_records:
        for f in fields:
            v = extract_number(r.get(f))
            if v is not None and v == 0:
                zeros[f] += 1

    row = {'file': os.path.basename(path), 'total_records': total, 'valid_records': valid}
    for f in fields:
        row[f + '_zero_count'] = zeros[f]
        row[f + '_zero_frac'] = (zeros[f] / valid) if valid else 0
    return row


def main():
    pattern = 'iot_devices*.json'
    files = sorted(glob.glob(pattern))
    if not files:
        print('No files found matching', pattern)
        return

    fields = ['w_medio', 'w_istantaneo', 'w_massimo']
    rows = []
    for f in files:
        try:
            r = process_file(f, fields)
            rows.append(r)
            print(f"Processed {r['file']}: total={r['total_records']} valid={r['valid_records']} zeros_w_medio={r['w_medio_zero_count']}")
        except Exception as e:
            print(f"Failed {f}: {e}")

    out = 'zeros_counts.csv'
    keys = ['file','total_records','valid_records'] + [x + suffix for x in fields for suffix in ('_zero_count','_zero_frac')]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, 0) for k in keys})

    print('\nWrote', out)


if __name__ == '__main__':
    main()
