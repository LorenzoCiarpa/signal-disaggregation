#!/usr/bin/env python3
"""Extract devices present and their usage (frequency/duration) per IMEI from survey Excel.

Outputs:
- imei_devices_usage.csv : rows (imei, device, present_count, present_label, frequency, duration, model)
- device_usage_by_imei/<imei>.json : JSON summary per imei
"""
import os
import json
import pandas as pd
import re
from collections import defaultdict


DEVICE_KEYWORDS = [
    'Frigorifero', 'Congelatore', 'Lavatrice', 'Asciugatrice', 'Lavastoviglie',
    'Forno', 'Piano cottura', 'Microonde', 'Boiler', 'Climatizzatore',
    'Televisori', 'Computer', 'Console', 'Auto elettrica', 'Frigorifero principale',
    'Frigorifero secondario'
]


def normalize_imei(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def find_columns_for_device(cols, device):
    # return columns that mention the device name
    pat = re.compile(re.escape(device), flags=re.IGNORECASE)
    return [c for c in cols if pat.search(c)]


def pick_freq_and_duration_cols(cols, device):
    # attempt to find frequency and duration columns related to device
    freq_col = None
    dur_col = None
    for c in cols:
        if re.search(device, c, flags=re.IGNORECASE) and re.search(r'Frequenza|Frequenza di utilizzo|Frequenza media|Orario', c, flags=re.IGNORECASE):
            freq_col = c
        if re.search(device, c, flags=re.IGNORECASE) and re.search(r'Durata|Durata tipica|durata', c, flags=re.IGNORECASE):
            dur_col = c
    return freq_col, dur_col


def extract_models(cols):
    # columns that are model descriptions often contain 'modello' or 'Se disponibile'
    models = [c for c in cols if re.search(r'modello', c, flags=re.IGNORECASE) or re.search(r'Se disponibile aggiungere il modello', c, flags=re.IGNORECASE)]
    return models


def main():
    path = 'NILM Questionario (Risposte).xlsx'
    if not os.path.exists(path):
        print('Responses Excel not found:', path)
        return
    df = pd.read_excel(path, sheet_name=0)
    cols = list(df.columns)

    # Build device columns map
    device_cols_map = {}
    for dev in DEVICE_KEYWORDS:
        cols_found = find_columns_for_device(cols, dev)
        if cols_found:
            device_cols_map[dev] = cols_found

    models_cols = extract_models(cols)

    # prepare outputs
    rows = []
    os.makedirs('device_usage_by_imei', exist_ok=True)

    for idx, r in df.iterrows():
        imei = normalize_imei(r.get('Codice IMEI dispositivo') if 'Codice IMEI dispositivo' in df.columns else r.iloc[0])
        if not imei:
            continue
        per_imei = {'imei': imei, 'devices': {}}
        for dev, dev_cols in device_cols_map.items():
            # pick the most relevant column (first)
            col = dev_cols[0]
            val = r.get(col)
            present = False
            present_count = ''
            present_label = ''
            if pd.isna(val):
                present = False
            else:
                s = str(val).strip()
                if s.lower() in ('non presente','non presente ','no','0','nan'):
                    present = False
                else:
                    present = True
                    present_label = s
                    # if numeric, capture count
                    try:
                        present_count = int(float(s))
                    except Exception:
                        present_count = ''

            # find freq/duration columns
            freq_col, dur_col = pick_freq_and_duration_cols(cols, dev)
            freq_val = r.get(freq_col) if freq_col else ''
            dur_val = r.get(dur_col) if dur_col else ''

            # attempt to find model value among model cols mentioning device
            model_val = ''
            for mcol in models_cols:
                if re.search(dev.split()[0], mcol, flags=re.IGNORECASE):
                    mv = r.get(mcol)
                    if not pd.isna(mv):
                        model_val = str(mv)
                        break

            per_imei['devices'][dev] = {
                'present': bool(present),
                'present_label': present_label,
                'present_count': present_count,
                'frequency': '' if pd.isna(freq_val) else str(freq_val),
                'duration': '' if pd.isna(dur_val) else str(dur_val),
                'model': model_val
            }

            rows.append({
                'imei': imei,
                'device': dev,
                'present': per_imei['devices'][dev]['present'],
                'present_label': per_imei['devices'][dev]['present_label'],
                'present_count': per_imei['devices'][dev]['present_count'],
                'frequency': per_imei['devices'][dev]['frequency'],
                'duration': per_imei['devices'][dev]['duration'],
                'model': per_imei['devices'][dev]['model']
            })

        # save per-IMEI JSON
        out_json = os.path.join('device_usage_by_imei', f'{imei}.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(per_imei, f, ensure_ascii=False, indent=2)

    out_df = pd.DataFrame(rows)
    out_csv = 'imei_devices_usage.csv'
    out_df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)
    print('Wrote per-IMEI JSON files in device_usage_by_imei/')


if __name__ == '__main__':
    main()
