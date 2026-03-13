#!/usr/bin/env python3
"""Generate responses_per_imei_clean.xlsx: one sheet per IMEI,
one row per appliance with columns: device, frequency, duration, preferred_moment.
"""
import os
import re
import pandas as pd


DEVICE_MOMENT_COL_MAP = {
    'Lavatrice': 'Lavatrice  - Fasce orarie più frequenti:',
    'Lavastoviglie': 'Lavastoviglie - Orario di avvio più frequente:',
    # fallback: many devices don't have a dedicated moment column
}


def normalize_imei(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def infer_moment_from_label(label):
    if not label or pd.isna(label):
        return ''
    s = str(label).strip()
    # common tokens
    tokens = ['Mattina', 'Pomeriggio', 'Sera', 'Notte', 'Cena', 'Pranzo']
    for t in tokens:
        if t.lower() in s.lower():
            return t
    return s


def main():
    usage_csv = 'imei_devices_usage.csv'
    resp_xlsx = 'NILM Questionario (Risposte).xlsx'
    out_xlsx = 'responses_per_imei_clean.xlsx'

    if not os.path.exists(usage_csv) or not os.path.exists(resp_xlsx):
        print('Required files missing (imei_devices_usage.csv and responses xlsx)')
        return

    df_usage = pd.read_csv(usage_csv, dtype=str)
    df_resp = pd.read_excel(resp_xlsx, sheet_name=0)
    # find imei column name in responses
    imei_col = None
    for c in df_resp.columns:
        if 'IMEI' in str(c) or 'Codice IMEI' in str(c):
            imei_col = c
            break
    if imei_col is None:
        print('IMEI column not found in responses')
        return

    df_resp['imei_norm'] = df_resp[imei_col].apply(normalize_imei)

    imeis = sorted(df_usage['imei'].unique())
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for imei in imeis:
            sheet = 'IMEI_' + imei
            rows = []
            sub = df_usage[df_usage['imei'] == imei]
            # find corresponding response row if any
            resp_row = df_resp[df_resp['imei_norm'] == imei]
            resp_row = resp_row.iloc[0] if not resp_row.empty else None

            for _, r in sub.iterrows():
                device = r.get('device','')
                frequency = r.get('frequency','') if pd.notna(r.get('frequency','')) else ''
                duration = r.get('duration','') if pd.notna(r.get('duration','')) else ''
                present_label = r.get('present_label','') if pd.notna(r.get('present_label','')) else ''

                # preferred moment: look into response specific columns if available
                moment = ''
                if device in DEVICE_MOMENT_COL_MAP and resp_row is not None:
                    col = DEVICE_MOMENT_COL_MAP[device]
                    if col in resp_row.index and pd.notna(resp_row[col]):
                        moment = resp_row[col]
                if not moment:
                    # try infer from present_label
                    moment = infer_moment_from_label(present_label)

                rows.append({
                    'imei': imei,
                    'device': device,
                    'frequency': frequency,
                    'duration': duration,
                    'preferred_moment': moment
                })

            if not rows:
                pd.DataFrame([{'note':'no devices found for imei'}]).to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

    print('Wrote', out_xlsx)


if __name__ == '__main__':
    main()
