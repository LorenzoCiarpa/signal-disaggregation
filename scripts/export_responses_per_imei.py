#!/usr/bin/env python3
"""Export responses and related device info into one Excel workbook with a sheet per IMEI."""
import os
import re
import json
import pandas as pd


def normalize_imei(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def safe_sheet_name(name):
    # Excel sheet names max 31 chars and cannot contain [:]\/?*[]
    s = re.sub(r'[:\\/?*\[\]]', '_', name)
    return s[:31]


def main():
    resp_xlsx = 'NILM Questionario (Risposte).xlsx'
    if not os.path.exists(resp_xlsx):
        print('Responses file not found:', resp_xlsx)
        return
    df_resp = pd.read_excel(resp_xlsx, sheet_name=0)
    # normalize imei column name
    imei_col = None
    for c in df_resp.columns:
        if 'IMEI' in str(c) or 'Codice IMEI' in str(c):
            imei_col = c
            break
    if imei_col is None:
        print('IMEI column not found in responses')
        return

    df_resp['imei_norm'] = df_resp[imei_col].apply(normalize_imei)
    imeis = [i for i in df_resp['imei_norm'].unique() if i]

    # load device usage and stats if present
    usage_path = 'imei_devices_usage.csv'
    stats_path = 'summary_stats.csv'
    df_usage = pd.read_csv(usage_path) if os.path.exists(usage_path) else None
    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None

    out_xlsx = 'responses_per_imei.xlsx'
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for imei in imeis:
            sheet = safe_sheet_name('IMEI_' + imei)
            # write the response row(s)
            resp_rows = df_resp[df_resp['imei_norm'] == imei].copy()
            if resp_rows.empty:
                resp_rows = pd.DataFrame({'note': ['No response row found for this IMEI']})
                resp_rows.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
                next_row = 3
            else:
                resp_rows.drop(columns=['imei_norm'], inplace=True)
                resp_rows.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
                next_row = len(resp_rows) + 2

            # write device usage lines
            if df_usage is not None:
                dev_rows = df_usage[df_usage['imei'] == imei]
                if not dev_rows.empty:
                    dev_rows.to_excel(writer, sheet_name=sheet, index=False, startrow=next_row)
                    next_row += len(dev_rows) + 2

            # write device stats summary (if available)
            if df_stats is not None:
                matched = df_stats[df_stats['file'].str.contains(str(imei), na=False)]
                if not matched.empty:
                    # write transposed summary for convenience
                    summary = matched.iloc[0].to_frame().T
                    summary.to_excel(writer, sheet_name=sheet, index=False, startrow=next_row)
                    next_row += 2

            # include link/info about daily plots
            dev_basename = f'iot_devices.{imei}.json'
            daily_dir = os.path.join('daily_plots', dev_basename.rsplit('.json',1)[0])
            info_rows = []
            if os.path.isdir(daily_dir):
                files = sorted([f for f in os.listdir(daily_dir) if f.lower().endswith('.png')])
                info_rows.append({'daily_plots_dir': daily_dir, 'num_plots': len(files)})
                # include up to first 10 filenames
                for i, fn in enumerate(files[:10], start=1):
                    info_rows.append({f'plot_{i}': fn})
            else:
                info_rows.append({'daily_plots_dir': '', 'num_plots': 0})
            pd.DataFrame(info_rows).to_excel(writer, sheet_name=sheet, index=False, startrow=next_row)

    print('Wrote', out_xlsx)


if __name__ == '__main__':
    main()
