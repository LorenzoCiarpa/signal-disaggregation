#!/usr/bin/env python3
"""Link survey responses to device stats by IMEI and produce merged CSV.

Output: responses_with_device_stats.csv
"""
import os
import sys
import pandas as pd


def normalize_imei(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    # remove decimal if present like 86684007269889.0
    if s.endswith('.0'):
        s = s[:-2]
    return s


def find_stats_for_imei(imei, stats_df):
    # stats_df has 'file' column with filenames like iot_devices.<imei>.json
    if not imei:
        return None
    for _, row in stats_df.iterrows():
        fname = str(row.get('file',''))
        if imei in fname:
            return row.to_dict()
    return None


def main():
    resp_path = 'NILM Questionario (Risposte).xlsx'
    if not os.path.exists(resp_path):
        print('Responses file not found:', resp_path, file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(resp_path, sheet_name=0)
    # load stats
    stats_path = 'summary_stats.csv'
    zeros_path = 'zeros_counts.csv'
    stats_df = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame()
    zeros_df = pd.read_csv(zeros_path) if os.path.exists(zeros_path) else pd.DataFrame()

    out_rows = []
    for idx, row in df.iterrows():
        imei_raw = row.get('Codice IMEI dispositivo') if 'Codice IMEI dispositivo' in df.columns else row.iloc[0]
        imei = normalize_imei(imei_raw)
        matched = find_stats_for_imei(imei, stats_df) if not stats_df.empty else None
        zeros = find_stats_for_imei(imei, zeros_df) if not zeros_df.empty else None

        out = row.to_dict()
        out['imei_norm'] = imei
        if matched:
            out['device_file'] = matched.get('file')
            out['total_records'] = matched.get('total_records')
            out['w_medio_mean'] = matched.get('w_medio_mean')
            out['w_medio_min'] = matched.get('w_medio_min')
            out['w_medio_max'] = matched.get('w_medio_max')
            out['start_utc'] = matched.get('start_utc')
            out['end_utc'] = matched.get('end_utc')
        else:
            out['device_file'] = ''
        if zeros:
            out['w_medio_zero_count'] = zeros.get('w_medio_zero_count')
            out['w_medio_zero_frac'] = zeros.get('w_medio_zero_frac')
        else:
            out['w_medio_zero_count'] = ''
            out['w_medio_zero_frac'] = ''

        # add path to daily plots directory if exists
        dev_basename = ''
        if out.get('device_file'):
            dev_basename = out['device_file'].rsplit('.json',1)[0]
        daily_dir = os.path.join('daily_plots', dev_basename) if dev_basename else ''
        out['daily_plots_dir'] = daily_dir if os.path.isdir(daily_dir) else ''

        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    out_csv = 'responses_with_device_stats.csv'
    out_df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)


if __name__ == '__main__':
    main()
