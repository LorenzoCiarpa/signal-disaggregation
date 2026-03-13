#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path

def main():
    fixed=Path('responses_per_imei_clean_fixed.xlsx')
    survey=Path('NILM Questionario (Risposte).xlsx')
    outfile=Path('responses_per_imei_clean_fixed_v2.xlsx')
    if not fixed.exists():
        print('ERROR: missing', fixed)
        return
    if not survey.exists():
        print('ERROR: missing survey file', survey)
        return
    r = pd.read_excel(survey, sheet_name=0)
    imei_col = None
    for c in r.columns:
        if 'imei' in str(c).lower():
            imei_col = c
            break
    if imei_col is None:
        print('ERROR: could not find IMEI column in survey')
        return
    cols = list(r.columns)

    def find_col_for_device(device_name, kind='count'):
        dn = device_name.lower()
        for c in cols:
            s = str(c).lower()
            if '[' in s and dn in s:
                return c
        if kind=='frequency':
            for c in cols:
                s = str(c).lower()
                if dn in s and ('freq' in s or 'frequenz' in s):
                    return c
            for c in cols:
                s = str(c).lower()
                if dn in s and ('orario' in s or 'fasce' in s or 'avvio' in s):
                    return c
        else:
            for c in cols:
                s = str(c).lower()
                if dn in s and ('elettrodomestici' in s or '[' in s or 'presenza' in s or 'numerosita' in s or 'numerosità' in s):
                    return c
        for c in cols:
            if dn in str(c).lower():
                return c
        return None

    wb = pd.read_excel(fixed, sheet_name=None)
    updated = {}
    changes = []
    for sheet, df in wb.items():
        df = df.copy()
        sheet_imei = None
        m = re.search(r"(\d{10,15})", sheet)
        if m:
            sheet_imei = m.group(1)
        else:
            if 'imei' in df.columns and not df['imei'].isna().all():
                try:
                    first_idx = df['imei'].first_valid_index()
                    sheet_imei = str(df.loc[first_idx,'imei'])
                except Exception:
                    sheet_imei = None
        if sheet_imei is None:
            updated[sheet] = df
            continue
        def norm_imei(x):
            if pd.isna(x):
                return ''
            s = str(x).strip()
            if s.endswith('.0'):
                s = s[:-2]
            return s
        r['__imei_norm'] = r[imei_col].apply(norm_imei)
        mask = r['__imei_norm'] == sheet_imei
        if not mask.any():
            mask = r['__imei_norm'].str.contains(sheet_imei, na=False)
        if not mask.any():
            updated[sheet] = df
            continue
        srow = r.loc[mask].iloc[0]
        if 'count' not in df.columns:
            df['count'] = pd.NA
        for idx, row in df.iterrows():
            device = str(row.get('device','')).strip()
            if not device:
                continue
            cnt_col = find_col_for_device(device, kind='count')
            cnt_val = pd.NA
            if cnt_col is not None:
                raw = srow.get(cnt_col, pd.NA)
                if pd.isna(raw):
                    cnt_val = 0
                else:
                    sval = str(raw).strip()
                    if 'non presente' in sval.lower() or sval.lower()=='no' or sval.lower()=='non presente':
                        cnt_val = 0
                    else:
                        nums = re.findall(r"\d+", sval)
                        if nums:
                            cnt_val = max(int(x) for x in nums)
                        else:
                            cnt_val = sval
            else:
                cnt_val = pd.NA
            df.at[idx,'count'] = cnt_val
            freq_col = find_col_for_device(device, kind='frequency')
            if freq_col is None:
                freq_val = pd.NA
            else:
                fv = srow.get(freq_col, pd.NA)
                if pd.isna(fv):
                    freq_val = pd.NA
                else:
                    freq_val = str(fv).strip()
            if 'frequency' in df.columns:
                df.at[idx,'frequency'] = freq_val
            else:
                df['frequency'] = pd.NA
                df.at[idx,'frequency'] = freq_val
            changes.append((sheet, device, cnt_col, cnt_val, freq_col, freq_val))
        updated[sheet] = df
    with pd.ExcelWriter(outfile, engine='openpyxl') as w:
        for sheet, df in updated.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    print('Wrote', outfile)
    print('\nSample changes (first 20):')
    for c in changes[:20]:
        print(c)
