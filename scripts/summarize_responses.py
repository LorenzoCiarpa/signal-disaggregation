#!/usr/bin/env python3
"""Read NILM Questionario (Risposte).xlsx and produce brief summaries.

Outputs:
- responses_preview.csv : first 200 rows exported to CSV
- responses_summary.txt : textual summary with total responses and per-column value counts
"""
import pandas as pd
import os
import sys


def main(path='NILM Questionario (Risposte).xlsx'):
    if not os.path.exists(path):
        print('Responses file not found:', path, file=sys.stderr)
        sys.exit(2)
    xls = pd.read_excel(path, sheet_name=0)
    # save preview
    preview_csv = 'responses_preview.csv'
    xls.head(200).to_csv(preview_csv, index=False)

    summary_lines = []
    summary_lines.append(f'Total responses (rows): {len(xls)}')

    # For each column, show non-null count and top 5 values
    for col in xls.columns:
        ser = xls[col]
        nonnull = ser.dropna()
        summary_lines.append('\nColumn: ' + str(col))
        summary_lines.append(f'  Non-null: {nonnull.shape[0]} / {len(ser)}')
        try:
            top = nonnull.value_counts(dropna=True).head(5)
            for v, c in top.items():
                summary_lines.append(f'    {repr(v)}: {c}')
        except Exception:
            summary_lines.append('    (Could not compute value_counts)')

    out_txt = 'responses_summary.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print('Wrote', preview_csv)
    print('Wrote', out_txt)


if __name__ == '__main__':
    main()
