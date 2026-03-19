#!/usr/bin/env python3
"""Plot bar chart of zero fractions (or counts) from zeros_counts.csv."""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description='Plot zero fractions per file')
    p.add_argument('--input', '-i', default='zeros_counts.csv', help='Input CSV with zero counts')
    p.add_argument('--field', '-f', default='w_medio', help='Field to plot zeros for (default: w_medio)')
    p.add_argument('--output', '-o', default='zeros_fraction_comparison.png', help='Output PNG')
    p.add_argument('--mode', choices=('fraction','count'), default='fraction', help='Plot fraction (default) or absolute count')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f'Input file not found: {args.input}')

    df = pd.read_csv(args.input)
    col_count = args.field + '_zero_count'
    col_frac = args.field + '_zero_frac'
    if args.mode == 'fraction':
        if col_frac not in df.columns:
            raise SystemExit(f'Column not found: {col_frac}')
        vals = pd.to_numeric(df[col_frac], errors='coerce').fillna(0)
        ylabel = 'Fraction of zeros (0-1)'
    else:
        if col_count not in df.columns:
            raise SystemExit(f'Column not found: {col_count}')
        vals = pd.to_numeric(df[col_count], errors='coerce').fillna(0)
        ylabel = 'Zero count'

    labels = df['file'].astype(str)

    plt.figure(figsize=(10,5))
    bars = plt.bar(labels, vals, color='C1')
    plt.ylabel(ylabel)
    plt.title(f'Zeros for {args.field} ({args.mode})')
    plt.xticks(rotation=45, ha='right')
    for bar, v in zip(bars, vals):
        if args.mode == 'fraction':
            txt = f"{v:.2%}"
        else:
            txt = f"{int(v)}"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), txt, ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
