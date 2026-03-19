#!/usr/bin/env python3
"""Plot comparative bar chart of w_medio_mean from summary CSV."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    p = argparse.ArgumentParser(description='Plot comparative w_medio_mean')
    p.add_argument('--input', '-i', default='summary_stats.csv', help='Input summary CSV')
    p.add_argument('--output', '-o', default='comparison_w_medio_mean.png', help='Output PNG')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f'Input file not found: {args.input}')

    df = pd.read_csv(args.input)
    # ensure numeric
    df['w_medio_mean'] = pd.to_numeric(df['w_medio_mean'], errors='coerce')
    df = df.sort_values('w_medio_mean', ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df['file'], df['w_medio_mean'], color='C0')
    plt.ylabel('w_medio_mean (W)')
    plt.title('Comparative w_medio_mean per file')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f'Saved comparative plot to {args.output}')


if __name__ == '__main__':
    main()
