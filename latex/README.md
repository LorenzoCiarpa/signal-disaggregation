# Beamer Presentation

This folder contains a complete Beamer presentation that summarizes the NILM project, the implemented approaches, the Gurobi roadmap, and the benchmark results reconstructed from the CSV outputs already present in `analysis/`.

## Files

- `main.tex`: main Beamer source.
- `data/mae_by_imei.csv`: MAE per household and approach.
- `data/mae_summary.csv`: average MAE and coverage per approach.
- `data/mae_summary_full_coverage_ranked.csv`: ranked MAE summary for the 8 approaches benchmarked on all 6 eligible households.
- `data/mae_matrix_full_coverage_ranked_short.csv`: household-by-household MAE table using H1-H6 short labels.

## Compile

Run from the `latex/` directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If `pdflatex` is not installed, the sources are still ready to compile in any standard TeX distribution with Beamer and PGFPlots.