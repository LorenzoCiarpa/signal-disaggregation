# Plot trace from IoT JSON

Script: `scripts/plot_tracciato.py`

Quick usage:

```bash
python scripts/plot_tracciato.py --file iot_devices.86853106211173.json --field w_medio --output trace.png --csv trace.csv
```

Options:
- `--file` (`-f`): input JSON file (required)
- `--field`: signal to plot, default `w_medio`
- `--time-key`: timestamp key (default `_id`)
- `--output` (`-o`): output PNG filename
- `--csv`: optional CSV export (columns epoch,value)
- `--show`: show plot interactively

Install dependencies:

```bash
pip install -r requirements.txt
```
