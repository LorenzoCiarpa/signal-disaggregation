#!/usr/bin/env python3
"""
Analisi NILM - Verifica qualità dati e generazione grafici
===========================================================
Questo script per ogni file JSON (IMEI):
1. Carica i dati e calcola statistiche
2. Filtra le letture con qualsiasi watt (min/max/medio) > 10000 W
3. Genera grafici giornalieri (w_max e w_medio) escludendo i punti con spike
4. Per gli IMEI con spike > 10000:
   - Crea file testo con gli istanti anomali
   - Genera grafici dove si vedono i picchi
5. Genera un file di verifica (verification_report.txt) che riscrive
   e verifica le informazioni del file NILM_dati.rtf originale

Output nella cartella: analysis/
  analysis/{imei}/
    daily_plots/         <- grafici giorno per giorno (w_max + w_medio)
    spike_plots/         <- grafici giorni con picchi > 10000 W
    spikes_{imei}.txt   <- lista istanti anomali
  analysis/verification_report.txt
"""
import json
import os
import glob
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# ─── Configurazione ──────────────────────────────────────────────────────────
THRESHOLD_W = 10000   # soglia in watt
JSON_DIR = os.path.join(os.path.dirname(__file__), '..', 'json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis')


# ─── Utilità ─────────────────────────────────────────────────────────────────

def extract_number(x):
    """Estrae valore numerico da campi MongoDB $numberLong o plain."""
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("$numberLong", "$numberInt", "$numberDouble"):
            if k in x:
                v = x[k]
                try:
                    return int(v) if isinstance(v, str) and v.lstrip('-').isdigit() else float(v)
                except Exception:
                    return None
    if isinstance(x, (int, float)):
        return x
    return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'result' in data:
        data = data['result']
    return data


def parse_record(rec):
    """Restituisce (epoch, w_min, w_max, w_medio) o None se non valido."""
    epoch = extract_number(rec.get('_id'))
    if epoch is None:
        return None
    # epoch_valid=False → scarto
    if rec.get('epoch_valid') is False:
        return None
    w_min = extract_number(rec.get('w_minimo'))
    w_max = extract_number(rec.get('w_massimo'))
    w_med = extract_number(rec.get('w_medio'))
    return epoch, w_min, w_max, w_med


def epoch_to_dt(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def imei_from_path(path):
    """Estrae l'IMEI dal nome file iot_devices.IMEI.json"""
    base = os.path.basename(path)          # iot_devices.86684007269866.json
    parts = base.split('.')
    if len(parts) >= 2:
        return parts[-2]                   # penultimo token = IMEI
    return base


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─── Caricamento e analisi dati ───────────────────────────────────────────────

def load_imei_data(path):
    """
    Ritorna:
      records_clean : lista (dt, w_min, w_max, w_med) senza spike
      records_spike : lista (dt, w_min, w_max, w_med) con almeno un valore > THRESHOLD_W
      all_epochs    : lista di tutti gli epoch validi (per rilevare buchi)
    """
    raw = load_json(path)
    records_clean = []
    records_spike = []
    all_epochs = []

    for rec in raw:
        parsed = parse_record(rec)
        if parsed is None:
            continue
        epoch, w_min, w_max, w_med = parsed
        all_epochs.append(epoch)
        dt = epoch_to_dt(epoch)

        # valori numerici (usa 0 se None per il confronto)
        vals = [v for v in (w_min, w_max, w_med) if v is not None]
        if vals and max(vals) > THRESHOLD_W:
            records_spike.append((dt, w_min, w_max, w_med))
        else:
            records_clean.append((dt, w_min, w_max, w_med))

    return records_clean, records_spike, all_epochs


# ─── Statistiche di sintesi ───────────────────────────────────────────────────

def compute_summary(records_clean, records_spike, all_epochs, imei):
    """Ritorna un dizionario con statistiche aggregate."""
    total_valid = len(records_clean) + len(records_spike)
    all_epochs_sorted = sorted(all_epochs)

    start_dt = epoch_to_dt(all_epochs_sorted[0]) if all_epochs_sorted else None
    end_dt   = epoch_to_dt(all_epochs_sorted[-1]) if all_epochs_sorted else None

    w_med_clean = [r[3] for r in records_clean if r[3] is not None]
    w_max_clean = [r[2] for r in records_clean if r[2] is not None]
    w_max_spike = [r[2] for r in records_spike if r[2] is not None]
    w_min_spike = [r[1] for r in records_spike if r[1] is not None]
    w_med_spike = [r[3] for r in records_spike if r[3] is not None]

    # rileva buchi > 2 ore
    gaps = []
    for i in range(1, len(all_epochs_sorted)):
        diff = all_epochs_sorted[i] - all_epochs_sorted[i-1]
        if diff > 7200:  # 2 ore
            gaps.append((epoch_to_dt(all_epochs_sorted[i-1]),
                         epoch_to_dt(all_epochs_sorted[i]),
                         diff / 3600))

    # conta zeri nel w_medio
    zero_count = sum(1 for r in records_clean if r[3] is not None and r[3] == 0)
    zero_count += sum(1 for r in records_spike if r[3] is not None and r[3] == 0)

    return {
        'imei': imei,
        'total_valid': total_valid,
        'spike_count': len(records_spike),
        'clean_count': len(records_clean),
        'start_dt': start_dt,
        'end_dt': end_dt,
        'w_med_mean': sum(w_med_clean)/len(w_med_clean) if w_med_clean else 0,
        'w_med_max_clean': max(w_med_clean) if w_med_clean else 0,
        'w_max_max_clean': max(w_max_clean) if w_max_clean else 0,
        'w_max_max_spike': max(w_max_spike) if w_max_spike else 0,
        'zero_count': zero_count,
        'gaps': gaps,
    }


# ─── Grafici giornalieri (puliti) ────────────────────────────────────────────

def plot_daily(records_clean, imei, out_dir):
    """Genera un PNG per ogni giorno con w_max e w_medio (dati puliti)."""
    ensure_dir(out_dir)

    # raggruppa per giorno (UTC)
    by_day = defaultdict(list)
    for dt, w_min, w_max, w_med in records_clean:
        day_key = dt.date()
        by_day[day_key].append((dt, w_max, w_med))

    for day, entries in sorted(by_day.items()):
        entries.sort(key=lambda x: x[0])
        times  = [e[0] for e in entries]
        w_maxs = [e[1] if e[1] is not None else float('nan') for e in entries]
        w_meds = [e[2] if e[2] is not None else float('nan') for e in entries]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(times, w_maxs, color='tomato',     linewidth=0.8, label='w_max')
        ax.plot(times, w_meds, color='steelblue',  linewidth=0.8, label='w_medio')
        ax.set_title(f'IMEI {imei} — {day}', fontsize=11)
        ax.set_xlabel('Ora (UTC)')
        ax.set_ylabel('Potenza (W)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        fname = os.path.join(out_dir, f'{imei}_{day}.png')
        fig.savefig(fname, dpi=100)
        plt.close(fig)


# ─── File spike + grafici spike ──────────────────────────────────────────────

def write_spike_file(records_spike, imei, out_path):
    """Scrive un file testo con tutti gli istanti anomali (> THRESHOLD_W)."""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Istanti con almeno un valore watt > {THRESHOLD_W} W\n")
        f.write(f"IMEI: {imei}\n")
        f.write(f"Totale record anomali: {len(records_spike)}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Data/Ora (UTC)':<25} {'w_min':>8} {'w_max':>8} {'w_medio':>8}\n")
        f.write("-" * 70 + "\n")
        for dt, w_min, w_max, w_med in sorted(records_spike, key=lambda x: x[0]):
            f.write(
                f"{dt.strftime('%Y-%m-%d %H:%M:%S'):<25}"
                f" {str(int(w_min)) if w_min is not None else 'N/A':>8}"
                f" {str(int(w_max)) if w_max is not None else 'N/A':>8}"
                f" {str(int(w_med)) if w_med is not None else 'N/A':>8}\n"
            )


def plot_spike_days(records_clean, records_spike, imei, out_dir):
    """
    Per ogni giorno in cui esiste almeno uno spike, genera un grafico
    che mostra anche i record puliti + i momenti di picco evidenziati.
    """
    ensure_dir(out_dir)

    # giornate con spike
    spike_days = set(r[0].date() for r in records_spike)

    # raggruppa puliti per giorno
    clean_by_day = defaultdict(list)
    for dt, w_min, w_max, w_med in records_clean:
        clean_by_day[dt.date()].append((dt, w_max, w_med))

    # raggruppa spike per giorno
    spike_by_day = defaultdict(list)
    for dt, w_min, w_max, w_med in records_spike:
        spike_by_day[dt.date()].append((dt, w_max, w_med))

    for day in sorted(spike_days):
        clean_entries = sorted(clean_by_day.get(day, []), key=lambda x: x[0])
        spike_entries = sorted(spike_by_day.get(day, []), key=lambda x: x[0])

        fig, ax = plt.subplots(figsize=(14, 5))

        if clean_entries:
            times  = [e[0] for e in clean_entries]
            w_maxs = [e[1] if e[1] is not None else float('nan') for e in clean_entries]
            w_meds = [e[2] if e[2] is not None else float('nan') for e in clean_entries]
            ax.plot(times, w_maxs, color='tomato',    linewidth=0.8, label='w_max (normale)', zorder=2)
            ax.plot(times, w_meds, color='steelblue', linewidth=0.8, label='w_medio (normale)', zorder=2)

        if spike_entries:
            stimes  = [e[0] for e in spike_entries]
            sw_maxs = [e[1] if e[1] is not None else float('nan') for e in spike_entries]
            sw_meds = [e[2] if e[2] is not None else float('nan') for e in spike_entries]
            ax.scatter(stimes, sw_maxs, color='red',    s=40, zorder=5, label=f'w_max SPIKE (>{THRESHOLD_W}W)', marker='^')
            ax.scatter(stimes, sw_meds, color='orange', s=30, zorder=5, label=f'w_medio SPIKE (>{THRESHOLD_W}W)', marker='v')
            # linea orizzontale soglia
            ax.axhline(THRESHOLD_W, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label=f'Soglia {THRESHOLD_W} W')

        ax.set_title(f'IMEI {imei} — {day}  [SPIKE > {THRESHOLD_W} W]', fontsize=11)
        ax.set_xlabel('Ora (UTC)')
        ax.set_ylabel('Potenza (W)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        fname = os.path.join(out_dir, f'{imei}_spike_{day}.png')
        fig.savefig(fname, dpi=100)
        plt.close(fig)


# ─── Verifica NILM_dati.rtf ──────────────────────────────────────────────────

# Contenuto originale del file NILM_dati.rtf (trascritto)
NILM_ORIGINAL = {
    "86853106211179": (
        "Zero fino al 22/11 con qualche picco prima assurdo (60000), "
        "poi dalle 16:00 del 22/11 tracciato normale"
    ),
    "86853106211173": (
        "Picchi isolati fino a 14000 fino al 18/11 poi dati mancanti, "
        "poi 27/11 alcuni dati, poi di nuovo quasi sempre zero "
        "(con qualche picco bassissimo) fino al 31/12"
    ),
    "86853106211166": (
        "Dati quasi tutti nulli fino al 18/11 poi dati mancanti "
        "riprende il 23/01 (arriva fino a 7000)"
    ),
    "86853106211162": (
        "Picco a 5000, i dati ci sono da subito (all'inizio solo frigo)"
    ),
    "86853106210447": "Pochissimi dati, inutilizzabile",
    "86684007269889": "Dati tutti presenti, picco a 6000, buco dal 2/01 al 2/02",
    "86684007269888": (
        "Inizia con dati mancanti, poi picchi assurdi a 60000, "
        "dato ragionevoli iniziano il 19/01"
    ),
    "86684007269887": (
        "Iniziano il 28/11 prima mancanti e picchi strani fino al 15/12 "
        "potenze bassissime, poi più normali (molti picchi stretti), "
        "valore zero dal 21/01 (qualche picco a 800 all'inizio, poi dal 23/01 "
        "proprio zero) fino al 31/01, pochissimi giorni utilizzabili"
    ),
    "86684007269869": (
        "Dati solo dal 28/01, prima zeri e picchi assurdi a 60000"
    ),
    "86684007269866": "Tutti dati ragionevoli",
}


def verify_claim(summary, imei):
    """
    Controlla le affermazioni del RTF rispetto ai dati reali.
    Restituisce lista di stringhe con esito verifica.
    """
    lines = []
    orig = NILM_ORIGINAL.get(imei, "(nessuna nota originale)")
    lines.append(f"  Nota originale : {orig}")

    start = summary['start_dt']
    end   = summary['end_dt']
    spike_max = summary['w_max_max_spike']
    clean_max = summary['w_max_max_clean']
    gaps  = summary['gaps']

    if start:
        lines.append(f"  Inizio dati    : {start.strftime('%Y-%m-%d %H:%M')} UTC")
    if end:
        lines.append(f"  Fine dati      : {end.strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append(f"  Record validi  : {summary['total_valid']} "
                 f"(puliti: {summary['clean_count']}, spike: {summary['spike_count']})")
    if spike_max > 0:
        lines.append(f"  Spike massimo  : {spike_max:.0f} W  (record con watt > {THRESHOLD_W})")
    else:
        lines.append(f"  Spike massimo  : nessun picco > {THRESHOLD_W} W")
    lines.append(f"  w_max massimo (dati puliti): {clean_max:.0f} W")
    lines.append(f"  Zero w_medio   : {summary['zero_count']} record")

    if gaps:
        lines.append(f"  Buchi rilevati : {len(gaps)}")
        for g_start, g_end, g_h in gaps[:10]:  # max 10 buchi mostrati
            lines.append(
                f"    da {g_start.strftime('%Y-%m-%d %H:%M')} "
                f"a {g_end.strftime('%Y-%m-%d %H:%M')} "
                f"({g_h:.1f} h)"
            )
        if len(gaps) > 10:
            lines.append(f"    ... e altri {len(gaps)-10} buchi")
    else:
        lines.append("  Buchi rilevati : nessuno")

    # verifica affermazioni specifiche per IMEI noto
    checks = []
    if imei == "86684007269866":
        checks.append(("Tutti dati ragionevoli", spike_max == 0,
                        f"spike_max={spike_max:.0f}W"))
    if imei == "86853106210447":
        checks.append(("Pochissimi dati, inutilizzabile",
                        summary['total_valid'] < 5000,
                        f"total_valid={summary['total_valid']}"))
    if imei in ("86853106211179", "86684007269888", "86684007269869"):
        checks.append((f"Picchi assurdi a ~60000 W",
                        spike_max > 50000,
                        f"spike_max={spike_max:.0f}W"))
    if imei == "86853106211173":
        checks.append((f"Picchi fino a 14000",
                        spike_max >= 10000,
                        f"spike_max={spike_max:.0f}W"))

    if checks:
        lines.append("  Verifiche specifiche:")
        for desc, ok, detail in checks:
            status = "OK" if ok else "NON CONFERMATO"
            lines.append(f"    [{status}] {desc}  — {detail}")

    return lines


def write_verification_report(summaries, out_path):
    """Scrive il file di verifica completo."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 78 + "\n")
        f.write("  NILM - VERIFICA QUALITÀ DATI\n")
        f.write(f"  Generato il: {now_str}\n")
        f.write(f"  Soglia spike: {THRESHOLD_W} W\n")
        f.write("=" * 78 + "\n\n")

        f.write("NOTA: Questo documento riscrive e verifica le informazioni\n")
        f.write("      contenute nel file originale NILM_dati.rtf.\n\n")
        f.write("Legenda verifica:\n")
        f.write("  [OK]            = l'affermazione è confermata dai dati\n")
        f.write("  [NON CONFERMATO]= l'affermazione non è confermata o è parzialmente diversa\n\n")
        f.write("-" * 78 + "\n\n")

        # ordina per IMEI
        for summary in sorted(summaries, key=lambda s: s['imei']):
            imei = summary['imei']
            f.write(f"IMEI: {imei}\n")
            lines = verify_claim(summary, imei)
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

        f.write("=" * 78 + "\n")
        f.write("RIEPILOGO PICCHI > 10000 W\n")
        f.write("=" * 78 + "\n")
        has_spikes = [s for s in summaries if s['spike_count'] > 0]
        if has_spikes:
            for s in sorted(has_spikes, key=lambda x: x['spike_count'], reverse=True):
                f.write(
                    f"  {s['imei']}: {s['spike_count']} record anomali, "
                    f"picco massimo {s['w_max_max_spike']:.0f} W\n"
                )
        else:
            f.write("  Nessun IMEI con picchi > 10000 W\n")
        f.write("\n")

        f.write("=" * 78 + "\n")
        f.write("RIEPILOGO BUCHI NEI DATI (> 2 ore)\n")
        f.write("=" * 78 + "\n")
        for s in sorted(summaries, key=lambda x: len(x['gaps']), reverse=True):
            if s['gaps']:
                f.write(f"  {s['imei']}: {len(s['gaps'])} buchi\n")
        f.write("\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, 'iot_devices*.json')))
    if not json_files:
        print(f"Nessun file JSON trovato in {JSON_DIR}")
        return

    ensure_dir(OUTPUT_DIR)
    summaries = []

    for path in json_files:
        imei = imei_from_path(path)
        print(f"\n{'='*60}")
        print(f"Elaborazione IMEI: {imei}")
        print(f"  File: {os.path.basename(path)}")

        # cartelle output per questo IMEI
        imei_dir    = os.path.join(OUTPUT_DIR, imei)
        daily_dir   = os.path.join(imei_dir, 'daily_plots')
        spike_dir   = os.path.join(imei_dir, 'spike_plots')
        spike_txt   = os.path.join(imei_dir, f'spikes_{imei}.txt')

        ensure_dir(imei_dir)

        # carica dati
        print("  Caricamento dati...")
        records_clean, records_spike, all_epochs = load_imei_data(path)
        print(f"  Record puliti: {len(records_clean)}, Record spike: {len(records_spike)}")

        # statistiche
        summary = compute_summary(records_clean, records_spike, all_epochs, imei)
        summaries.append(summary)
        print(f"  Periodo: {summary['start_dt']} → {summary['end_dt']}")
        print(f"  Buchi > 2h: {len(summary['gaps'])}")

        # grafici giornalieri (dati puliti)
        print("  Generazione grafici giornalieri (w_max + w_medio)...")
        plot_daily(records_clean, imei, daily_dir)
        print(f"  Grafici salvati in: {daily_dir}")

        # gestione spike
        if records_spike:
            print(f"  Trovati {len(records_spike)} picchi > {THRESHOLD_W} W — picco max: {summary['w_max_max_spike']:.0f} W")

            # file testo spike
            write_spike_file(records_spike, imei, spike_txt)
            print(f"  File spike salvato: {spike_txt}")

            # grafici giorni con spike
            print("  Generazione grafici giorni con spike...")
            plot_spike_days(records_clean, records_spike, imei, spike_dir)
            print(f"  Grafici spike salvati in: {spike_dir}")
        else:
            print(f"  Nessun picco > {THRESHOLD_W} W")

    # report di verifica
    report_path = os.path.join(OUTPUT_DIR, 'verification_report.txt')
    print(f"\n{'='*60}")
    print("Scrittura report di verifica...")
    write_verification_report(summaries, report_path)
    print(f"Report salvato: {report_path}")
    print("\nElaborazione completata.")


if __name__ == '__main__':
    main()
