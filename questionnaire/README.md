# Nuovo questionario NILM (schema v3) — prototipo

Prototipo Streamlit del questionario ridisegnato. **È completamente separato dal
resto del progetto**: non importa nulla da `scripts/` e non tocca
`device_usage_by_imei_v2/`. La pipeline esistente continua a funzionare con il
questionario vecchio (schema v2).

```bash
streamlit run questionnaire/app.py     # dipendenze: streamlit, pandas, openpyxl
```

## Cosa cambia rispetto al questionario attuale

**Rimosso** — tipologia abitazione, superficie, anno di costruzione, numero di
stanze, tipo di riscaldamento, pannelli fotovoltaici, numero di persone e fasce
d'età, tabella di presenza in casa, note in testo libero, domande su acquisti /
assenze / cambi di abitudine.

**Aggiunto**
- potenza contrattuale della casa (kW);
- carico base sempre presente (W) con la relativa confidenza — evita che router,
  allarme e standby vengano attribuiti agli elettrodomestici;
- potenze dichiarate per dispositivo: tipica, minima, di picco;
- marca/modello opzionale che **precompila** i consumi dal catalogo;
- livello di confidenza per ogni dispositivo, per pesare i vincoli come soft.

**Modificato** — il questionario non è più a elenco fisso. L'utente aggiunge i
dispositivi che vuole scegliendo da un catalogo (`catalog/device_catalog.json`),
e fasce orarie / frequenze / durate vengono chieste per **tutti** i dispositivi,
non solo per i quattro principali.

## Il cambio strutturale: istanze invece di tipologie

Nello schema v2 `devices` è un dict `{tipologia: {...}}` con `count_min/max`: due
climatizzatori condividono per forza le stesse fasce orarie. Qui `devices` è una
**lista di istanze indipendenti**, ognuna con etichetta, consumi e orari propri.

Da qui deriva anche l'unificazione di "frigorifero principale/secondario" in un
unico `Frigorifero`: quella distinzione serviva solo perché la lista fissa dello
schema v2 concedeva una riga per tipologia. Con le istanze si dichiarano N
frigoriferi, uguali o diversi, distinti dall'etichetta (`Cucina`, `Garage`).

```jsonc
"devices": [
  {"id": "climatizzatore-a1b2c3", "type": "Climatizzatore", "label": "Camera",
   "daily_usage_hours_min": 6, "daily_usage_hours_max": 8,
   "start_window_start_min": 1380, "start_window_end_min": 420,
   "start_window_wraps_next_day": true, "active_months": [6,7,8,9],
   "p_typical_w": 1100.0, "confidence": "media"},
  {"id": "climatizzatore-d4e5f6", "type": "Climatizzatore", "label": "Salotto",
   "daily_usage_hours_min": 2, "daily_usage_hours_max": 4, "...": "..."}
]
```

Per usare questi dati nei modelli servirà un adattatore che converta le istanze
in `DeviceProfile` (`scripts/nilm/devices.py`), oppure un caricatore v3 dedicato.
Non è incluso qui, per scelta: il prototipo resta isolato.

## Tre modalità di utilizzo

Ogni tipologia del catalogo dichiara un `usage_mode`, che decide quali domande
vengono mostrate:

| modalità | domande | esempi |
|---|---|---|
| `cycle` | frequenza settimanale, durata del ciclo, fascia di avvio | lavatrice, forno, auto elettrica |
| `daily_hours` | ore medie al giorno, fascia di utilizzo | climatizzatore, TV, computer |
| `always_on` | nessuna (solo potenza e duty cycle) | frigorifero, congelatore |

Le fasce a cavallo della mezzanotte sono gestite: se l'orario di fine precede
quello di inizio, `start_window_wraps_next_day` viene impostato automaticamente.

## File

| file | contenuto |
|---|---|
| `app.py` | interfaccia Streamlit |
| `schema.py` | dataclass, validazione, export JSON/righe Excel (nessuna dipendenza da Streamlit) |
| `catalog/device_catalog.json` | 18 tipologie con potenze di default e modelli |
| `output/` | destinazione del salvataggio locale (creata al primo salvataggio) |

L'export produce sia il JSON (schema v3) sia un Excel a due fogli: `household`
con i dati casa, `devices` con una riga per istanza.

## Limite noto del catalogo

I modelli in `catalog/device_catalog.json` sono uno **stub dimostrativo** con
marca "Generico" e potenze indicative, sufficiente per mostrare il meccanismo di
precompilazione. Per l'uso reale va sostituito con un database prodotti
(etichette energetiche europee, API produttori): l'app dipende solo dalla firma
di `lookup_model()`, quindi basta riscrivere quella funzione.
