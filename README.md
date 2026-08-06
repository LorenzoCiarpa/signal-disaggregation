# NILM — Disaggregazione dei consumi elettrici

Disaggregazione **Non-Intrusive Load Monitoring** (NILM): a partire dal solo consumo
elettrico **aggregato** di un'abitazione, misurato da un unico dispositivo IoT, il sistema
stima quanto ciascun elettrodomestico ha consumato nel tempo.

Il problema è mal posto — molte combinazioni di apparecchi producono lo stesso segnale
aggregato — e viene reso trattabile usando le dichiarazioni di un **questionario** compilato
dall'utente (quali apparecchi possiede, quanto spesso e a che ora li usa, in quali mesi) come
*prior*: ogni dichiarazione diventa una **penalità** nella funzione obiettivo, mai un vincolo
rigido, così che il segnale possa smentirla quando l'evidenza è sufficientemente forte.

Il modello è formulato come **programmazione lineare mista intera (MILP)** con errore di
ricostruzione L1, risolto giorno per giorno con il solver libero **HiGHS**.

> La formulazione matematica completa del modello è documentata a parte in
> `presentation/delivery/` (non trattata in questo README).

---

## Requisiti e installazione

```bash
pip install -r requirements.txt
```

Dipendenze (`requirements.txt`):

| Pacchetto    | Uso |
|--------------|-----|
| `pandas`     | serie temporali, I/O CSV |
| `numpy`      | algebra vettoriale nella costruzione del modello |
| `scipy`      | filtri di segnale nel preprocessing (`medfilt`, `savgol_filter`, `binary_closing`) |
| `matplotlib` | grafici di output |
| `highspy`    | interfaccia Python al solver MILP HiGHS |

---

## Esecuzione

Dalla radice del progetto:

```bash
# tutti gli IMEI utilizzabili, modello di default
python3 scripts/run_disaggregation.py

# un singolo IMEI, senza grafici (più veloce)
python3 scripts/run_disaggregation.py --imei 86853106211162 --plots none

# variante a 30 minuti (meno variabili, più rapida)
python3 scripts/run_disaggregation.py --approach highs_survey_prior_30min
```

Argomenti principali:

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--imei` | tutti e 6 gli IMEI utilizzabili | IMEI singolo da elaborare |
| `--approach` | `all` | approccio/i da eseguire (`highs_survey_prior`, `highs_survey_prior_30min`) |
| `--plots` | `daily` | grafici temporali da salvare: `none` / `daily` / `weekly` / `both` |
| `--verbose` | off | mostra l'output del solver HiGHS per ogni giorno |
| `--json-dir` | `json` | cartella dei segnali IoT grezzi |
| `--output-dir` | `analysis` | cartella dei risultati |
| `--signal-source` | `plateau_max_mediana` | segnale da disaggregare: grezzo o preprocessato |

---

## Struttura del progetto

```
signal-disaggregation/
├── scripts/
│   ├── run_disaggregation.py        # entrypoint / orchestratore
│   └── nilm/                        # package con la logica NILM
│       ├── preprocessing.py
│       ├── devices.py
│       ├── time_windows.py
│       ├── highs_methods.py
│       ├── approach_highs_survey_prior.py
│       └── output.py
├── json/                            # INPUT — segnali IoT grezzi (uno per IMEI)
├── device_usage_by_imei_v2/         # INPUT — prior del questionario (uno per IMEI)
├── NILM Questionario.xlsx           # INPUT — questionario sorgente
├── analysis/                        # OUTPUT — risultati per (IMEI, approccio)
├── requirements.txt
└── README.md
```

### Codice — `scripts/`

| File | Ruolo |
|------|-------|
| **`run_disaggregation.py`** | Entrypoint. Carica i segnali e i profili dispositivo, seleziona i blocchi di giorni da analizzare (3 blocchi da 5 giorni, con seme fisso ed esclusione delle festività), esegue l'approccio scelto giorno per giorno e salva i risultati. Definisce `APPROACH_MAP`, il registro degli approcci disponibili. |

### Codice — `scripts/nilm/` (package)

| File | Ruolo |
|------|-------|
| **`__init__.py`** | Marker di package Python. |
| **`preprocessing.py`** | Carica i JSON grezzi da `json/`, filtra e ricampiona il segnale al minuto. Espone `get_usable_imeis()` (lista `USABLE_IMEIS` dei 6 IMEI con dati sufficienti), `load_imei()` e `load_imei_plateau_median()` (variante con appiattimento dei plateau e filtro mediano). |
| **`devices.py`** | Base di conoscenza degli elettrodomestici. Definisce la dataclass `DeviceProfile`, il dizionario `DEVICE_KNOWLEDGE_BASE` (potenze e durate tipiche di default) e `get_device_profiles(imei)`, che unisce i default con le dichiarazioni del questionario lette da `device_usage_by_imei_v2/`. |
| **`time_windows.py`** | Utilità sulle fasce orarie condivise dal solver: fattore di penalità temporale graduato con la distanza dalla fascia dichiarata, misura di genericità della fascia, maschera di stagionalità (mesi attivi). |
| **`highs_methods.py`** | Il **solver**. `constrained_highs_full()` costruisce il modello MILP di un singolo giorno (variabili, vincoli, funzione obiettivo L1) e lo risolve con HiGHS. Always-on come variabili, 3 livelli di potenza per apparecchio, durate flessibili, quota settimanale, stagionalità. |
| **`approach_highs_survey_prior.py`** | L'**approccio in uso**. Traduce i profili del questionario nei parametri del solver, propaga la quota settimanale di utilizzi fra i giorni, invoca `constrained_highs_full()` per ogni giornata e ricompone il risultato. È l'unica voce di `APPROACH_MAP`. |
| **`output.py`** | `save_results()`: per ogni coppia (IMEI, approccio) scrive `disaggregation.csv`, i grafici temporali (`weekly_plots/`, `daily_plots/`) e il grafico riepilogativo dell'energia (`energy_report.png`). |

### Dati di input

#### `json/` — segnali IoT grezzi
Un file per abitazione, denominato `iot_devices.<IMEI>.json` (dieci IMEI presenti). Ogni file è
una **lista di record al minuto** del dispositivo IoT. Campi principali di ogni record:

| Campo | Significato |
|-------|-------------|
| `_id` / `t_*_absolute` | timestamp epoch (secondi) |
| `w_medio` | potenza media nel minuto (W) — è il segnale da disaggregare |
| `w_minimo`, `w_massimo` | potenza minima e massima nel minuto (W) |
| `w_istantaneo` | ultimo campione istantaneo (W) |
| `epoch_valid` | flag di validità del record |
| `granularity` | secondi effettivamente coperti dal minuto |

> `json/` è nel `.gitignore`: i segnali sono dati locali, non versionati nel repository.

#### `device_usage_by_imei_v2/` — prior del questionario
Un file per abitazione, `<IMEI>.json` (schema v2). Traduce in forma numerica le risposte del
questionario per quell'abitazione. Struttura: `{ schema_version, imei, devices: { <nome>: {...} } }`.
Per ogni elettrodomestico:

| Campo | Significato |
|-------|-------------|
| `present`, `count_min/max` | se l'apparecchio è presente e in quante unità |
| `usage_frequency_per_week_min/max` | quante volte a settimana viene usato |
| `duration_minutes_min/max` | durata tipica di un ciclo |
| `start_window_start_min/end_min`, `..._wraps_next_day` | fascia oraria di avvio (minuti dalla mezzanotte, con eventuale scavalco della mezzanotte) |
| `daily_usage_hours_min/max` | ore al giorno di accensione |
| `active_months` | mesi dell'anno in cui l'apparecchio viene usato (stagionalità) |

Questi campi diventano le **penalità** del modello (fascia oraria, quota settimanale, durata dei
cicli, stagionalità) e sono la sorgente letta da `get_device_profiles()`.

#### `NILM Questionario.xlsx`
Il questionario sorgente in formato Excel, da cui derivano i file numerici in
`device_usage_by_imei_v2/`. È materiale di riferimento; non viene letto a runtime.

### Dati di output

#### `analysis/` — risultati
Generata dall'esecuzione, organizzata per `analysis/<IMEI>/<approccio>/`:

| Elemento | Contenuto |
|----------|-----------|
| `disaggregation.csv` | serie temporale con `w_total`, una colonna per dispositivo e il `residuo` |
| `weekly_plots/`, `daily_plots/` | grafici temporali per settimana / giorno |
| `energy_report.png` | riepilogo dell'energia (kWh) attribuita per dispositivo |

> `analysis/` è nel `.gitignore`: gli output vengono rigenerati a ogni esecuzione.

---

## Flusso dei dati

```
json/<IMEI>.json ─────────────────► preprocessing.py ──► segnale al minuto
device_usage_by_imei_v2/<IMEI>.json ──► devices.py ────► DeviceProfile (con prior)
                                             │
                                             ▼
               approach_highs_survey_prior.py   (un modello MILP per giorno)
                                             │
                        highs_methods.constrained_highs_full()  ──► HiGHS
                                             │
                                             ▼
                        output.py    ──►  analysis/<IMEI>/<approccio>/
```
