# NILM Signal Disaggregation — Documentazione

## Indice

1. [Panoramica del progetto](#panoramica-del-progetto)
2. [Struttura del repository](#struttura-del-repository)
3. [Dati di input](#dati-di-input)
4. [Pipeline di disaggregazione](#pipeline-di-disaggregazione)
5. [I 5 approcci implementati](#i-5-approcci-implementati)
6. [Output atteso](#output-atteso)
7. [Come eseguire la disaggregazione](#come-eseguire-la-disaggregazione)
8. [Stato di completamento](#stato-di-completamento)

---

## Panoramica del progetto

Il progetto affronta il problema del **NILM (Non-Intrusive Load Monitoring)**: dato il consumo
energetico aggregato di un'abitazione misurato da un singolo dispositivo IoT, si stima **quali
elettrodomestici sono accesi in ogni istante**, senza etichette di ground truth.

Il modello matematico è:

```
w_medio(t) ≈ Σ_i  P_i · x_i(t)
```

dove `P_i` è la potenza tipica del dispositivo i-esimo e `x_i(t) ∈ {0,1}` indica se è acceso.

Sono implementati **5 approcci unsupervised** che vengono confrontati tramite un benchmark proxy
(senza labels).

---

## Struttura del repository

```
signal-disaggregation/
├── json/                          # Dati grezzi: un file JSON per IMEI
│   └── iot_devices.{imei}.json
├── device_usage_by_imei/          # Inventario elettrodomestici per IMEI (da survey)
│   └── {imei}.json
├── scripts/
│   ├── run_disaggregation.py      # 🚀 Script principale di orchestrazione
│   └── nilm/                      # Package Python con tutta la logica
│       ├── __init__.py
│       ├── preprocessing.py       # Caricamento, filtraggio, ricampionamento dati
│       ├── devices.py             # Knowledge base dispositivi + inventario per IMEI
│       ├── approach_event_based.py  # Approccio A: Event-based
│       ├── approach_hmm.py          # Approccio B: HMM per singolo dispositivo
│       ├── approach_fhmm.py         # Approccio C: Factorial HMM semplificato
│       ├── approach_template.py     # Approccio D: Template matching
│       ├── approach_event_prior.py  # Approccio E: Event-based con prior bayesiano
│       ├── output.py              # Generazione CSV, grafici giornalieri, report energetico
│       └── benchmark.py           # Metriche proxy e ranking degli approcci
├── analysis/                      # Output generato dalla pipeline
├── other_data/                    # Statistiche e CSV di supporto
├── requirements.txt               # Dipendenze: pandas, matplotlib, hmmlearn
└── docs/                          # Questa documentazione
```

### Moduli chiave in `scripts/nilm/`

| Modulo | Responsabilità |
|--------|----------------|
| `preprocessing.py` | Carica i JSON, filtra record invalidi e spike (>10kW), ricampiona a 1 min, gestisce i buchi (NaN) |
| `devices.py` | Definisce `DeviceProfile` (knowledge base dispositivi con potenza tipica, durata, duty cycle), carica l'inventario per IMEI e assegna `prior_weight` (1.0=presente, 0.05=assente) |
| `approach_*.py` | Ognuno implementa una funzione `run(signal, devices) -> dict[str, pd.Series]` con lo stesso formato di output |
| `output.py` | Salva CSV di disaggregazione, grafici giornalieri (PNG), report energetico a barre (kWh) |
| `benchmark.py` | Calcola metriche di qualità proxy (MAE, RMSE, errore energetico, consistenza temporale) e produce ranking + heatmap |

---

## Dati di input

### File JSON di consumo (`json/iot_devices.{imei}.json`)

Ogni file contiene un array di record temporali (~1 per minuto):

| Campo | Descrizione |
|-------|-------------|
| `_id` | Epoch Unix in secondi (possibile formato `{"$numberLong": "..."}`) |
| `epoch_valid` | Flag qualità: se `false`, il record viene scartato |
| `w_medio` | **Potenza media** nell'intervallo (Watt) — metrica principale |
| `w_massimo` / `w_minimo` | Picchi e minimi nell'intervallo |
| `granularity` | Durata intervallo in secondi (tipicamente 60-61) |

### Inventario dispositivi (`device_usage_by_imei/{imei}.json`)

Per ogni IMEI, contiene la lista di elettrodomestici dichiarati dagli utenti con flag `present`
e frequenza d'uso. **Non esistono etichette temporali**: non sappiamo quando ogni dispositivo è acceso.

### IMEI utilizzabili

Solo **6 IMEI** hanno dati sufficienti (>50k record):

- `86853106211162`
- `86684007269866`
- `86853106211179`
- `86684007269887`
- `86853106211173`
- `86684007269889`

---

## Pipeline di disaggregazione

La pipeline viene eseguita dallo script `scripts/run_disaggregation.py` e segue questi passaggi:

```
1. PREPROCESSING (preprocessing.py)
   └─ Per ogni IMEI: carica JSON → filtra epoch_valid=false → filtra spike >10kW
      → converte a DatetimeIndex UTC → ricampiona a 1 min → NaN nei buchi

2. CARICAMENTO DISPOSITIVI (devices.py)
   └─ Per ogni IMEI: carica inventario survey → crea 16 DeviceProfile
      con potenza tipica, durata, duty cycle, prior_weight

3. DISAGGREGAZIONE (5 approcci, uno alla volta)
   └─ Ogni approccio riceve (signal, devices) e restituisce
      dict[device_name → pd.Series di potenza stimata]

4. OUTPUT (output.py)
   └─ Per ogni (IMEI, approccio): salva CSV + grafici giornalieri + report energetico

5. BENCHMARK (benchmark.py)
   └─ Calcola metriche proxy per tutte le combinazioni (IMEI, approccio)
      → salva CSV + heatmap di confronto → stampa ranking
```

---

## I 5 approcci implementati

### Approccio A — Event-based (`approach_event_based.py`)

**Idea**: rileva transizioni ON/OFF cercando variazioni brusche nel segnale (`delta_w = signal.diff()`).

- Rileva eventi ON quando `delta_w > threshold` (default 50W)
- Rileva eventi OFF quando `delta_w < -threshold`
- Per ogni evento ON: seleziona il dispositivo **presente** con `|p_typical - |delta_w||` minimo
- Abbina ogni ON con il primo OFF compatibile entro `dur_typical × 3` minuti
- Se non trova OFF: usa `dur_typical_min` come durata di default
- Output: serie con 0 (OFF) o `p_typical_w` (ON) per ogni dispositivo

### Approccio B — HMM per singolo dispositivo (`approach_hmm.py`)

**Idea**: per ogni dispositivo presente, fitta un HMM Gaussiano a 2 stati (ON/OFF) sul segnale residuo.

- Usa `GaussianHMM(n_components=2)` di `hmmlearn`
- Inizializza le medie degli stati a `[5W, p_typical_w]` (OFF, ON)
- Fitta sul segnale residuo (aggregato meno dispositivi già stimati) — approccio sequenziale
- Dopo il fit, `predict()` determina la sequenza di stati
- Lo stato con media più alta viene mappato a ON
- Dispositivi processati uno alla volta; ogni stima viene sottratta dal residuo per il successivo

### Approccio C — Factorial HMM semplificato (`approach_fhmm.py`)

**Idea**: modella **tutti i dispositivi simultaneamente** con variabili latenti binarie indipendenti.

- Inferenza greedy (coordinate-ascent): ad ogni iterazione, per ogni dispositivo:
  - Calcola il residuo escludendo il contributo degli altri
  - Imposta `x_i(t) = 1` se `residuo(t) > p_typical / 2` (minimizza errore)
- Itera fino a convergenza (`< tol × T` cambi di stato) o `max_iter` (default 50)
- **Smoothing temporale**: dopo la convergenza, rimuove blocchi ON più brevi di `dur_min_min / 2`
- Vantaggio: cattura combinazioni simultanee senza esplosione combinatoria degli stati

### Approccio D — Template matching (`approach_template.py`)

**Idea**: confronta il segnale con template rettangolari costruiti dalla knowledge base.

- Per ogni dispositivo (presente, `dur_typical >= 3 min`): crea un template rettangolare
  di potenza `p_typical_w` e lunghezza `dur_typical_min` campioni
- Calcola la **cross-correlazione normalizzata** (`scipy.signal.correlate`)
- Rileva picchi di correlazione > `corr_threshold` (default 0.6)
- Assegnazione greedy ordinata per score decrescente (first-come-first-served, no overlap)
- I dispositivi più potenti vengono processati per primi

### Approccio E — Event-based con prior bayesiano (`approach_event_prior.py`)

**Idea**: come l'Approccio A, ma assegna gli eventi con uno **score bayesiano** che tiene conto
dell'inventario (dispositivi presenti/assenti) e della frequenza d'uso dichiarata nel survey.

- Stesso rilevamento eventi di A (delta_w > threshold)
- Per ogni evento, per ogni dispositivo calcola:
  - `likelihood = N(|delta_w|; p_typical, (sigma_factor × p_typical)²)` — gaussiana
  - `frequency_factor = min(frequency_per_week / 7, 1.0)`
  - `score = likelihood × prior_weight × frequency_factor`
- Assegna l'evento al dispositivo con **score massimo**
- Considera **tutti** i dispositivi (presenti con prior=1.0, assenti con prior=0.05)

---

## Output atteso

Dopo l'esecuzione, la cartella `analysis/` conterrà:

```
analysis/
├── benchmark_results.csv          # Metriche per ogni (IMEI, approccio)
├── benchmark_summary.png          # Heatmap di confronto tra approcci
├── run_log.txt                    # Log con timestamp e errori
└── {imei}/
    └── {approccio}/               # event, hmm, fhmm, template, event_prior
        ├── disaggregation.csv     # Serie temporale disaggregata (ISO8601)
        ├── energy_report.png      # Grafico a barre kWh per dispositivo
        └── daily_plots/
            └── {YYYY-MM-DD}.png   # Grafico giornaliero per ogni giorno
```

### Dettaglio dei file di output

#### `disaggregation.csv`

CSV con colonne:
- `timestamp` — formato ISO8601 (es. `2025-10-15T08:30:00Z`)
- `w_total` — potenza aggregata misurata
- Una colonna per ogni dispositivo (potenza stimata: 0 o `p_typical_w`)
- `residuo` — differenza tra misurato e somma dei dispositivi stimati

#### Grafici giornalieri (`daily_plots/{YYYY-MM-DD}.png`)

Per ogni giorno con dati disponibili:
- Linea grigia: segnale aggregato misurato ("Totale misurato")
- Aree colorate sovrapposte: contributi stimati per ogni dispositivo attivo (media > 5W)
- Linea nera tratteggiata: residuo non spiegato
- Asse x in formato HH:MM, legenda con nomi dispositivi

#### Report energetico (`energy_report.png`)

Grafico a barre orizzontali:
- Mostra i kWh stimati per ogni dispositivo su tutto il periodo
- Include il residuo
- Ogni barra ha etichetta con valore kWh e percentuale sul totale

#### Benchmark (`benchmark_results.csv` + `benchmark_summary.png`)

Metriche calcolate per ogni combinazione (IMEI, approccio):

| Metrica | Significato |
|---------|-------------|
| `mae_recon` | Errore medio assoluto di ricostruzione (W) |
| `rmse_recon` | Errore quadratico medio di ricostruzione (W) |
| `energy_error_pct` | Errore percentuale sull'energia totale |
| `residuo_medio_w` | Potenza media residua non spiegata (W) |
| `n_devices_found` | Numero di dispositivi con >1% dell'energia totale |
| `temporal_consistency` | % di blocchi ON con durata plausibile |

Il ranking finale ordina gli approcci per `mae_recon` crescente (migliore = errore più basso).

---

## Come eseguire la disaggregazione

### 1. Installare le dipendenze

```bash
pip install -r requirements.txt
```

Dipendenze richieste: `pandas`, `matplotlib`, `hmmlearn>=0.3` (che installa anche `scipy`, `numpy`).

### 2. Eseguire la pipeline completa

Dalla root del progetto:

```bash
python3 scripts/run_disaggregation.py
```

Questo processa tutti i 6 IMEI con tutti e 5 gli approcci, genera grafici e benchmark.

### 3. Opzioni CLI

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--imei IMEI` | tutti i 6 | Processa solo un singolo IMEI |
| `--approach {event,hmm,fhmm,template,event_prior,all}` | `all` | Esegue solo un approccio specifico |
| `--no-plots` | disattivato | Salta i grafici giornalieri (molto più veloce) |
| `--json-dir DIR` | `json` | Cartella con i file JSON di input |
| `--output-dir DIR` | `analysis` | Cartella di output |

### Esempi di utilizzo

```bash
# Solo un IMEI, solo l'approccio FHMM
python3 scripts/run_disaggregation.py --imei 86853106211162 --approach fhmm

# Tutti gli IMEI, tutti gli approcci, senza grafici giornalieri (veloce)
python3 scripts/run_disaggregation.py --no-plots

# Solo event-based, con output in una cartella diversa
python3 scripts/run_disaggregation.py --approach event --output-dir results/

# Vedere l'help completo
python3 scripts/run_disaggregation.py --help
```

### Nota sulle performance

- L'esecuzione completa (6 IMEI × 5 approcci + grafici giornalieri) può richiedere diversi minuti.
- Con `--no-plots` si risparmia molto tempo: i grafici giornalieri sono il collo di bottiglia.
- L'approccio HMM è il più lento per la fase di fitting del modello.
- L'approccio FHMM è computazionalmente intensivo per le iterazioni di convergenza.

---

## Stato di completamento

Tutte le 10 User Story del PRD sono state implementate e verificate:

| US | Titolo | Stato |
|----|--------|-------|
| US-001 | Pipeline di preprocessing | ✅ Completata |
| US-002 | Knowledge base dispositivi | ✅ Completata |
| US-003 | Approccio A — Event-based | ✅ Completata |
| US-004 | Approccio B — HMM | ✅ Completata |
| US-005 | Approccio C — Factorial HMM | ✅ Completata |
| US-006 | Approccio D — Template matching | ✅ Completata |
| US-007 | Approccio E — Event-based con prior | ✅ Completata |
| US-008 | Modulo di output | ✅ Completata |
| US-009 | Benchmark proxy | ✅ Completata |
| US-010 | Script di orchestrazione | ✅ Completata |
