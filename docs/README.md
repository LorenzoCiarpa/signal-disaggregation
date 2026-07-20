# NILM Signal Disaggregation — Documentazione

## Indice

1. [Panoramica del progetto](#panoramica-del-progetto)
2. [Struttura del repository](#struttura-del-repository)
3. [Dati di input](#dati-di-input)
4. [Pipeline di disaggregazione](#pipeline-di-disaggregazione)
5. [Gli approcci implementati](#gli-approcci-implementati)
6. [Output atteso](#output-atteso)
7. [Come eseguire la disaggregazione](#come-eseguire-la-disaggregazione)

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

Gli approcci sono tutti **unsupervised** e vengono confrontati tramite un benchmark proxy
(senza labels). Tutti sfruttano il **questionario V2** compilato dagli utenti: finestre orarie
d'uso, durate, frequenze settimanali e mesi di attività entrano nel modello come prior.

---

## Struttura del repository

```
signal-disaggregation/
├── json/                          # Dati grezzi: un file JSON per IMEI
│   └── iot_devices.{imei}.json
├── device_usage_by_imei_v2/       # Inventario elettrodomestici per IMEI (da questionario V2)
│   └── {imei}.json
├── scripts/
│   ├── run_disaggregation.py      # 🚀 Script principale di orchestrazione
│   └── nilm/                      # Package Python con tutta la logica
│       ├── __init__.py
│       ├── preprocessing.py       # Caricamento, filtraggio, ricampionamento dati
│       ├── devices.py             # Knowledge base dispositivi + inventario per IMEI
│       ├── baseline_load.py       # Stima della baseline always-on (frigo, congelatore)
│       ├── approach_fhmm_survey.py        # FHMM con prior da questionario
│       ├── approach_hsmm_survey.py        # HSMM con Viterbi e prior da questionario
│       ├── approach_gurobi_activation.py  # MIQP Gurobi, baseline sottratta
│       ├── approach_gurobi_full.py        # MIQP Gurobi, always-on come variabili
│       ├── approach_gurobi_soft_duration.py  # MIQP Gurobi, durata soft
│       ├── approach_gurobi_weekly_quota.py   # MIQP Gurobi, quota settimanale accensioni
│       ├── approach_cvxpy*.py             # Equivalenti MILP L1 su solver libero (HiGHS)
│       ├── gurobi_methods.py      # Solver MIQP: solve_activation, solve_full
│       ├── time_windows.py       # Finestre orarie: maschere e penalità graduate
│       ├── highs_methods.py       # Solver MILP L1 equivalenti via HiGHS
│       ├── output.py              # Generazione CSV, grafici settimanali, report energetico
│       └── benchmark.py           # Metriche proxy e ranking degli approcci
├── analysis/                      # Output generato dalla pipeline
├── requirements.txt
└── docs/                          # Questa documentazione
```

### Moduli chiave in `scripts/nilm/`

| Modulo | Responsabilità |
|--------|----------------|
| `preprocessing.py` | Carica i JSON, filtra record invalidi e spike (>10kW), ricampiona a 1 min, gestisce i buchi (NaN) |
| `devices.py` | Definisce `DeviceProfile` (knowledge base dispositivi con potenza tipica, durata, duty cycle), carica l'inventario survey per IMEI e assegna `prior_weight` (1.0=presente, 0.05=assente) |
| `baseline_load.py` | Stima la potenza costante dei device `always_on` da sottrarre dal segnale |
| `approach_*.py` | Ognuno implementa una funzione `run(signal, devices) -> dict[str, pd.Series]` con lo stesso formato di output |
| `gurobi_methods.py` / `highs_methods.py` | Costruiscono e risolvono i modelli di ottimizzazione giorno per giorno |
| `time_windows.py` | Finestre orarie del questionario: maschera dentro/fuori e fattore di penalità graduato per distanza |
| `output.py` | Salva CSV di disaggregazione, grafici settimanali (PNG), report energetico a barre (kWh) |
| `benchmark.py` | Calcola metriche di qualità proxy (MAE, RMSE, errore energetico, consistenza temporale) e produce ranking + heatmap |

### Documenti di dettaglio

| Documento | Contenuto |
|-----------|-----------|
| [approaches_fhmm_hsmm.md](approaches_fhmm_hsmm.md) | Formulazioni di `fhmm_survey` e `hsmm_survey` |
| [approaches_gurobi.md](approaches_gurobi.md) | Formulazioni MIQP di `gurobi_activation`, `gurobi_full` e `gurobi_soft_duration` |
| [approaches_cvxpy.md](approaches_cvxpy.md) | Equivalenti MILP L1 su HiGHS |
| [fhmm_formule.md](fhmm_formule.md) | Derivazione matematica del modello FHMM |

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

### Inventario dispositivi (`device_usage_by_imei_v2/{imei}.json`)

Per ogni IMEI, contiene la lista di elettrodomestici dichiarati dagli utenti con flag `present`
e i campi del questionario V2: frequenza d'uso settimanale (min/max), durata in minuti (min/max),
finestra oraria di accensione, ore d'uso giornaliere e mesi di attività.
**Non esistono etichette temporali**: non sappiamo quando ogni dispositivo è acceso.

I profili vengono caricati da `get_device_profiles()`; i valori del questionario sovrascrivono i
default della `DEVICE_KNOWLEDGE_BASE` (15 dispositivi, di cui 3 `always_on`: frigorifero
principale, frigorifero secondario, congelatore).

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
      → (default) preprocessing plateau_max + mediana

2. CARICAMENTO DISPOSITIVI (devices.py)
   └─ Per ogni IMEI: carica inventario survey V2 → crea 15 DeviceProfile
      con potenza tipica, durata, finestre orarie, frequenza, prior_weight

3. DISAGGREGAZIONE (un approccio alla volta)
   └─ Ogni approccio riceve (signal, devices) e restituisce
      dict[device_name → pd.Series di potenza stimata]

4. OUTPUT (output.py)
   └─ Per ogni (IMEI, approccio): salva CSV + grafici settimanali + report energetico

5. BENCHMARK (benchmark.py)
   └─ Calcola metriche proxy per tutte le combinazioni (IMEI, approccio)
      → salva CSV + heatmap di confronto → stampa ranking
```

### Finestra di analisi

La pipeline non processa l'intero storico ma **15 giorni per IMEI**, come 3 blocchi da 5 giorni
consecutivi distribuiti nel periodo (costanti `_N_BLOCKS`, `_BLOCK_DAYS`, `_BLOCK_SEED` in
`run_disaggregation.py`). Le scelte di progetto:

- **Blocchi identici per tutti gli IMEI** — sono estratti dai giorni presenti in *tutti* i segnali,
  così le differenze di metrica tra case riflettono la casa e non il periodo.
- **Estrazione casuale ma riproducibile** — `random.Random(_BLOCK_SEED)`, quindi rieseguire dà gli
  stessi blocchi.
- **Festività escluse** — nessun blocco può cadere tra `_HOLIDAY_START` e `_HOLIDAY_END`
  (22 dic – 6 gen): i consumi di Natale, Capodanno ed Epifania non sono rappresentativi.
- **Blocchi distanziati** — almeno `_MIN_BLOCK_START_GAP_DAYS` (15) tra un inizio e l'altro, per
  evitare che si addensino nello stesso periodo.
- **Ogni blocco è risolto separatamente** e i risultati vengono concatenati. Concatenare *prima*
  farebbe inventare al resampling migliaia di slot vuoti negli intervalli tra un blocco e l'altro.

Con il seed attuale i blocchi sono **19–23 nov**, **9–13 dic**, **8–12 gen**.
I dati disponibili coprono 2025-11-13 → 2026-02-05 (85 giorni comuni a tutti gli IMEI, 69 dopo
l'esclusione delle festività).

Nota sull'interazione con `*_weekly_quota`: il contatore settimanale riparte a ogni blocco (i
blocchi distano settimane, quindi non c'è continuità da propagare) e si azzera comunque a ogni
lunedì interno al blocco. Un blocco di 5 giorni riceve quindi l'intera quota settimanale del
questionario, il che è leggermente permissivo rispetto a un budget su 7 giorni.

---

## Gli approcci implementati

Due famiglie: modelli probabilistici grafici (nessun solver) e ottimizzazione combinatoria (MIQP/MILP).

### Modelli probabilistici

| Approccio | File | Idea |
|-----------|------|------|
| `fhmm_survey` | `approach_fhmm_survey.py` | Factorial HMM con coordinate-ascent greedy. La soglia di attivazione `p_i/2` è modulata dal questionario (finestra oraria, stagione); il post-processing applica commitment window e cap su durata, frequenza e ore giornaliere. |
| `hsmm_survey` | `approach_hsmm_survey.py` | Hidden Semi-Markov Model con Viterbi segment-based. Le durate entrano come **distribuzione esplicita** dentro l'inferenza invece che come post-processing — formalmente più corretto, ma più lento. |

Dettagli e formule: [approaches_fhmm_hsmm.md](approaches_fhmm_hsmm.md).

### Ottimizzazione

Tutti risolvono un modello **giorno per giorno** su segnale ricampionato a 15 min (o 30), con
3 livelli di potenza per device, penalità di attivazione e finestre temporali soft.

| Approccio | File | Solver | Always-on |
|-----------|------|--------|-----------|
| `gurobi_activation` | `approach_gurobi_activation.py` | Gurobi (MIQP, L2) | stimati come baseline e sottratti |
| `gurobi_full` | `approach_gurobi_full.py` | Gurobi (MIQP, L2) | modellati come variabili, nessuna baseline |
| `gurobi_soft_duration` | `approach_gurobi_soft_duration.py` | Gurobi (MIQP, L2) | come `gurobi_full`, ma durata min/max penalizzata invece che imposta |
| `gurobi_weekly_quota` | `approach_gurobi_weekly_quota.py` | Gurobi (MIQP, L2) | come sopra + quota settimanale di accensioni dal questionario, azzerata ogni lunedì |
| `cvxpy` | `approach_cvxpy.py` | HiGHS (MILP, L1) | baseline, 1 livello binario |
| `cvxpy_activation` | `approach_cvxpy_activation.py` | HiGHS (MILP, L1) | baseline |
| `cvxpy_full` | `approach_cvxpy_full.py` | HiGHS (MILP, L1) | modellati come variabili |
| `cvxpy_soft_duration` | `approach_cvxpy_soft_duration.py` | HiGHS (MILP, L1) | come `cvxpy_full`, con durata soft |
| `cvxpy_weekly_quota` | `approach_cvxpy_weekly_quota.py` | HiGHS (MILP, L1) | come `gurobi_weekly_quota` sul solver libero |

Dettagli: [approaches_gurobi.md](approaches_gurobi.md) e [approaches_cvxpy.md](approaches_cvxpy.md).

### Varianti

Ogni approccio ha varianti registrate in `APPROACH_MAP` che cambiano solo i parametri di
ricampionamento, senza toccare la formulazione:

- `_max` / `_median` — metodo di aggregazione nel resampling (default: `mean`)
- `_30min` — granularità a 30 minuti invece di 15
- `_block` (solo `*_soft_duration`) — solo penalità per-blocco, senza budget giornaliero

---

## Output atteso

Dopo l'esecuzione, la cartella `analysis/` conterrà:

```
analysis/
├── benchmark_results.csv          # Metriche per ogni (IMEI, approccio)
├── benchmark_summary.png          # Heatmap di confronto tra approcci
├── run_log.txt                    # Log con timestamp e errori
└── {imei}/
    └── {approccio}/               # es. gurobi_full, hsmm_survey, ...
        ├── disaggregation.csv     # Serie temporale disaggregata (ISO8601)
        ├── energy_report.png      # Grafico a barre kWh per dispositivo
        └── weekly_plots/
            └── {YYYY-MM-DD}_to_{YYYY-MM-DD}.png   # Grafico settimanale
```

### Dettaglio dei file di output

#### `disaggregation.csv`

CSV con colonne:
- `timestamp` — formato ISO8601 (es. `2025-10-15T08:30:00Z`)
- `w_total` — potenza aggregata misurata
- Una colonna per ogni dispositivo (potenza stimata in W)
- `residuo` — differenza tra misurato e somma dei dispositivi stimati

#### Grafici settimanali (`weekly_plots/{YYYY-MM-DD}_to_{YYYY-MM-DD}.png`)

Per ogni settimana con dati disponibili:
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

`gurobipy` richiede una licenza per problemi oltre la dimensione limite della versione free.
Gli approcci `cvxpy*` usano HiGHS e non richiedono licenza.

### 2. Eseguire la pipeline completa

Dalla root del progetto:

```bash
python3 scripts/run_disaggregation.py
```

Questo processa tutti i 6 IMEI con tutti gli approcci, genera grafici e benchmark.

### 3. Opzioni CLI

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--imei IMEI` | tutti i 6 | Processa solo un singolo IMEI |
| `--approach A [B ...]` | `all` | Uno o più approcci (chiavi di `APPROACH_MAP`) |
| `--no-plots` | disattivato | Salta i grafici settimanali (molto più veloce) |
| `--verbose` | disattivato | Mostra l'output del solver giorno per giorno |
| `--json-dir DIR` | `json` | Cartella con i file JSON di input |
| `--output-dir DIR` | `analysis` | Cartella di output |
| `--signal-source {raw,plateau_max_mediana}` | `plateau_max_mediana` | Segnale grezzo o preprocessato |

### Esempi di utilizzo

```bash
# Un solo IMEI, un solo approccio
python3 scripts/run_disaggregation.py --imei 86853106211162 --approach gurobi_full

# Più approcci a confronto, senza grafici settimanali (veloce)
python3 scripts/run_disaggregation.py --approach fhmm_survey hsmm_survey gurobi_full --no-plots

# Solo solver liberi, output in una cartella diversa
python3 scripts/run_disaggregation.py --approach cvxpy_full --output-dir results/

# Vedere l'help completo (in caso di approccio errato elenca quelli validi)
python3 scripts/run_disaggregation.py --help
```

### Nota sulle performance

- Con `--no-plots` si risparmia molto tempo: i grafici settimanali sono un collo di bottiglia.
- Gli approcci `gurobi_*` sono i più lenti: MIQP con time limit di 60s per giorno.
- `gurobi_full` è più costoso di `gurobi_activation` (aggiunge le variabili always-on).
- `fhmm_survey` è il più veloce (frazioni di secondo); `hsmm_survey` costa qualche secondo.
- Le varianti `_30min` dimezzano il numero di slot rispetto alle `15min`.
