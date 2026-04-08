# PRD: NILM Signal Disaggregation — Implementazioni e Benchmark

## Introduction

Implementare un sistema di **signal disaggregation** (NILM — Non-Intrusive Load Monitoring) su
dati reali di consumo energetico aggregato di abitazioni italiane. Il sistema deve stimare quali
elettrodomestici sono accesi minuto per minuto, senza etichette di ground truth e senza addestrare
nuovi modelli da zero. Vengono implementati 5 approcci stato dell'arte e confrontati tramite
metriche proxy (senza label).

---

## Contesto dati

### IMEI utilizzabili (criteri: >50k record validi dopo pulizia spike)

| IMEI | Record validi | Spike | Zero w_medio | Note |
|------|--------------|-------|--------------|------|
| 86853106211162 | 120,951 | 0 | 139 | Migliore qualità, nessun spike |
| 86684007269866 | 120,951 | 1 | 2 | Ottima qualità, 1 gap piccolo (5.3h) |
| 86853106211179 | 114,969 | 62 | 5,880 | Buona, spike rimovibili, 1 gap (101.4h) |
| 86684007269887 | 109,118 | 94 | 17,423 | Usabile con cautela, molti zeri |
| 86853106211173 | 109,051 | 235 | 45,740 | Problematico (42% zeri), usare con cautela |
| 86684007269889 | 60,940 | 188 | 0 | Due gap grandi, ma no zeri |

**Esclusi** (< 50k record): 86684007269888, 86853106211166, 86684007269869, 86853106210447.

### Formato dati raw

I file si trovano in `json/iot_devices.{imei}.json`. Ogni record è un minuto di misurazione:

```json
{
  "_id": {"$numberLong": "1763018881"},   // epoch Unix (secondi)
  "epoch_valid": true,                    // se false, scartare
  "w_medio": 50,                          // watt medi → metrica principale
  "w_massimo": 50,
  "w_minimo": 50,
  "granularity": 61                       // durata intervallo secondi
}
```

### Inventario dispositivi

Per ogni IMEI: `device_usage_by_imei/{imei}.json`. Campi rilevanti:
- `present`: bool — dispositivo presente in casa
- `frequency`: frequenza d'uso dichiarata (es. "quasi ogni giorno", "1-2 volte")
- `duration`: durata tipica (es. "1-2 ore", ">2 ore")

Categorie dispositivi possibili (16 totali):
`Frigorifero`, `Congelatore`, `Lavatrice`, `Asciugatrice`, `Lavastoviglie`,
`Forno`, `Piano cottura`, `Microonde`, `Boiler`, `Climatizzatore`,
`Televisori`, `Computer`, `Console`, `Auto elettrica`,
`Frigorifero principale`, `Frigorifero secondario`

---

## Goals

- Implementare una **pipeline di preprocessing** condivisa tra tutti gli approcci
- Implementare **5 approcci SOTA** per disaggregazione non supervisionata
- Integrare l'**inventario dispositivi** come prior (quali device cercare e con che peso)
- Creare un **benchmark proxy** oggettivo per confrontare gli approcci
- Produrre **output completi** (grafici, CSV, report energetico) per ogni casa e ogni approccio

---

## User Stories

### US-001: Pipeline di preprocessing

**Description:** Come sviluppatore, voglio una pipeline che carichi, pulisca e normalizzi i
dati raw per ogni IMEI, così che tutti gli approcci partano dallo stesso input coerente.

**Acceptance Criteria:**

- [ ] Carica `json/iot_devices.{imei}.json` e filtra record con `epoch_valid=false`
- [ ] Rimuove record con `w_medio > 10000 W` (spike hardware)
- [ ] Converte epoch Unix → DatetimeIndex pandas con timezone UTC
- [ ] Ricampiona a griglia regolare a 1 minuto (reindex), riempie buchi con `NaN`
- [ ] Espone funzione `load_imei(imei: str) -> pd.Series` che restituisce serie `w_medio`
- [ ] Espone funzione `get_usable_imeis() -> list[str]` con i 6 IMEI usabili
- [ ] Il modulo è importabile da tutti i successivi script di disaggregazione
- [ ] Nessuna dipendenza oltre `pandas`, `numpy`, `json` (già in requirements)

---

### US-002: Inventario dispositivi e knowledge base di potenze

**Description:** Come sviluppatore, voglio un modulo che carichi l'inventario per IMEI e
fornisca la lista dei dispositivi presenti con le loro potenze tipiche note, così che tutti
gli approcci possano usare questa informazione.

**Acceptance Criteria:**

- [ ] Carica `device_usage_by_imei/{imei}.json` e restituisce lista di device presenti
- [ ] Mappa ogni categoria di device a una tupla `(P_min_W, P_typical_W, P_max_W,
      duration_min_min, duration_typical_min, duty_cycle)` usando la seguente knowledge base:

  | Device | P_min | P_typ | P_max | dur_min | dur_typ | duty_cycle |
  |--------|-------|-------|-------|---------|---------|------------|
  | Frigorifero | 50 | 150 | 300 | 5 | 20 | 0.30 |
  | Congelatore | 30 | 100 | 200 | 5 | 20 | 0.25 |
  | Lavatrice | 300 | 1800 | 2500 | 60 | 90 | — |
  | Asciugatrice | 1500 | 2500 | 3000 | 45 | 75 | — |
  | Lavastoviglie | 800 | 1500 | 2400 | 90 | 120 | — |
  | Forno | 1500 | 2500 | 3500 | 20 | 45 | — |
  | Piano cottura | 500 | 1500 | 3000 | 10 | 30 | — |
  | Microonde | 600 | 900 | 1200 | 2 | 5 | — |
  | Boiler | 1000 | 2000 | 3000 | 20 | 45 | — |
  | Climatizzatore | 800 | 1500 | 3500 | 30 | 90 | — |
  | Televisori | 30 | 100 | 200 | 30 | 120 | — |
  | Computer | 50 | 150 | 400 | 30 | 180 | — |
  | Console | 80 | 150 | 220 | 30 | 120 | — |
  | Auto elettrica | 3000 | 3700 | 7400 | 120 | 360 | — |

- [ ] `frequency` e `duration` dal JSON vengono parsati e mappati a valori numerici (es.
      "1-2 ore" → `[60, 120]` minuti, "quasi ogni giorno" → `freq=0.9`)
- [ ] Espone `get_device_profile(imei: str) -> list[DeviceProfile]` dove `DeviceProfile`
      è un dataclass con tutti i campi sopra
- [ ] Il modulo è importabile come `from nilm.devices import get_device_profile`

---

### US-003: Approccio A — Event-based disaggregation

**Description:** Come ricercatore, voglio rilevare le transizioni ON/OFF nel segnale aggregato
tramite variazioni brusche di `w_medio`, così da segmentare il segnale in eventi e attribuirli
a device con potenza compatibile.

**Acceptance Criteria:**

- [ ] Calcola la derivata discreta del segnale: `delta_w[t] = w[t] - w[t-1]`
- [ ] Rileva eventi positivi (accensione) e negativi (spegnimento) su `|delta_w| > soglia`
  (soglia default = 50 W, configurabile)
- [ ] Raggruppa coppie accensione/spegnimento in "blocchi dispositivo"
- [ ] Per ogni blocco, seleziona il device presente in casa la cui `P_typical` è più vicina
  a `|delta_w|` dell'evento
- [ ] Gestisce eventi sovrapposti (più device contemporanei)
- [ ] Restituisce `pd.DataFrame` con colonne `[timestamp, device_name, power_W, state]`
- [ ] Salvato come `scripts/nilm/approach_event_based.py`

---

### US-004: Approccio B — HMM per singolo dispositivo

**Description:** Come ricercatore, voglio modellare ogni dispositivo con un HMM a 2 stati
(ON/OFF) fittato in modo non supervisionato con Baum-Welch, così da stimare la probabilità
che ogni dispositivo sia acceso in ogni istante.

**Acceptance Criteria:**

- [ ] Usa `hmmlearn` (da aggiungere a requirements.txt) con `GaussianHMM(n_components=2)`
- [ ] Per ogni device presente in casa, crea un HMM separato con parametri iniziali basati
  sulla knowledge base (es. stato ON → emissione gaussiana centrata su `P_typical`)
- [ ] Esegue Baum-Welch sul segnale residuo (segnale totale - contributo device già stimati)
- [ ] Ordina gli stati per emissione media crescente (stato 0 = OFF, stato 1 = ON)
- [ ] Restituisce per ogni device una serie binaria `x_i(t) ∈ {0, 1}`
- [ ] Stima potenza disaggregata: `p_i(t) = x_i(t) * P_typical_i`
- [ ] Salvato come `scripts/nilm/approach_hmm.py`

---

### US-005: Approccio C — Factorial HMM (FHMM)

**Description:** Come ricercatore, voglio un Factorial HMM che modelli tutti i dispositivi
simultaneamente con variabili latenti indipendenti, così da catturare combinazioni di accensioni
contemporanee.

**Acceptance Criteria:**

- [ ] Implementa FHMM con inferenza approssimata (mean-field variational o structured
  variational EM) — usare `pyfhmm` o implementazione manuale
- [ ] Ogni dispositivo è una catena di Markov con 2 stati (ON/OFF)
- [ ] Parametri iniziali fissati dalla knowledge base (`P_typical` per stato ON,
  `P_standby ≈ 5W` per stato OFF)
- [ ] Emissione aggregata: `w(t) ~ N(Σ_i p_i · x_i(t), σ²)` con `σ² = 100` (configurabile)
- [ ] EM viene eseguito fino a convergenza (max 100 iterazioni, toll=1e-4)
- [ ] Se `pyfhmm` non disponibile, implementare FHMM semplificato con Viterbi approssimato
- [ ] Restituisce stessa struttura output degli altri approcci
- [ ] Salvato come `scripts/nilm/approach_fhmm.py`

---

### US-006: Approccio D — Template matching

**Description:** Come ricercatore, voglio identificare i dispositivi cercando nel segnale
pattern temporali compatibili con la firma energetica nota (potenza e durata tipica) di ogni
device, senza alcun fitting.

**Acceptance Criteria:**

- [ ] Per ogni device presente in casa, definisce un template: blocco di durata `dur_typical`
  minuti a potenza `P_typical` W
- [ ] Applica cross-correlazione normalizzata tra il segnale `w_medio` e il template di ogni
  device
- [ ] Marca come "accensione rilevata" i picchi di correlazione superiori a una soglia (0.7,
  configurabile)
- [ ] Rimuove overlap: se due device si sovrappongono, tiene quello con correlazione maggiore
- [ ] Non usa nessun fitting parametrico — funziona puramente sulla conoscenza a priori
- [ ] Salvato come `scripts/nilm/approach_template.py`

---

### US-007: Approccio E — Event-based + device prior pesato

**Description:** Come ricercatore, voglio combinare il rilevamento di eventi con un prior
derivato dall'inventario dichiarato dall'utente, così da aumentare la probabilità dei device
noti e ridurre falsi positivi per device assenti.

**Acceptance Criteria:**

- [ ] Parte dalla lista di eventi rilevati (come approccio A)
- [ ] Per ogni evento `delta_w`, calcola uno score per ogni device:
      `score = P(device | delta_w) ∝ likelihood(delta_w | P_typical) * prior_weight`
- [ ] `likelihood(delta_w | P_typical) = N(delta_w; P_typical, sigma²)`
  con `sigma = 0.3 * P_typical`
- [ ] `prior_weight` per device presente = 1.0, per device assente = 0.05
  (configurable, implementa il 5C: device assente non viene eliminato ma de-ponderato)
- [ ] Frequency/duration dalla survey vengono usati per modulare ulteriormente il prior:
  "quasi ogni giorno" → moltiplicatore 1.2, "1-2 volte" → 0.8
- [ ] Assegna a ogni evento il device con score massimo
- [ ] Salvato come `scripts/nilm/approach_event_prior.py`

---

### US-008: Modulo di output comune

**Description:** Come ricercatore, voglio un modulo che prenda i risultati di qualunque
approccio e produca grafici, CSV e report energetico in formato uniforme.

**Acceptance Criteria:**

- [ ] Input: dict `{device_name: pd.Series(power_W, index=DatetimeIndex)}`
  + segnale originale `pd.Series`
- [ ] **Grafico temporale**: segnale aggregato originale (grigio) + strati colorati per device
  (uno colore per device), salvato in `analysis/{imei}/{approach}/daily_plots/`
- [ ] **Report energetico** (barre): kWh per device su tutto il periodo, con percentuale sul
  totale misurato, salvato come PNG in `analysis/{imei}/{approach}/energy_report.png`
- [ ] **CSV**: `analysis/{imei}/{approach}/disaggregation.csv` con colonne
  `[timestamp, w_total, {device_1}, {device_2}, ..., residuo]`
- [ ] **Residuo** = `w_total - Σ device_i` — colonna esplicita nel CSV e nel grafico
- [ ] I grafici temporali sono giornalieri (un file per giorno) — asse x = ore del giorno
- [ ] Salvato come `scripts/nilm/output.py`

---

### US-009: Benchmark proxy senza labels

**Description:** Come ricercatore, voglio metriche quantitative per confrontare gli approcci
senza disporre di ground truth, così da avere un ranking oggettivo.

**Acceptance Criteria:**

- [ ] Calcola e salva in `analysis/benchmark_results.csv` le seguenti metriche per ogni
  (imei, approccio):

  | Metrica | Formula | Ideale |
  |---------|---------|--------|
  | MAE_recon | `mean(|w_total - Σ p_i|)` | 0 |
  | RMSE_recon | `sqrt(mean((w_total - Σ p_i)²))` | 0 |
  | Energy_error_% | `|E_total - Σ E_i| / E_total * 100` | 0 |
  | Residuo_medio_W | `mean(residuo)` | 0 |
  | N_devices_found | conta device con >1% energia | — |
  | Temporal_consistency | % blocchi ON compatibili con dur_typical ± 50% | alto |

- [ ] Genera `analysis/benchmark_summary.png` con tabella heatmap (approccio × metrica)
- [ ] Stampa a terminale un ranking ordinato per `MAE_recon`
- [ ] Salvato come `scripts/nilm/benchmark.py`

---

### US-010: Script main di orchestrazione

**Description:** Come utente, voglio un singolo script che esegua tutta la pipeline
(preprocessing → tutti gli approcci → output → benchmark) per tutti gli IMEI usabili,
così da poter riprodurre l'esperimento con un solo comando.

**Acceptance Criteria:**

- [ ] `scripts/run_disaggregation.py` — eseguibile con `python3 scripts/run_disaggregation.py`
- [ ] Flag CLI opzionali:
  - `--imei IMEI` — processa solo un IMEI specifico
  - `--approach {event,hmm,fhmm,template,event_prior,all}` — default `all`
  - `--no-plots` — salta generazione grafici (più veloce)
- [ ] Crea automaticamente tutte le directory di output necessarie
- [ ] Stampa progress per ogni IMEI e ogni approccio
- [ ] Salva log in `analysis/run_log.txt` con timestamp e eventuali errori
- [ ] Alla fine stampa il benchmark summary

---

## Functional Requirements

- **FR-1**: Il preprocessing rimuove spike (>10,000 W) e record con `epoch_valid=false` prima
  di passare i dati a qualsiasi approccio.
- **FR-2**: Tutti gli approcci operano sullo stesso segnale preprocessato — nessun approccio
  fa il suo preprocessing interno.
- **FR-3**: Il modulo dispositivi (`US-002`) è l'unica sorgente di verità su quali device
  cercare per ogni IMEI — nessun approccio hardcoda nomi di device.
- **FR-4**: Ogni approccio restituisce un dict `{device_name: pd.Series}` con la stessa
  interfaccia — il modulo output (`US-008`) è indipendente dall'approccio.
- **FR-5**: I 6 IMEI con >50k record vengono processati; gli altri 4 sono ignorati
  automaticamente.
- **FR-6**: Nessun approccio richiede dataset di training esterno (UK-DALE, REDD, ecc.).
  I parametri iniziali vengono dalla knowledge base hardcodata in `US-002`.
- **FR-7**: L'approccio FHMM usa parametri iniziali dalla knowledge base — non parte da zero.
- **FR-8**: Il prior dei device (`US-007`) abbassa ma non azzera la probabilità dei device
  assenti (prior_weight = 0.05, non 0) per robustezza.
- **FR-9**: Tutti i grafici usano datetime italiano (asse x leggibile, formato `HH:MM`).
- **FR-10**: Il residuo (non disaggregato) è sempre visibile e quantificato nell'output.

---

## Struttura file finale

```
scripts/
  nilm/
    __init__.py
    preprocessing.py      # US-001
    devices.py            # US-002
    approach_event_based.py  # US-003
    approach_hmm.py          # US-004
    approach_fhmm.py         # US-005
    approach_template.py     # US-006
    approach_event_prior.py  # US-007
    output.py                # US-008
    benchmark.py             # US-009
  run_disaggregation.py      # US-010

analysis/
  {imei}/
    {approach}/
      daily_plots/        # grafici giornalieri
      energy_report.png   # torta/barre energia per device
      disaggregation.csv  # serie temporale disaggregata
  benchmark_results.csv   # tabella metriche proxy
  benchmark_summary.png   # heatmap comparativa
  run_log.txt
```

---

## Non-Goals (Out of Scope)

- Nessun training su dataset esterni (UK-DALE, REDD, NILMTK training pipeline)
- Nessun deep learning (seq2point, seq2seq, trasformer-based)
- Nessuna interfaccia web o dashboard interattiva
- Nessuna gestione real-time — solo analisi batch offline
- Nessuna previsione futura del consumo
- Non vengono processati gli IMEI con <50k record (4 IMEI esclusi)

---

## Technical Considerations

- **Dipendenze nuove**: `hmmlearn>=0.3`, `scipy` (già probabile), eventualmente `pyfhmm`
  o implementazione manuale FHMM se `pyfhmm` non installabile
- **Memoria**: FHMM su 120k campioni con 10 device può essere pesante — considerare di
  lavorare su finestre di 1 settimana se necessario
- **NaN handling**: i buchi nei dati vengono lasciati come `NaN` — gli approcci devono
  skipparli o interpolarli localmente (interpolazione lineare su gap <60 min)
- **Modularità**: ogni approccio è in un file separato con interfaccia comune
  `run(signal: pd.Series, devices: list[DeviceProfile]) -> dict[str, pd.Series]`
- **Riproducibilità**: `random_state=42` ovunque ci siano inizializzazioni randomiche

---

## Success Metrics

- Tutti e 5 gli approcci producono output senza errori per tutti i 6 IMEI usabili
- `MAE_recon < 100 W` per almeno 3 approcci su 6 IMEI (indicativo di buona ricostruzione)
- Il residuo medio è < 20% del consumo totale medio
- I grafici giornalieri mostrano pattern riconoscibili (es. cicli del frigo, lavaggio)
- Il benchmark identifica un approccio "vincitore" su ogni metrica proxy

---

## Open Questions

- `pyfhmm` è installabile in questo ambiente sandbox? Se no, implementare FHMM manualmente
  o usare una approssimazione mean-field semplificata.
- Per IMEI con molti zeri (86853106211173: 42% zeri), trattare gli zeri come "casa vuota"
  o come missing data? Da decidere in fase di implementazione preprocessing.
- La knowledge base di potenze è quella standard dalla letteratura — potrebbe non rispecchiare
  gli elettrodomestici italiani reali. Parametri facilmente modificabili in `devices.py`.
