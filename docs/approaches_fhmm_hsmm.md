# Approcci FHMM e HSMM — Formulazioni e Differenze

Questa famiglia di approcci usa modelli probabilistici grafici per la disaggregazione, senza solver di ottimizzazione.  Tutti seguono lo schema **coordinate-ascent**: ogni dispositivo viene inferito a turno, sottraendo il contributo stimato degli altri dal segnale.

---

## Notazione comune

| Simbolo | Significato |
|---------|-------------|
| `T` | numero di timestep |
| `n` | numero di event device |
| `i` | indice device |
| `t` | indice timestep |
| `y[t]` | segnale aggregato al timestep t (W) |
| `p_i` | `p_typical_w` del device i (W) |
| `x_i[t] ∈ {0,1}` | stato binario (ON/OFF) del device i |
| `r_i[t]` | residuo per il device i = segnale meno contributo degli altri device |
| `B_ao` | baseline always-on (W, scalare) |

---

## fhmm_survey

**File**: `approach_fhmm_survey.py`
**Granularità**: 1 minuto (nessun resampling)
**Tipo**: Factorial HMM con coordinate-ascent greedy, baseline always-on, soglie survey-aware e post-processing esteso

### Modello

Il FHMM assume che il segnale aggregato sia la sovrapposizione di `n` catene di Markov **binarie e indipendenti**:

```
y[t]  =  Σ_i  p_i · x_i[t]  +  ε[t]
```

con `ε[t]` rumore additivo. La dipendenza temporale (Markov) non è sfruttata esplicitamente nell'ottimizzazione — il coordinate-ascent minimizza direttamente l'errore quadratico istantaneo.

### 1. Baseline always-on e coordinate-ascent sul residuo

I device `always_on=True` non entrano nel coordinate-ascent. Vengono trattati come una **baseline costante**:

```
B_ao  =  Σ_{j ∈ always_on}  b_j

con  b_j  =  p_j · duty_j   (baseline_mode="duty_avg")
          o  p_j             (baseline_mode="peak")
```

Il coordinate-ascent opera sul **residuo clipato a zero**:

```
residual_ca[t]  =  max(0,  y[t] − B_ao)
```

Ad ogni iterazione, per ogni device `i`:

1. Calcola il **residuo escludendo il device i**:
   ```
   r_i[t]  =  residual_ca[t]  −  Σ_{j≠i}  p_j · x_j[t]
   ```

2. Scegli lo stato che minimizza `|r_i[t] − p_i · x_i[t]|`:
   ```
   x_i[t]  =  1   se  r_i[t] > threshold_i[t]
   x_i[t]  =  0   altrimenti
   ```

3. Ripeti finché il numero di cambiamenti di stato è `< tol · T`.

**Complessità per iterazione**: O(T · n) — nessuna ricorsione temporale.

### 2. Threshold modifiers survey-aware

Senza survey il punto di indifferenza tra ON e OFF sarebbe `p_i / 2`:

```
|r_i[t] − p_i|  <  |r_i[t] − 0|
  ⟺   r_i[t] > p_i / 2
```

Il questionario V2 rende la soglia **dipendente dal timestep**:

```
threshold_i[t]  =  (p_i / 2)  ·  φ_i[t]

con  φ_i[t]  =  φ_window(t) · φ_season(t)
```

**Fattore finestra temporale** `φ_window(t)`:
```
φ_window(t)  =  0.85   se t ∈ [start_window, end_window)   → soglia abbassata, più facile attivarsi
             =  1.35   altrimenti                            → soglia alzata, più difficile attivarsi
```

**Fattore stagione** `φ_season(t)`:
```
φ_season(t)  =  0.85   se mese ∈ active_months   → soglia abbassata
             =  1.75   altrimenti                 → soglia alzata
```

Abbassare la soglia equivale a rendere il device "più propenso" ad attivarsi — il questionario introduce un prior soft sull'orario e sulla stagione senza cambiare la struttura del modello.

### 3. Post-processing

**Passo A — rimozione blocchi brevi** (prima del commitment):
```
rimuovi tutti i blocchi ON con durata < dur_min_min
```
Questo filtra il rumore prima di estendere le attivazioni superstiti.

**Passo B — commitment window**:
```
estendi ogni blocco superstite a dur_typical_min campioni
```

Questo previene lo spegnimento prematuro durante le fasi silenziose dei device discontinui (es. pausa tra riscaldamento e risciacquo di una lavatrice): invece di eliminare i picchi brevi, li estende alla durata tipica.

**Passo C — cap durata massima** (opzionale, da `duration_minutes_max`):
```
tronca ogni blocco a max(dur_max) campioni
```

**Passo D — cap numero di attivazioni** (da `frequency_per_week_max`):
```
max_blocks  =  ceil(freq_max_week · observed_days / 7)
se n_blocchi > max_blocks:
    mantieni i max_blocks blocchi con Σ residual_ca più alta
```

**Passo E — cap ore giornaliere** (da `daily_usage_hours_max`):
```
max_min_day  =  daily_usage_hours_max · 60
per ogni giorno:
    se Σ on_minutes > max_min_day:
        mantieni i blocchi in ordine di punteggio residuo
        fino a max_min_day minuti totali
```

### 4. Output potenza variabile

Dopo il post-processing degli stati, la potenza assegnata a ciascun device viene stimata dal **residuo rimanente** (non `p_typical` fisso), in ordine decrescente di `p_typical`:

```
pool[t]  =  max(0,  y[t] − B_ao)

for i in ordine decrescente di p_i:
    if device i è bursty  (dur_typical_min ≤ 5 min):
        assigned_i[t]  =  p_i                    se x_i[t]=1
        pool[t]        =  max(0, pool[t] − p_i)  se x_i[t]=1
    else:
        assigned_i[t]  =  clip(pool[t], 0, p_max_i)   se x_i[t]=1
        pool[t]        =  max(0, pool[t] − assigned_i[t])
```

**Device bursty** (es. microonde, ≤5 min tipici): potenza fissa a `p_typical` perché il residuo istantaneo è poco informativo su scale così brevi.
**Device discontinui** (es. lavatrice): potenza variabile tra 0 e `p_max`, stimata dal pool.

---

## hsmm_survey

**File**: `approach_hsmm_survey.py`
**Granularità**: 15 minuti (configurabile)
**Tipo**: Factorial Hidden Semi-Markov Model con Viterbi e prior survey

### Differenza fondamentale rispetto all'FHMM

Nell'FHMM la durata degli stati è **geometrica** (la transizione OFF→ON ha probabilità fissa ad ogni timestep, indipendente dalla durata del segmento corrente). Questo è il modello di Markov standard a memoria nulla.

L'**HSMM** (Hidden Semi-Markov Model) aggiunge una **distribuzione esplicita sulla durata** di ogni segmento: la probabilità di stare nello stato `s` per esattamente `d` slot è modellata separatamente. Questo permette di dire "la lavatrice dura circa 90 minuti" senza affidarsi a post-processing.

### Modello generativo

Per ogni device (2 stati: OFF=0, ON=1):

**Emissione ON** al slot t con residuo `r[t]`:
```
P(r[t] | stato=ON)  ∝  N(r[t]; p_typical, σ_on²)  ·  exp(log_bias[t])

con  σ_on  =  max(10, 0.35 · p_typical)
     log_bias[t]  =  log(1.5)   se t in finestra oraria   → bonus ON
                  =  log(0.15)  se fuori finestra           → penalità ON
                  +  log(1.3)   se in active_months
                  +  log(0.1)   se fuori stagione
```

**Emissione OFF** al slot t:
```
P(r[t] | stato=OFF)  ∝  N(r[t]; 0, σ_off²)

con  σ_off  =  80.0 W
```

**Distribuzione durata ON** (truncated discrete Gaussian):
```
P(d | ON)  ∝  exp( −(d − d_typ)² / (2 σ_d²) )   per d ∈ [d_min, d_max]

con  d_min  =  ceil(dur_min_min / granularity_min)
     d_typ  =  round(dur_typical_min / granularity_min)
     d_max  =  ceil(duration_minutes_max / granularity_min)
     σ_d    =  max(1, (d_max − d_min) / 4)
```

**Distribuzione durata OFF** (geometrica):
```
P(d | OFF)  =  (1−p_off)^(d−1) · p_off

con  p_off  =  1 / mean_off_slots
     mean_off_slots  =  slots_per_day / activations_per_day  −  d_typ
     activations_per_day  =  frequency_per_week / 7
```

### Viterbi HSMM (segment-based DP)

Il Viterbi per HSMM non lavora slot-per-slot come nell'HMM standard, ma **segmento per segmento**. La DP è definita su:

```
best[t, s]  =  log-probabilità massima di coprire i slot 0..t−1
               con l'ultimo segmento nello stato s
back[t, s]  =  inizio dell'ultimo segmento (per il traceback)
```

**Inizializzazione**: `best[0, 0] = best[0, 1] = 0`

**Transizione ON** (gli stati si alternano: OFF→ON→OFF):
```
per ogni t = 1..T:
    per ogni durata d = d_min..min(d_max, t):
        seg_start = t − d
        emit = Σ_{τ=seg_start}^{t−1} log P(r[τ] | ON)   [O(1) con prefix-sum]
        dur  = log P(d | ON)
        prev = best[seg_start, 0]   (deve venire da un segmento OFF)
        score = prev + emit + dur

        if score > best[t, 1]:
            best[t, 1] = score
            back[t, 1] = seg_start
```

**Transizione OFF** (simmetrica, viene da ON):
```
per ogni t = 1..T:
    per ogni durata d = 1..min(dur_max_off, t):
        seg_start = t − d
        emit = Σ_{τ=seg_start}^{t−1} log P(r[τ] | OFF)
        dur  = (d−1) · log(1−p_off) + log(p_off)
        prev = best[seg_start, 1]   (deve venire da un segmento ON)
        score = prev + emit + dur

        if score > best[t, 0]:
            best[t, 0] = score
            back[t, 0] = seg_start
```

**Complessità**: O(T · (d_max_on + d_max_off)) per device — più costoso dell'FHMM ma esatto.

Il prefix-sum sulle emissioni riduce il calcolo dell'emissione di un segmento da O(d) a O(1):
```
cs_on[t]  =  Σ_{τ=0}^{t−1} log P(r[τ] | ON)
emit di [seg_start..t-1]  =  cs_on[t] − cs_on[seg_start]
```

**Traceback**: partendo da `argmax(best[T])`, si risale alternando gli stati lungo i puntatori `back`.

### Coordinate-ascent con Viterbi

Il Viterbi è single-device. Per gestire `n` device in modo factorial, si usa lo stesso schema coordinate-ascent di `fhmm_survey`:

```
for _iter in range(max_iter):
    for i in range(n_dev):
        others[t] = Σ_{j≠i} p_j · x_j[t]
        dev_residual[t] = max(0, residual_ca[t] − others[t])
        x_i  =  _hsmm_viterbi(obs=dev_residual, ...)
    if cambiamenti < tol · T:
        break
```

### Survey nel modello vs post-processing

Nell'FHMM le informazioni survey (finestre, stagione, durate) entrano **dopo** l'inferenza, come post-processing sugli stati. Nell'HSMM entrano **dentro** l'inferenza:

| Info survey | fhmm_survey | hsmm_survey |
|-------------|--------------|-------------|
| Finestra oraria | threshold modifier (×0.85 / ×1.35) | log-bias sull'emissione ON (principled) |
| Stagione attiva | threshold modifier (×0.85 / ×1.75) | log-bias sull'emissione ON (principled) |
| `dur_min_min` | rimozione blocchi brevi (post) | lower bound distribuzione P(d\|ON) (inside DP) |
| `dur_typical_min` | commitment window (post) | modo distribuzione P(d\|ON) (inside DP) |
| `duration_minutes_max` | cap blocchi (post) | upper bound distribuzione P(d\|ON) (inside DP) |
| `frequency_per_week` | cap n. blocchi (post) | parametro p_off distribuzione geometrica (inside DP) |
| `daily_usage_hours_max` | cap ore/giorno (post) | non usato |

### Output potenza variabile

Identico a `fhmm_survey`: assegnazione dal pool di residuo in ordine decrescente di `p_typical`, con distinzione bursty/non-bursty.

---

## Tabella riassuntiva

| Approccio | Tipo | Granularità | Always-on | Threshold | Durata ON | Finestra/Stagione |
|-----------|------|-------------|-----------|-----------|-----------|-------------------|
| `fhmm_survey` | CA greedy + commitment | 1 min | baseline | `p_i/2 · φ` survey | commitment + cap max + cap freq + cap daily | threshold modifier (soft) |
| `hsmm_survey` | Viterbi HSMM | 15 min | baseline | N/A (Viterbi) | P(d\|ON) trunc. Gaussiana | log-bias emissione (principled) |

### Quando preferire cosa

- **fhmm_survey**: sfrutta tutto il questionario con euristiche veloci; il più economico dei due.
- **hsmm_survey**: modello formalmente più corretto — la distribuzione di durata è un prior Bayesiano, non una regola post-hoc. Più lento (DP per device per iterazione), ma gestisce meglio i device con durate vincolate e pattern intermittenti durante il ciclo.
