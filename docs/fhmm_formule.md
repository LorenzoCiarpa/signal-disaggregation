# FHMM — Formule e Pipeline Completa

## 1. Modello Generativo

Hai **M device**, ognuno con stato binario per ogni timestep:

```
s_m^t ∈ {0, 1}     (0 = OFF, 1 = ON)
```

Il segnale osservato è la somma delle potenze più rumore gaussiano:

```
y^t = Σ_m  s_m^t · μ_m  +  ε^t,     ε^t ~ N(0, σ²)
```

dove `μ_m` è la potenza tipica del device `m` quando acceso.

**Emission probability** — probabilità di osservare `y^t` dato lo stato globale:

```
P(y^t | s^t) = N(y^t ; Σ_m s_m^t · μ_m,  σ²)
```

**Transition probability** — ogni device evolve indipendentemente:

```
P(s_m^t | s_m^{t-1}) = A_m[s_m^{t-1}, s_m^t]
```

dove `A_m` è la matrice di transizione 2×2 del device `m`:

```
A_m = | 1-p_on    p_on  |   (riga OFF →)
      | p_off   1-p_off |   (riga ON  →)
```

con:
- `p_on`  = probabilità di accendersi al prossimo minuto
- `p_off` = probabilità di spegnersi al prossimo minuto ≈ `1 / dur_typical_min`

---

## 2. Il Problema di Inferenza

Vogliamo trovare la sequenza di stati più probabile dato il segnale osservato:

```
s*^{1:T} = argmax  P(s^{1:T} | y^{1:T})
```

Per la regola di Bayes:

```
P(s^{1:T} | y^{1:T})  ∝  P(y^{1:T} | s^{1:T}) · P(s^{1:T})
```

**Problema**: lo spazio degli stati è **2^M** combinazioni per ogni timestep.
Con 10 device e 10.000 minuti → calcolo esatto intrattabile.

---

## 3. Approssimazione: Mean Field Variational Inference

Si approssima la distribuzione congiunta con una distribuzione fattorizzata:

```
Q(s^{1:T}) ≈ Π_m  Q_m(s_m^{1:T})
```

Ogni `Q_m` è un HMM indipendente.
Si minimizza la KL-divergence `KL(Q || P)` → porta al **coordinate ascent**:
si ottimizza un device alla volta tenendo fissi tutti gli altri.

---

## 4. Aggiornamento del Coordinate Ascent (esatto)

Per device `m`, il log dell'aggiornamento ottimale è:

```
log Q_m(s_m^{1:T}) = E_{Q_{-m}}[ log P(y^{1:T}, s^{1:T}) ] + const
```

Il termine di emission "visto" dal device `m` diventa un **segnale residuo effettivo**:

```
ỹ_m^t = y^t - Σ_{k≠m}  <s_k^t> · μ_k
```

dove `<s_k^t> = E_{Q_k}[s_k^t]` è la **probabilità marginale** che device `k` sia ON al tempo `t`.

Con questo residuo si esegue **forward-backward** su device `m` con emission:

```
N(ỹ_m^t ; s_m^t · μ_m,  σ²)
```

---

## 5. Approssimazione Hard: il Codice

Il codice **non esegue forward-backward**. Usa stati hard `s_k^t ∈ {0,1}` invece di marginali soft, trasformando il coordinate ascent variazionale in un **greedy ascent**:

```
ỹ_m^t = y^t - Σ_{k≠m}  s_k^t · μ_k          # residuo hard
```

**Regola di attivazione MAP** con prior piatto e varianza gaussiana uguale per tutti:

```
s_m^t = 1   sse   ỹ_m^t > μ_m / 2
```

**Derivazione della soglia** `μ_m / 2`:

```
P(y^t | s_m=1) > P(y^t | s_m=0)
N(ỹ_m^t ; μ_m, σ²) > N(ỹ_m^t ; 0, σ²)
(ỹ_m^t - μ_m)² < (ỹ_m^t)²
-2μ_m · ỹ_m^t + μ_m² < 0
ỹ_m^t > μ_m / 2   ✓
```

`μ_m / 2` è la **soglia di massima verosimiglianza**: il punto esattamente a metà tra
"device spento → emette 0W" e "device acceso → emette μ_m W".

---

## 6. Integrazione del Questionario: Prior Temporale

Quando il questionario indica una finestra d'uso (es. lavatrice 8:00–12:00), si introduce un **prior non uniforme** sull'attivazione:

```
P(s_m^t = 1) = π_m^t
```

Il MAP corretto con questo prior diventa:

```
s_m^t = 1   sse   ỹ_m^t > μ_m/2 - (σ²/μ_m) · log[ π_m^t / (1 - π_m^t) ]
```

Il secondo termine è la **correzione log-odds** del prior:
- se `π_m^t > 0.5` (device probabile in questa fascia) → log-odds positivo → soglia scende → più facile attivarsi
- se `π_m^t < 0.5` (device improbabile) → soglia sale → difficile attivarsi

**Approssimazione euristica nel codice** — invece della correzione log-odds si moltiplica la soglia per un fattore:

```
threshold^t = (μ_m / 2) · factor_m^t
```

con i valori:

| Condizione | `factor` | Effetto |
|---|---|---|
| Dentro la finestra oraria | 0.85 | Soglia più bassa → più facile ON |
| Fuori dalla finestra oraria | 1.35 | Soglia più alta → difficile ON |
| In stagione (`active_months`) | 0.85 | Più facile ON |
| Fuori stagione | 1.75 | Quasi impossibile ON |

I fattori sono moltiplicativi tra loro: un device fuori finestra e fuori stagione riceve `1.35 × 1.75 = 2.36`.

---

## 7. Struttura Temporale: Durate

In un HMM corretto, i vincoli di durata emergono dalla matrice di transizione tramite Viterbi o forward-backward — la sequenza di stati è ottimizzata globalmente.

Nel codice il coordinate ascent è **memoryless**: la decisione `s_m^t` dipende solo da `ỹ_m^t`, non da `s_m^{t-1}`. La durata è gestita nel **post-processing**, dopo l'inferenza.

### 7.1 Commitment Window (durata minima)

Ogni rising edge `(s_m^{t-1}=0, s_m^t=1)` forza ON per i successivi `dur_typical_min` timestep:

```
commit_end = t + dur_typical_min
s_m^τ = 1    per  t ≤ τ < commit_end
```

Se nel corso della finestra appare un altro rilevamento, la finestra si estende:

```
commit_end = max(commit_end, τ + dur_typical_min)
```

### 7.2 Cap Durata Massima

Ogni blocco ON viene troncato a `duration_minutes_max`:

```
se (end - start) > dur_max:
    s_m^τ = 0    per  start + dur_max ≤ τ < end
```

### 7.3 Limite Blocchi Totali

Dato l'orizzonte osservato in giorni, il numero massimo di attivazioni è:

```
max_blocks = ceil( frequency_per_week_max × observed_days / 7 )
```

Si mantengono solo i blocchi con score (integrale del residuo) più alto.

### 7.4 Cap Giornaliero

Per ogni giorno, se i minuti ON superano `daily_usage_hours_max × 60`, si tengono solo i blocchi con score più alto fino a saturare il budget.

---

## 8. Assegnazione della Potenza

Dopo aver determinato gli stati ON/OFF, si assegna la potenza effettiva:

```
remaining^t = max(0,  y^t - baseline_always_on)
```

I device sono processati in ordine decrescente di `μ_m` (prima i più potenti):

- **Device bursty** (`dur_typical_min ≤ 5 min`, es. microonde): potenza fissa
  ```
  power_m^t = μ_m    se  s_m^t = 1
  ```

- **Device discontinui** (lavatrice, forno, ...): potenza variabile dal residuo rimasto
  ```
  power_m^t = clip(remaining^t, 0, p_max_m)    se  s_m^t = 1
  remaining^t -= power_m^t
  ```

---

## 9. Pipeline Completa

```
y^t  (segnale 1 minuto)
  │
  ├─ [1] BASELINE ALWAYS-ON
  │       sottrai Σ_m p_m per frigo/congelatore
  │       → residual_ca^t
  │
  ├─ [2] COORDINATE ASCENT (max_iter volte, convergenza se Δstates < tol·T)
  │       per ogni device m:
  │         ỹ_m^t = residual_ca^t - Σ_{k≠m} s_k^t · μ_k
  │         threshold^t = (μ_m/2) · factor_survey(m, t)
  │         s_m^t = 1  sse  ỹ_m^t > threshold^t
  │
  ├─ [3] POST-PROCESSING (per device m):
  │       a. commitment_window(s_m, dur_typical_min)     ← durata minima
  │       b. cap_block_duration(s_m, dur_max)            ← durata massima
  │       c. limit_on_blocks(s_m, max_blocks)            ← frequenza settimanale
  │       d. cap_daily_on_minutes(s_m, daily_cap)        ← ore giornaliere
  │
  └─ [4] ASSEGNAZIONE POTENZA
          remaining = max(0, y^t - baseline)
          per device m in ordine di μ_m decrescente:
            se bursty → power_m = μ_m
            altrimenti → power_m = clip(remaining, 0, p_max_m)
            remaining -= power_m
```

---

## 10. Limite Principale

Il post-processing (step 3) avviene **dopo** il coordinate ascent (step 2).
Durante l'inferenza, i device si "vedono" con stati non ancora vincolati dalle durate.
Questo significa che il residuo che un device lascia agli altri non è coerente con i vincoli di durata reali.

**Soluzione**: integrare il vincolo di durata minima dentro il loop di coordinate ascent — applicare la commitment window ad ogni iterazione prima di ricalcolare il residuo per il device successivo.
