# Approcci Gurobi — Formulazioni e Differenze

Quattro approcci, tutti MIQP a 15 minuti (configurabili) con 3 livelli di potenza per device, penalità di attivazione e finestre temporali soft. Differiscono per il trattamento dei device always-on e dei vincoli di durata.

---

## Pipeline comune

1. **Baseline always-on** — i device `always_on=True` (frigo, congelatore) vengono stimati con `estimate_always_on_baseline()` e sottratti dal segnale aggregato *(eccezione: `gurobi_full` li modella come variabili, vedi sotto)*.
2. **Resampling** — il segnale 1-minuto viene ricampionato a `granularity_min` (default 15 min) con il metodo scelto (mean/max/min/median).
3. **Ottimizzazione day-by-day** — per ogni giorno si costruisce e risolve un modello Gurobi su quel giorno.
4. **Ricostruzione 1-min** — il risultato a granularità coarse viene forward-filled al timestamp originale 1-minuto.

**Notazione**

| Simbolo | Significato |
|---------|-------------|
| `i` | indice device (event device) |
| `j` | indice device always-on |
| `t` | indice timestep (slot da `granularity_min` minuti) |
| `k` | indice livello di potenza (0=low, 1=mid, 2=high) |
| `T` | numero di slot nel giorno |
| `p_i` | `p_typical_w` del device `i` |
| `y[t]` | segnale al timestep `t` |
| `a_i` | slot minimi ON (`dur_min_min / granularity_min`, arrotondato su) |
| `b_i` | slot massimi consecutivi ON (`dur_typical_min × 1.5 / granularity_min`; in `gurobi_soft_duration` da `duration_minutes_max` se presente) |
| `λ_w` | `time_window_penalty` |
| `λ_a` | `activation_penalty` |
| `λ_d` | `duration_penalty_block` |
| `λ_D` | `duration_penalty_daily` |
| `λ_q` | penalità di eccedenza sulla quota settimanale |
| `r_i` | accensioni ancora disponibili nella settimana |
| `f_i(t)` | fattore di distanza dalla finestra (vedi sotto) |
| `δ` | `power_level_variation` (default 0.15) |

---

## gurobi_activation

**File**: `approach_gurobi_activation.py` → `gurobi_methods.solve_activation`
**Tipo**: MIQP
**Granularità**: 15 min (configurabile)
**Always-on**: stimati con la baseline e sottratti dal segnale — il solver vede solo il residuo.

### Livelli di potenza

Ogni device opera a uno di tre livelli invece che binary ON/OFF:

```
p_i,0 = p_i · (1 − δ)   (livello low)
p_i,1 = p_i              (livello mid)
p_i,2 = p_i · (1 + δ)   (livello high)
```

### Variabili

```
z[i,t,k] ∈ {0,1}   — device i al timestep t è nel livello k
x[i,t]   ∈ {0,1}   — indicatore ON aggregato = Σ_k z[i,t,k]
u[i,t]   ∈ {0,1}   — rising edge (device accende a t)
dw[i,t]  ∈ {0,1}   — falling edge (device spegne a t)
```

**Vincolo di link** (al più un livello attivo):
```
x[i,t]  =  Σ_{k=0}^{2}  z[i,t,k]    ∀i,t
```

Poiché `x ∈ {0,1}` e le `z ∈ {0,1}`, questo garantisce automaticamente al più un livello attivo.

### Vincoli di transizione

```
u[i,0]  = x[i,0]                                   (init rising edge)
dw[i,0] = 0                                        (init falling edge)
x[i,t] − x[i,t−1]  =  u[i,t] − dw[i,t]   ∀t ≥ 1   (transizione)
u[i,t] + dw[i,t]   ≤  1                  ∀t ≥ 1   (mutex)
```

### Durata minima ON

```
Σ_{τ=t}^{t+a_i−1}  x[i,τ]  ≥  a_i · u[i,t]    ∀t ∈ [0, T−a_i]
```

### Durata massima ON consecutiva

```
Σ_{τ=t}^{t+b_i}  x[i,τ]  ≤  b_i    ∀t ∈ [0, T−b_i−1]
```

### Obiettivo

```
min  Σ_t ( Σ_i Σ_k p_i,k · z[i,t,k] − y[t] )²
   + Σ_i  Σ_t  λ_w · p_i² · f_i(t) · x[i,t]     (finestre temporali soft, graduate)
   + Σ_i  Σ_t  λ_a · p_i²         · u[i,t]     (penalità di attivazione)
```

Entrambe le penalità sono scalate per `p_i²` così da essere commensurabili con l'errore quadratico di ricostruzione:

- con `λ_w = 1`, un'attivazione sul bordo della finestra costa `p_i²`, equivalente a un errore di ricostruzione pari a `p_i` per uno slot;
- con `λ_a = 1`, ogni accensione costa quanto avere il device acceso per uno slot con segnale zero. Valori alti scoraggiano cicli ON/OFF rapidi.

### Fattore di distanza dalla finestra

La penalità di finestra non è piatta: cresce con la distanza dalla fascia dichiarata nel questionario, così un'attivazione poco fuori orario resta plausibile mentre una a ore di distanza diventa proibitiva.

```
d_i(t) = distanza circolare in minuti dallo slot t alla finestra più vicina di i

f_i(t) =  0                                        se t è dentro la finestra
       =  min(max_factor, 1 + d_i(t) / ramp_min)   altrimenti
```

Con i default (`ramp_min = 60`, `max_factor = 6`): sul bordo si paga la penalità base, un'ora fuori il doppio, e il costo satura a 6× invece di divergere. La distanza è circolare, quindi le finestre che scavalcano la mezzanotte (es. 23:00–02:00) sono gestite correttamente.

**Nota di taratura**: nel modello L1 (HiGHS) la penalità vale `λ_w · p_i`, quindi con `λ_w = 1` un'attivazione fuori finestra costa esattamente quanto lasciare quel watt non spiegato — la finestra diventa di fatto un vincolo hard. Per restare nella zona in cui la gradazione discrimina davvero, usare `λ_w ≈ 0.2–0.4`.

L'output restituisce la potenza effettiva stimata (`Σ_k p_i,k · z[i,t,k]`), non solo `p_i`.

---

## gurobi_full

**File**: `approach_gurobi_full.py` → `gurobi_methods.solve_full`
**Tipo**: MIQP
**Granularità**: 15 min (configurabile)
**Always-on**: modellati come variabili, nessuna baseline sottratta.

Estende `gurobi_activation`. Gli event device mantengono formulazione, vincoli e penalità identici. La differenza è come vengono trattati i device always-on.

### Cambiato: device always-on come variabili (non baseline)

In `gurobi_activation` i device always-on (frigo, congelatore) vengono stimati con la baseline e sottratti dal segnale — il solver li ignora. In `gurobi_full` entrano nel modello come variabili a 3 livelli:

```
z_ao[j,t,k] ∈ {0,1}   — livello k del device always-on j al timestep t
```

con il vincolo che *esattamente* un livello sia attivo ad ogni timestep valido:

```
Σ_{k=0}^{2}  z_ao[j,t,k]  =  1    ∀j, ∀t valido
```

(nessuna variabile `u`, `dw`, nessun vincolo di transizione/durata/finestra per gli always-on).

### Obiettivo

```
min  Σ_t ( Σ_j Σ_k p_j,k^ao · z_ao[j,t,k]  +  Σ_i Σ_k p_i,k · z[i,t,k]  −  y_raw[t] )²
   + penalità finestre e attivazione sugli event device (come solve_activation)
```

Il segnale `y_raw` qui è il segnale aggregato raw (non il residuo), perché la baseline non viene più sottratta: il solver ottimizza congiuntamente la scelta dei livelli always-on e le attivazioni degli event device.

---

## gurobi_soft_duration

**File**: `approach_gurobi_soft_duration.py` → `gurobi_methods.solve_full` con `duration_penalty_block > 0`
**Tipo**: MIQP
**Granularità**: 15 min (configurabile)
**Always-on**: modellati come variabili (come `gurobi_full`)

Stesso modello di `gurobi_full`, ma i limiti di durata **non sono più vincoli hard**.

### Perché

Le durate del questionario sono approssimative, e un tetto rigido fa rifiutare al solver un device la cui potenza corrisponde esattamente a un plateau, solo perché il plateau dura poco più del massimo dichiarato. Con lavatrice a `max = 2h` e un plateau di 2h15, il modello hard copre 8 slot su 9 e lascia 15 minuti inspiegati. Qui lo stesso blocco è ammesso pagando una penalità proporzionale allo sforamento: il questionario torna a essere un prior, non un veto.

### Cambiato: vincoli di durata → penalità

I due vincoli mantengono la stessa struttura di righe, con l'aggiunta di una slack non negativa:

```
Σ_{τ=t}^{t+a_i−1} x[i,τ]  +  s⁻[i,t]  ≥  a_i · u[i,t]      s⁻ ≥ 0
Σ_{τ=t}^{t+b_i}   x[i,τ]  −  s⁺[i,t]  ≤  b_i               s⁺ ≥ 0
```

con il termine di costo `λ_d · p_i² · (s⁻[i,t] + s⁺[i,t])`.

**Il costo risulta lineare nello sforamento senza doverlo imporre.** Un blocco di lunghezza `L > b_i` contiene esattamente `L − b_i` finestre di ampiezza `b_i+1` interamente al suo interno, ognuna delle quali richiede `s⁺ ≥ 1`: il costo totale è quindi `λ_d · p_i² · (L − b_i)`. Simmetricamente un blocco più corto di `a_i` paga `λ_d · p_i² · (a_i − L)`.

Costo computazionale: 2 variabili **continue** per `(device, slot)`, nessuna binaria aggiuntiva — la difficoltà del MIQP resta praticamente invariata.

### Aggiunto: budget giornaliero

Penalizza lo scostamento del monte ore giornaliero dall'attesa del questionario:

```
E_i  =  (frequency_per_week / 7) · dur_typical_min / granularity_min

Σ_t x[i,t]  −  E_i  =  d⁺[i] − d⁻[i]        d⁺, d⁻ ≥ 0
```

con costo `λ_D · p_i² · (d⁺[i] + d⁻[i])`. Sono 2 variabili e 1 riga per device. A differenza della penalità per-blocco è **globale**: non distingue un blocco lungo da tanti corti, quindi i due termini sono complementari.

### Taratura

Per entrambe le penalità il **break-even è a λ = 1.0**, come per `λ_a`: uno slot di sforamento costa `p_i²`, esattamente quanto lasciare `p_i` watt non spiegati per uno slot.

- `λ < 1` → la durata è elastica, il segnale vince sul questionario
- `λ ≥ 1` → i limiti del questionario vincono

Default degli approcci: `λ_d = λ_D = 0.5`. La variante `gurobi_soft_duration_block` disattiva il budget giornaliero (`λ_D = 0`) tenendo solo la penalità per-blocco.

Il limite massimo `b_i` viene preso da `duration_minutes_max` del questionario quando disponibile (invece di `dur_typical_min × 1.5`), dato che ora è un bound soft e ha senso usare il valore dichiarato.

---

## gurobi_weekly_quota

**File**: `approach_gurobi_weekly_quota.py` → `gurobi_methods.solve_full` con `over_activation_penalty_by_device`
**Tipo**: MIQP
**Granularità**: 15 min (configurabile)
**Always-on**: modellati come variabili

Estende `gurobi_soft_duration` usando la frequenza settimanale del questionario come **budget scorrevole** invece che come prior giornaliero: se il questionario dice che la lavatrice si usa 2 volte a settimana e il solver l'ha già accesa 2 volte, la terza accensione costa molto di più.

### Il contatore vive fuori dal modello

Ogni giorno è un modello indipendente, quindi il conteggio settimanale **non può essere una variabile interna**: è stato che attraversa i solve.

```
1. i giorni si processano in ordine cronologico
2. il contatore si azzera quando cambia la settimana ISO (cioè ogni lunedì)
3. la quota residua r_i entra nel modello del giorno d come COSTANTE
4. dopo il solve, le accensioni realizzate si contano dalla soluzione
   (fronti di salita) e si sommano al contatore
```

### Dentro il modello: solo l'eccedenza

Serve una sola variabile continua per device:

```
Σ_t u[i,t]  −  e_i  ≤  r_i          e_i ≥ 0
```

con costo `λ_q,i · p_i² · e_i`. Le prime `r_i` accensioni della settimana pagano solo `λ_a`; ognuna oltre la quota paga anche il sovrapprezzo. Se `r_i = 0` (quota esaurita) ogni accensione del giorno paga il sovrapprezzo.

### Taratura: il break-even dipende dal device

A differenza delle altre penalità **il break-even non è a λ = 1**. Spiegare un'attivazione di un device che resta acceso `s_i` slot fa guadagnare circa `s_i · p_i²`, quindi il sovrapprezzo deve superare quel valore. Verificato sperimentalmente: con un'attivazione da 4 slot il solver cambia comportamento esattamente a `λ_q = 4`.

Per questo il sovrapprezzo è **per device**:

```
λ_q,i  =  over_activation_factor  ×  s_i          s_i = dur_typical_min / granularity_min
```

Con `over_activation_factor = 1.5` (default) la forza effettiva è la stessa per un ciclo di microonde da 15 minuti e per uno di lavatrice da 2 ore. Un λ scalare sarebbe inevitabilmente mal calibrato tra i due.

### Limite noto: allocazione greedy

Poiché i giorni si risolvono uno alla volta, la quota viene spesa in modo **greedy**: un giorno iniziale la consuma liberamente e i giorni successivi della stessa settimana pagano. Distribuirla in modo ottimo richiederebbe risolvere i sette giorni congiuntamente, molto più costoso. La versione greedy evita comunque il risultato implausibile "lavatrice accesa tutti i giorni", che è lo scopo.

La quota viene da `frequency_per_week_max` del questionario, con fallback su `frequency_per_week`.

---

## Tabella riassuntiva

| Approccio | Solver fn | Tipo | Granularità | Livelli | Finestre | Act. Penalty | Always-on |
|-----------|-----------|------|-------------|---------|----------|--------------|-----------|
| `gurobi_activation` | `solve_activation` | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²·f) | sì (λ·p²·u) | baseline |
| `gurobi_full` | `solve_full` | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²·f) | sì (λ·p²·u) | variabili |
| `gurobi_soft_duration` | `solve_full` (λ_d>0) | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²·f) | sì (λ·p²·u) | variabili |
| `gurobi_weekly_quota` | `solve_full` (λ_q>0) | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²·f) | sì + quota settimanale | variabili |

Nella colonna Finestre, `f` è il fattore di distanza descritto sopra. `gurobi_soft_duration` e `gurobi_weekly_quota` hanno i limiti di durata soft; negli altri due restano vincoli hard. Solo `gurobi_weekly_quota` porta avanti il contatore settimanale.

Tutte le varianti `gurobi_*_max`, `gurobi_*_median`, `gurobi_*_30min` usano la stessa formulazione della variante base con `resample_method` o `granularity_min` diversi.
