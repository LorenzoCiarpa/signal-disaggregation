# Approcci Gurobi — Formulazioni e Differenze

Tutti gli approcci seguono la stessa pipeline di alto livello, poi divergono sulla formulazione del sotto-problema.

---

## Pipeline comune

1. **Baseline always-on** — i device `always_on=True` (frigo, congelatore) vengono stimati con `estimate_always_on_baseline()` e sottratti dal segnale aggregato *(eccezione: `gurobi_full` li modella come variabili, vedi sotto)*.
2. **Resampling** — il residuo 1-minuto viene ricampionato a `granularity_min` (default 15 min) con il metodo scelto (mean/max/min/median) *(eccezione: `gurobi_daywise` usa 1 min)*.
3. **Ottimizzazione day-by-day** — per ogni giorno si costruisce e risolve un modello Gurobi sul residuo di quel giorno.
4. **Ricostruzione 1-min** — il risultato a granularità coarse viene forward-filled al timestamp originale 1-minuto.

**Notazione**

| Simbolo | Significato |
|---------|-------------|
| `i` | indice device (event device) |
| `t` | indice timestep (slot da `granularity_min` minuti) |
| `k` | indice livello di potenza (0=low, 1=mid, 2=high) |
| `T` | numero di slot nel giorno |
| `n` | numero di event device |
| `p_i` | `p_typical_w` del device `i` |
| `y[t]` | segnale residuo al timestep `t` |
| `a_i` | slot minimi ON (`dur_min_min / granularity_min`, arrotondato su) |
| `b_i` | slot massimi consecutivi ON (`dur_typical_min × 1.5 / granularity_min`, arrotondato su) |
| `λ_w` | `time_window_penalty` |
| `λ_a` | `activation_penalty` |
| `δ` | `power_level_variation` (default 0.15) |

---

## gurobi_daywise

**File**: `approach_gurobi_daywise.py` → `solve_unconstrained_window`
**Tipo**: MIQP
**Granularità**: 1 minuto (nessun resampling)
**Note**: versione sperimentale, processa solo i primi 5 giorni.

### Formulazione base (condivisa da tutte le versioni successive)

**Variabili**: `x[i,t] ∈ {0,1}` — device `i` è acceso al timestep `t`

**Obiettivo**:
```
min  Σ_t  ( Σ_i p_i · x[i,t]  −  y[t] )²
```

**Vincoli**: nessuno (problema binario non vincolato).

> Questa formulazione di base viene estesa da tutti gli approcci successivi. Le sezioni seguenti descrivono *solo le aggiunte o modifiche* rispetto alla versione precedente.

---

## gurobi_15min

**File**: `approach_gurobi_15min.py` → `constrained_v3`
**Tipo**: MIQP
**Granularità**: 15 min (configurabile)

### Aggiunto: variabili di transizione

```
u[i,t] ∈ {0,1}   — rising edge (device accende a t)
dw[i,t] ∈ {0,1}  — falling edge (device spegne a t)
```

### Aggiunto: vincoli di transizione

```
u[i,0]  = x[i,0]                           (init rising edge)
dw[i,0] = 0                                (init falling edge)
x[i,t] − x[i,t−1]  =  u[i,t] − dw[i,t]   ∀t ≥ 1   (C3 — transizione)
u[i,t] + dw[i,t]   ≤  1                    ∀t ≥ 1   (C4 — mutex)
```

### Aggiunto: durata minima ON (C5)

```
Σ_{τ=t}^{t+a_i−1}  x[i,τ]  ≥  a_i · u[i,t]    ∀t ∈ [0, T−a_i]
```

### Aggiunto: durata massima ON consecutiva (C6)

```
Σ_{τ=t}^{t+b_i}  x[i,τ]  ≤  b_i    ∀t ∈ [0, T−b_i−1]
```

### Aggiunto: finestre temporali hard (C7 — hard exclusion)

I timestep fuori dalla finestra `[start_window_start_min, start_window_end_min)` vengono esclusi con:
```
x[i,t].UB = 0    ∀t fuori finestra    (e uguale per u[i,t])
```

---

## gurobi_soft

**File**: `approach_gurobi_soft.py` → `constrained_v4`
**Tipo**: MIQP
**Granularità**: 15 min

Estende `gurobi_15min`. Mantiene tutti i vincoli C3–C6. La sola differenza è la gestione delle finestre temporali.

### Cambiato: finestre temporali hard → penalità soft

La hard exclusion (UB=0) viene rimossa. Al suo posto, un termine lineare nell'obiettivo penalizza ogni slot fuori-finestra:

```
min  Σ_t ( Σ_i p_i · x[i,t] − y[t] )²
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i² · x[i,t]
```

Con `λ_w = 1` il costo di un'attivazione fuori-finestra è `p_i²`, equivalente a un errore di ricostruzione pari a `p_i` per uno slot. Valori più alti rendono il deterrente più forte.

---

## gurobi_multistate

**File**: `approach_gurobi_multistate.py` → `constrained_v5`
**Tipo**: MIQP
**Granularità**: 15 min

Estende `gurobi_soft`. Mantiene tutti i vincoli e la penalità soft.

### Cambiato: 3 livelli di potenza per device

Ogni device può operare a uno di tre livelli invece di binary ON/OFF:

```
p_i,0 = p_i · (1 − δ)   (livello low)
p_i,1 = p_i              (livello mid)
p_i,2 = p_i · (1 + δ)   (livello high)
```

**Nuove variabili**:
```
z[i,t,k] ∈ {0,1}   — device i al timestep t è nel livello k
x[i,t]   ∈ {0,1}   — indicatore ON aggregato = Σ_k z[i,t,k]
```

**Vincolo di link** (al più un livello attivo):
```
x[i,t]  =  Σ_{k=0}^{2}  z[i,t,k]    ∀i,t
```

Poiché `x ∈ {0,1}` e le `z ∈ {0,1}`, questo garantisce automaticamente al più un livello attivo.

**Obiettivo aggiornato** (usa i livelli invece di `p_i`):
```
min  Σ_t ( Σ_i Σ_k p_i,k · z[i,t,k] − y[t] )²
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i² · x[i,t]
```

Tutti i vincoli C3–C6 operano invariati su `x[i,t]` (indicatore ON aggregato).

L'output restituisce la potenza effettiva stimata (`Σ_k p_i,k · z[i,t,k]`), non solo `p_i`.

---

## gurobi_activation

**File**: `approach_gurobi_activation.py` → `constrained_v6`
**Tipo**: MIQP
**Granularità**: 15 min
**Note**: contiene un `break` al giorno 5 — processa solo i primi 5 giorni.

Estende `gurobi_multistate`. Aggiunge un termine di penalità sulle accensioni.

### Aggiunto: penalità di attivazione

```
min  Σ_t ( Σ_i Σ_k p_i,k · z[i,t,k] − y[t] )²
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i² · x[i,t]
   + Σ_i  Σ_t               λ_a · p_i² · u[i,t]
```

Con `λ_a = 1`, ogni accensione costa quanto avere il device acceso per uno slot con segnale zero. Valori alti scoraggiano cicli ON/OFF rapidi.

---

## gurobi_full

**File**: `approach_gurobi_full.py` → `constrained_v7`
**Tipo**: MIQP
**Granularità**: 15 min

Estende `gurobi_activation`. La differenza principale è come vengono trattati i device always-on.

### Cambiato: device always-on come variabili (non baseline)

In tutti gli approcci precedenti i device always-on (frigo, congelatore) vengono stimati con la baseline e sottratti dal segnale — il solver li ignora. In `gurobi_full` entrano nel modello come variabili a 3 livelli:

```
z_ao[j,t,k] ∈ {0,1}   — livello k del device always-on j al timestep t
```

con il vincolo che *esattamente* un livello sia attivo ad ogni timestep valido:

```
Σ_{k=0}^{2}  z_ao[j,t,k]  =  1    ∀j, ∀t valido
```

(nessuna variabile `u`, `dw`, nessun vincolo di transizione/durata per gli always-on).

**Obiettivo aggiornato** (somma su always-on E event devices):
```
min  Σ_t ( Σ_j Σ_k p_j,k^ao · z_ao[j,t,k]  +  Σ_i Σ_k p_i,k · z[i,t,k]  −  y_raw[t] )²
   + penalità finestre e attivazione sugli event device (come v6)
```

Nota: il segnale `y` qui è il segnale aggregato raw (non il residuo), perché il baseline non viene più sottratto.

---

## Tabella riassuntiva

| Approccio | Solver fn | Tipo | Granularità | Livelli | Finestre | Act. Penalty | Always-on |
|-----------|-----------|------|-------------|---------|----------|--------------|-----------|
| `gurobi_daywise` | `solve_unconstrained_window` | MIQP | 1 min | 1 (binary) | nessuna | no | baseline |
| `gurobi_15min` | `constrained_v3` | MIQP | 15 min | 1 (binary) | hard (UB=0) | no | baseline |
| `gurobi_soft` | `constrained_v4` | MIQP | 15 min | 1 (binary) | soft (λ·p²) | no | baseline |
| `gurobi_multistate` | `constrained_v5` | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²) | no | baseline |
| `gurobi_activation` | `constrained_v6` | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²) | sì (λ·p²·u) | baseline |
| `gurobi_full` | `constrained_v7` | MIQP | 15 min | 3 (z[i,t,k]) | soft (λ·p²) | sì (λ·p²·u) | variabili |

Tutte le varianti `gurobi_*_max`, `gurobi_*_median`, `gurobi_*_30min` usano la stessa formulazione della variante base con `resample_method` o `granularity_min` diversi.
