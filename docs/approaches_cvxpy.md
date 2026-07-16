# Approcci liberi (cvxpy_*) — Formulazioni e Differenze

Questi approcci usano solver open-source (HiGHS e SCIP) senza licenza Gurobi.
Il nome `cvxpy_*` è storico: l'implementazione attuale non usa più CVXPY.

---

## Differenze strutturali rispetto agli approcci Gurobi

Tutti e tre gli approcci mantengono la stessa **pipeline** degli approcci Gurobi (baseline, resampling, day-by-day, forward-fill). Quello che cambia è il modello di ottimizzazione.

| Proprietà | Gurobi | cvxpy / cvxpy_full | cvxpy_activation |
|-----------|--------|-------------------|-----------------|
| Solver | Gurobi (licenza) | HiGHS (free) | SCIP via pyscipopt (free) |
| Tipo problema | MIQP | MILP | MINLP (vincoli quadratici convessi) |
| Obiettivo | L2: `min Σ (error)²` | L1: `min Σ \|error\|` | L2: `min Σ (error)²` (epigraph) |
| Livelli potenza | 3 (ove applicabile) | 1 (binary ON/OFF) | 3 (come Gurobi) |
| Penalità scala | `p_i²` | `p_i` | `p_i²` |

---

## Notazione

Stessa notazione del documento Gurobi, più:

| Simbolo | Significato |
|---------|-------------|
| `ep[t]` | parte positiva del residuo al timestep `t` (≥ 0) |
| `en[t]` | parte negativa del residuo al timestep `t` (≥ 0) |
| `r[t]` | residuo = `Σ_i p_i x[i,t] − y[t]` |
| `q[t]` | variabile epigrafo: `q[t] ≥ r[t]²` (≥ 0) |

---

## cvxpy

**File**: `approach_cvxpy.py` → `constrained_highs` in `highs_methods.py`
**Solver**: HiGHS (via `highspy`)
**Tipo**: MILP
**Granularità**: 15 min
**Equivalente Gurobi**: `gurobi_soft` (stessa struttura di vincoli, diverso obiettivo)

### Formulazione

**Variabili**:
```
x[i,t] ∈ {0,1}     — device i acceso al timestep t
u[i,t] ∈ {0,1}     — rising edge
dw[i,t] ∈ {0,1}    — falling edge
ep[t]  ≥ 0         — errore positivo (lineare)
en[t]  ≥ 0         — errore negativo (lineare)
```

**Obiettivo L1**:
```
min  Σ_t  ( ep[t] + en[t] )
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i · x[i,t]
```

La penalità finestra è scalata per `p_i` (non `p_i²` come in Gurobi) per mantenere le grandezze omogenee con l'obiettivo L1.

**Vincolo di ricostruzione** (linearizzazione del valore assoluto):
```
Σ_i p_i · x[i,t]  −  ep[t]  +  en[t]  =  y[t]    ∀t
```

Poiché `ep[t], en[t] ≥ 0`, minimizzare `ep[t] + en[t]` equivale a minimizzare `|Σ_i p_i x[i,t] − y[t]|`.

**Vincoli di transizione, durata min/max**: identici a `constrained_v4` (vedi documento Gurobi).

### Perché HiGHS e non SCIP?

L'obiettivo L1 non ha termini quadratici → problema MILP puro. HiGHS è specializzato in LP/MILP e risolve questi problemi in pochi secondi (1-2 s per giorno, ~7 device).

---

## cvxpy_activation

**File**: `approach_cvxpy_activation.py` → `constrained_highs_multistate` in `highs_methods.py`
**Solver**: HiGHS (via `highspy`)
**Tipo**: MILP
**Granularità**: 15 min
**Equivalente Gurobi**: `gurobi_activation` — stessa struttura di vincoli, obiettivo L1 invece di L2

### Perché non L2 (come Gurobi)?

HiGHS supporta solo obiettivi lineari (LP/MILP). Sono stati tentati due approcci L2 alternativi:

1. **CVXPY + SCIP**: malloc crash fatale a livello C — CVXPY costruiva ~22.000 oggetti Python per il grafo simbolico quadratico prima di passarli a SCIP.
2. **pyscipopt diretto + epigraph**: SCIP crashava ugualmente (`munmap_chunk(): invalid pointer`) su modelli grandi con vincoli quadratici.

La soluzione adottata è L1 con 3 livelli in HiGHS: stesso constraint structure di v6, zero rischio crash.

### Formulazione

**Variabili**:
```
z[i,t,k] ∈ {0,1}    — device i al timestep t nel livello k
x[i,t]   ∈ {0,1}    — indicatore ON aggregato = Σ_k z[i,t,k]
u[i,t]   ∈ {0,1}    — rising edge
dw[i,t]  ∈ {0,1}    — falling edge
ep[t]    ≥ 0        — parte positiva dell'errore
en[t]    ≥ 0        — parte negativa dell'errore
```

**Livelli di potenza** (identici a v5/v6):
```
p_i,0 = p_i · (1 − δ)    (low)
p_i,1 = p_i               (mid)
p_i,2 = p_i · (1 + δ)    (high)
```

**Vincolo di link** (al più un livello attivo):
```
x[i,t]  =  Σ_k z[i,t,k]    ∀i,t
```

**Vincolo L1** (linearizzazione del valore assoluto):
```
Σ_i Σ_k p_i,k · z[i,t,k]  −  ep[t]  +  en[t]  =  y[t]    ∀t valido
```

**Obiettivo L1**:
```
min  Σ_t  ( ep[t] + en[t] )
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i · x[i,t]
   + Σ_i  Σ_t               λ_a · p_i · u[i,t]
```

Le penalità scalano per `p_i` (non `p_i²`), coerente con la scala L1.

**Vincoli di transizione, durata min/max**: identici a `constrained_v6` (operano su `x[i,t]`).

---

## cvxpy_full

**File**: `approach_cvxpy_full.py` → `constrained_highs_full` in `highs_methods.py`
**Solver**: HiGHS (via `highspy`)
**Tipo**: MILP
**Granularità**: 15 min
**Equivalente Gurobi**: `gurobi_full` (constrained_v7) — stessa struttura, obiettivo L1

Estende `cvxpy_activation` modellando anche i device always-on come variabili. Nessuna sottrazione di baseline.

### Aggiunto: device always-on come variabili

```
z_ao[j,t,k] ∈ {0,1}   — device always-on j al timestep t nel livello k
```

**Vincolo** (esattamente un livello attivo per ogni timestep valido):
```
Σ_{k=0}^{2}  z_ao[j,t,k]  =  1    ∀j, ∀t valido
```

Nessuna variabile `u`/`dw`, nessun vincolo di transizione o durata per gli always-on.

**Obiettivo L1** (tutti i device):
```
min  Σ_t  ( ep[t] + en[t] )
   + Σ_i  Σ_{t ∉ window_i}  λ_w · p_i · x_ev[i,t]
   + Σ_i  Σ_t               λ_a · p_i · u_ev[i,t]
```

**Vincolo di ricostruzione** (include sempre-accesi):
```
Σ_j Σ_k p_ao_j,k · z_ao[j,t,k]  +  Σ_i Σ_k p_ev_i,k · z_ev[i,t,k]  −  ep[t]  +  en[t]  =  y_raw[t]
```

Il segnale `y_raw` è il segnale aggregato grezzo (non il residuo).

### Confronto con gurobi_full

| | `gurobi_full` (v7) | `cvxpy_full` |
|--|---|---|
| Segnale input | raw | raw |
| Always-on | variabili z_ao nel solver | variabili z_ao nel solver |
| Obiettivo | L2 | L1 |
| Livelli potenza | 3 | 3 |
| Solver | Gurobi | HiGHS |

### Perché non era così prima?

La versione precedente di `cvxpy_full` sottraeva la baseline degli always-on e passava solo il residuo al solver, comportandosi come `cvxpy` + activation penalty. Ora modella tutti i device direttamente come in `gurobi_full`.

---

## Tabella riassuntiva

| Approccio | Solver | Tipo | Obiettivo | Livelli | Act. Penalty | Always-on | Mirror Gurobi |
|-----------|--------|------|-----------|---------|--------------|-----------|---------------|
| `cvxpy` | HiGHS | MILP | L1 | 1 (binary) | no | baseline | `gurobi_soft` (struttura, non obj) |
| `cvxpy_activation` | HiGHS | MILP | L1 | 3 | sì (λ·p) | baseline | `gurobi_activation` (stessa struttura, L1 vs L2) |
| `cvxpy_full` | HiGHS | MILP | L1 | 3 | sì (λ·p) | variabili | `gurobi_full` (stessa struttura, L1 vs L2) |

Le varianti `*_max`, `*_median`, `*_30min` usano la stessa formulazione della variante base con `resample_method` o `granularity_min` diversi.
