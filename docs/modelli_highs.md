# Modelli HiGHS — formulazione completa

Documento autoconsistente: contiene per intero la formulazione dei modelli di ottimizzazione
risolti con **HiGHS**, senza rimandi ai documenti Gurobi. I modelli sono in ordine di
costruzione, ognuno estende il precedente, e l'ultimo (`cvxpy_survey_prior`) è quello attualmente
in uso.

> Nota sui nomi: le chiavi degli approcci iniziano con `cvxpy_` per ragioni storiche, ma **CVXPY
> non è più usato**. Il solver è HiGHS invocato direttamente via `highspy` (`highs_methods.py`).

## Indice

1. [Perché L1 e MILP](#1-perché-l1-e-milp)
2. [Notazione](#2-notazione)
3. [Pipeline comune](#3-pipeline-comune)
4. [Modello base — `cvxpy`](#4-modello-base--cvxpy)
5. [Multilivello — `cvxpy_activation`](#5-multilivello--cvxpy_activation)
6. [Always-on come variabili — `cvxpy_full`](#6-always-on-come-variabili--cvxpy_full)
7. [Durata soft — `cvxpy_soft_duration`](#7-durata-soft--cvxpy_soft_duration)
8. [Quota settimanale — `cvxpy_weekly_quota`](#8-quota-settimanale--cvxpy_weekly_quota)
9. [Prior del questionario — `cvxpy_survey_prior`](#9-prior-del-questionario--cvxpy_survey_prior-versione-finale)
10. [Il fattore temporale f_i(t)](#10-il-fattore-temporale-f_it)
11. [Tabella dei parametri](#11-tabella-dei-parametri)
12. [Note di taratura](#12-note-di-taratura)

---

## 1. Perché L1 e MILP

HiGHS risolve **LP e MILP, non MIQP**: non accetta termini quadratici. Inoltre non esiste una
funzione obiettivo simbolica come in Gurobi — l'obiettivo è un **vettore di coefficienti lineari**
`c`, uno per variabile, e il solver minimizza `cᵀv`. Ogni penalità deve quindi essere un
coefficiente costante su una variabile.

Questo esclude l'errore quadratico `Σ_t (ŷ[t] − y[t])²` usato dai modelli Gurobi. Si usa invece
l'errore **L1**, `Σ_t |ŷ[t] − y[t]|`, linearizzato con la classica decomposizione in parte
positiva e negativa:

```
ŷ[t] − ep[t] + en[t] = y[t]        ep[t] ≥ 0,  en[t] ≥ 0
```

Minimizzando `ep[t] + en[t]` con costo 1 su entrambe, all'ottimo una delle due è sempre nulla e la
somma vale esattamente `|ŷ[t] − y[t]|`. Nessun vincolo aggiuntivo serve a imporlo: siccome
entrambe hanno costo positivo, il solver non ha interesse a gonfiarle.

Conseguenza pratica per la taratura: nei modelli Gurobi le penalità sono scalate per `p_i²` per
essere commensurabili con l'errore quadratico; **qui sono scalate per `p_i`**, che è la scala
dell'errore L1.

---

## 2. Notazione

| Simbolo | Significato |
|---------|-------------|
| `T` | numero di slot nel giorno |
| `t` | indice slot (ampiezza `granularity_min`, default 15 min) |
| `i` | indice event device |
| `j` | indice device always-on |
| `k` | indice livello di potenza, `k ∈ {0,1,2}` (low, mid, high) |
| `K` | numero di livelli = 3 |
| `y[t]` | segnale aggregato allo slot `t` (W) |
| `p_i` | `p_typical_w` del device `i` |
| `p_{i,k}` | livello `k` del device `i` |
| `δ` | `power_level_variation` (default 0.15) |
| `a_i` | slot minimi ON, `⌈dur_min_min / granularity_min⌉` |
| `b_i` | slot massimi ON consecutivi |
| `f_i(t)` | fattore temporale del device `i` allo slot `t` (§10) |
| `V` | insieme degli slot con segnale valido (non NaN) |
| `λ_w` | `time_window_penalty` |
| `λ_a` | `activation_penalty` |
| `λ_d` | `duration_penalty_block` |
| `λ_D` | `duration_penalty_daily` |
| `λ_q,i` | sovrapprezzo di quota settimanale del device `i` |
| `λ_s` | `season_penalty` |

Tutte le variabili binarie e continue sono vincolate a zero sugli slot non validi
(`t ∉ V`), così i buchi nel segnale non generano attivazioni fittizie.

---

## 3. Pipeline comune

Ogni approccio segue lo stesso schema; le differenze sono indicate per modello.

```
1. BASELINE ALWAYS-ON        (solo cvxpy e cvxpy_activation)
   i device always_on=True vengono stimati con estimate_always_on_baseline()
   e sottratti dal segnale; il solver lavora sul residuo.
   In cvxpy_full e successivi questo passo NON esiste: gli always-on
   entrano nel modello come variabili.

2. RESAMPLING
   il segnale a 1 minuto viene ricampionato a granularity_min
   con resample_method (mean | max | min | median).

3. OTTIMIZZAZIONE GIORNO PER GIORNO
   per ogni giorno si costruisce e risolve un MILP indipendente,
   con time_limit secondi di budget.

4. RICOSTRUZIONE A 1 MINUTO
   il risultato a granularità grossa viene forward-filled sull'indice originale.
```

Il fatto che i giorni siano **indipendenti** è rilevante: qualunque grandezza che debba
attraversare i giorni (la quota settimanale) non può essere una variabile del modello e
dev'essere passata come costante e aggiornata fuori dal solver.

---

## 4. Modello base — `cvxpy`

**File**: `approach_cvxpy.py` → `highs_methods.constrained_highs`
**Always-on**: stimati come baseline e sottratti — il solver vede il residuo.
**Livelli**: 1 (ON/OFF binario).

### Variabili

```
x[i,t]  ∈ {0,1}     device i acceso allo slot t
u[i,t]  ∈ {0,1}     fronte di salita (si accende in t)
dw[i,t] ∈ {0,1}     fronte di discesa (si spegne in t)
ep[t]   ≥ 0         errore positivo di ricostruzione
en[t]   ≥ 0         errore negativo di ricostruzione
```

### Obiettivo

```
min  Σ_{t∈V} ( ep[t] + en[t] )
   + Σ_i Σ_{t∈V}  λ_w · p_i · f_i(t) · x[i,t]
   + Σ_i Σ_{t∈V}  λ_a · p_i · u[i,t]
```

### Vincoli

**Ricostruzione L1** — un'uguaglianza per ogni slot valido:
```
Σ_i p_i · x[i,t]  −  ep[t]  +  en[t]  =  y[t]           ∀t ∈ V
```

**Transizioni** — legano `u` e `dw` ai cambi di stato:
```
u[i,0]  =  x[i,0]                                        ∀i
dw[i,0] =  0                                             ∀i
x[i,t] − x[i,t−1] − u[i,t] + dw[i,t]  =  0               ∀i, ∀t ≥ 1
u[i,t] + dw[i,t]  ≤  1                                   ∀i, ∀t ≥ 1
```

**Durata minima ON** (attiva solo se `a_i ≥ 2`) — se il device si accende in `t`, deve restare
acceso per almeno `a_i` slot:
```
Σ_{τ=t}^{t+a_i−1} x[i,τ]  −  a_i · u[i,t]  ≥  0          ∀t ∈ [0, T−a_i]
```

**Durata massima ON consecutiva** — nessuna finestra di `b_i+1` slot può essere tutta accesa:
```
Σ_{τ=t}^{t+b_i} x[i,τ]  ≤  b_i                           ∀t ∈ [0, T−b_i−1]
```

---

## 5. Multilivello — `cvxpy_activation`

**File**: `approach_cvxpy_activation.py` → `highs_methods.constrained_highs_multistate`
**Always-on**: baseline sottratta (come sopra).
**Livelli**: 3.

Estende il modello base: ogni device può operare a tre potenze invece che ON/OFF.

```
p_{i,0} = p_i · (1 − δ)      p_{i,1} = p_i      p_{i,2} = p_i · (1 + δ)
```

### Variabile aggiuntiva

```
z[i,t,k] ∈ {0,1}    device i allo slot t è al livello k
```

### Vincolo di link

```
x[i,t]  −  Σ_{k=0}^{2} z[i,t,k]  =  0                    ∀i, ∀t
```

Poiché `x[i,t] ∈ {0,1}`, questo impone automaticamente che **al più un livello** sia attivo: non
serve un vincolo SOS separato. `x` resta l'indicatore ON aggregato e tutti i vincoli di
transizione e durata del §4 continuano a operare su di esso, invariati.

### Ricostruzione aggiornata

```
Σ_i Σ_k p_{i,k} · z[i,t,k]  −  ep[t]  +  en[t]  =  y[t]  ∀t ∈ V
```

L'output restituisce la potenza effettivamente stimata `Σ_k p_{i,k}·z[i,t,k]`, non `p_i` fisso.

---

## 6. Always-on come variabili — `cvxpy_full`

**File**: `approach_cvxpy_full.py` → `highs_methods.constrained_highs_full`
**Always-on**: **modellati come variabili**, nessuna baseline sottratta.

Elimina il passo 1 della pipeline. I device `always_on=True` non vengono più stimati a monte e
sottratti: entrano nel modello, e il solver ottimizza **congiuntamente** il loro livello di
potenza e le attivazioni degli event device contro il segnale grezzo.

I device si dividono in due gruppi con trattamento diverso.

### Always-on (`j`)

```
z_ao[j,t,k] ∈ {0,1}
```

Vincolo di attività permanente — *esattamente* un livello attivo a ogni slot valido:

```
Σ_{k=0}^{2} z_ao[j,t,k]  =  1                            ∀j, ∀t ∈ V
```

Nessuna variabile `u`/`dw`, nessun vincolo di transizione, durata o finestra: non si spengono mai.

### Event device (`i`)

Identici al §5: `z_ev`, `x_ev`, `u_ev`, `dw_ev` con tutti i vincoli di transizione e durata.

### Ricostruzione

```
Σ_j Σ_k p^ao_{j,k} · z_ao[j,t,k]  +  Σ_i Σ_k p_{i,k} · z_ev[i,t,k]
    −  ep[t]  +  en[t]  =  y[t]                          ∀t ∈ V
```

`y` è qui il segnale **aggregato grezzo**, non il residuo.

> ⚠️ **Limite strutturale.** Il vincolo `Σ_k z_ao = 1` è un'uguaglianza: gli always-on *devono*
> consumare almeno `Σ_j p^ao_j·(1−δ)` a ogni istante. Se il questionario dichiara always-on la cui
> somma supera il segnale medio della casa, il modello sovrastima per costruzione e il residuo
> diventa negativo. È il caso osservato sull'IMEI `86853106211179`: 325 W dichiarati contro
> 159 W di segnale medio, con errore energetico del 119%.

---

## 7. Durata soft — `cvxpy_soft_duration`

**File**: `approach_cvxpy_soft_duration.py` → `constrained_highs_full` con `λ_d > 0`

Stesso modello del punto 6, ma i limiti di durata **non sono più vincoli rigidi**.

### Motivazione

Le durate del questionario sono approssimative. Con un tetto rigido, un device la cui potenza
combacia esattamente con un plateau viene rifiutato solo perché il plateau dura poco più del
massimo dichiarato: con lavatrice a `max = 2h` e un plateau di 2h15 il modello copre 8 slot su 9 e
lascia 15 minuti inspiegati.

### Formulazione

Le stesse righe del §4, con l'aggiunta di uno **slack non negativo**:

```
Σ_{τ=t}^{t+a_i−1} x[i,τ]  +  s⁻[i,t]  −  a_i·u[i,t]  ≥  0        s⁻[i,t] ≥ 0
Σ_{τ=t}^{t+b_i}   x[i,τ]  −  s⁺[i,t]                 ≤  b_i       s⁺[i,t] ≥ 0
```

con il termine di costo aggiunto all'obiettivo:

```
+ Σ_i Σ_t  λ_d · p_i · ( s⁻[i,t] + s⁺[i,t] )
```

**Il costo risulta lineare nello sforamento senza doverlo imporre.** Un blocco acceso di lunghezza
`L > b_i` contiene esattamente `L − b_i` finestre di ampiezza `b_i+1` interamente al suo interno, e
ognuna richiede `s⁺ ≥ 1`: il costo totale è `λ_d · p_i · (L − b_i)`. Simmetricamente un blocco più
corto di `a_i` paga `λ_d · p_i · (a_i − L)`.

Costo computazionale: **2 variabili continue** per `(device, slot)`, nessuna binaria aggiuntiva —
la difficoltà del MILP resta sostanzialmente invariata.

### Budget giornaliero (opzionale, `λ_D > 0`)

Penalizza lo scostamento del monte ore giornaliero dall'attesa del questionario:

```
E_i  =  (frequency_per_week / 7) · dur_typical_min / granularity_min

Σ_t x[i,t]  −  d⁺[i]  +  d⁻[i]  =  E_i                   d⁺, d⁻ ≥ 0
```

costo `+ Σ_i λ_D · p_i · ( d⁺[i] + d⁻[i] )`.

A differenza dello slack per-blocco è **globale**: non distingue un blocco lungo da tanti corti.
Nei modelli successivi il default è `λ_D = 0` (disattivato), perché sui dati reali peggiorava
leggermente il MAE.

Il limite `b_i` viene preso da `duration_minutes_max` del questionario quando disponibile, invece
di `dur_typical_min × 1.5`: ora che è un bound soft ha senso usare il valore dichiarato.

---

## 8. Quota settimanale — `cvxpy_weekly_quota`

**File**: `approach_cvxpy_weekly_quota.py` → `constrained_highs_full` con quota

Usa la frequenza settimanale del questionario come **budget scorrevole**: se la lavatrice è
dichiarata 2 volte a settimana e ne sono già state usate 2, la terza accensione costa molto di più.

### Il contatore vive fuori dal modello

Poiché ogni giorno è un modello indipendente, il conteggio settimanale **non può essere una
variabile**: è stato che attraversa i solve.

```
1. i giorni si processano in ordine cronologico
2. il contatore si azzera al cambio di settimana ISO (ogni lunedì)
3. la quota residua r_i entra nel modello del giorno come COSTANTE
4. dopo il solve si contano i fronti di salita nella soluzione
   e si sommano al contatore
```

### Dentro il modello: solo l'eccedenza

Una sola variabile continua per device:

```
Σ_t u[i,t]  −  e_i  ≤  r_i                               e_i ≥ 0
```

costo `+ Σ_i λ_q,i · p_i · e_i`.

Le prime `r_i` accensioni della settimana pagano solo `λ_a`; ognuna oltre la quota paga anche il
sovrapprezzo. Con `r_i = 0` (quota esaurita) ogni accensione del giorno lo paga.

### Taratura: il break-even dipende dal device

A differenza delle altre penalità **il break-even non è a λ = 1**. Spiegare un'attivazione di un
device che resta acceso `s_i` slot fa guadagnare circa `s_i · p_i`, quindi il sovrapprezzo deve
superare quel valore. Verificato sperimentalmente: con attivazioni da 4 slot il comportamento
cambia esattamente a `λ_q = 4`.

Per questo il sovrapprezzo è **per device**:

```
λ_q,i  =  over_activation_factor  ×  s_i          s_i = dur_typical_min / granularity_min
```

Con `over_activation_factor = 1.5` la forza effettiva è la stessa per un ciclo di microonde da 15
minuti e per uno di lavatrice da 2 ore. Un λ scalare sarebbe mal calibrato tra i due.

### Limite noto: allocazione greedy

Risolvendo un giorno alla volta, la quota viene spesa in modo greedy: un giorno iniziale la consuma
liberamente e i successivi della stessa settimana pagano. Distribuirla in modo ottimo richiederebbe
risolvere i sette giorni congiuntamente, molto più costoso. La versione greedy evita comunque
l'esito implausibile "lavatrice accesa tutti i giorni", che è lo scopo.

---

## 9. Prior del questionario — `cvxpy_survey_prior` (versione finale)

**File**: `approach_cvxpy_survey_prior.py` → `constrained_highs_full` con tutti i prior attivi

Chiude due buchi nell'uso del questionario. La formulazione è quella del §8 più due termini.

### 9.1 Stagionalità

`active_months` era raccolto dal questionario ma **ignorato da tutti i solver**: solo gli approcci
FHMM/HSMM lo leggevano. Un climatizzatore dichiarato attivo giugno–settembre era quindi libero di
spiegare un plateau di dicembre.

```
costo  +=  λ_s · p_i · x[i,t]        ∀t il cui mese ∉ active_months_i
```

Con `λ_s = 8` il device è di fatto rimosso fuori stagione, ma resta una penalità **soft**: se
l'evidenza del segnale è schiacciante può comunque emergere, coerentemente con la filosofia
"questionario come prior, non veto" adottata per durata e finestre.

### 9.2 Specificità della finestra oraria

Un device che non aveva dichiarato alcuna finestra **non pagava nulla, mai, da nessuna parte**: era
un jolly gratuito capace di assorbire qualsiasi plateau a qualsiasi ora. Contemporaneamente un
device *dentro* la propria finestra pagava anch'esso zero. Pareggiavano, e la scelta tra i due era
arbitraria.

La correzione **non** è punire chi non ha un orario — molti device davvero non ne hanno, e
l'assenza di dichiarazione è assenza di informazione, non evidenza di implausibilità. Si legge
invece "nessuna finestra" come **"finestra di 24h"** e si penalizza quanto poco la dichiarazione
restringe il campo:

```
vaghezza_i  =  log(W_i) / log(24)          W_i = ampiezza dichiarata in ore (24 se assente)
```

Deriva da `−log P` con prior uniforme su `W` ore: chi si sbilancia su poche ore fa un'affermazione
più falsificabile e incassa lo sconto quando l'evento cade lì dentro.

| device | finestra | W | vaghezza |
|---|---|---:|---:|
| Lavastoviglie | 19:00–22:00 | 3h | 0.35 |
| Lavatrice | 17:00–22:00 | 5h | 0.51 |
| Lavatrice | 10:00–17:00 | 7h | 0.61 |
| Climatizzatore | nessuna | 24h | 1.00 |

La vaghezza entra come **pavimento** sul fattore `f_i(t)` già esistente (§10) — non è un
meccanismo nuovo.

> **Perché penalità e non premio.** Un premio per stare dentro la finestra sarebbe un coefficiente
> **negativo** nell'obiettivo, e il solver accenderebbe device solo per incassarlo anche in assenza
> di segnale da spiegare. La penalità sulla vaghezza produce lo **stesso ordinamento relativo**
> mantenendo tutti i costi ≥ 0.

---

## 10. Il fattore temporale `f_i(t)`

Compare in tutti i modelli, moltiplicato per `λ_w · p_i` sul termine `x[i,t]`. Implementato in
`time_windows.window_penalty_factors`.

```
d_i(t)  =  distanza circolare in minuti da t alla finestra più vicina di i  (0 se dentro)

f_i(t)  =  w · vaghezza_i                                       se t è dentro la finestra
        =  w · vaghezza_i + min(max_factor, 1 + d_i(t)/ramp)    se t è fuori
```

dove `w = window_vagueness_weight`. Con `w = 0` (default nei modelli §4–§8) il pavimento sparisce e
si ritrova il comportamento originario: 0 dentro la finestra, e un device senza finestra non paga
nulla.

Il termine di distanza cresce di 1 ogni `ramp` minuti di allontanamento e satura a `max_factor`
invece di divergere. La distanza è **circolare**, quindi le finestre che scavalcano la mezzanotte
(es. 23:00–02:00) sono gestite correttamente.

Esempio con `w = 1`, finestra lavastoviglie 19:00–22:00, contro un climatizzatore senza finestra:

| ora | lavastoviglie | climatizzatore | vince |
|---|---:|---:|---|
| 10:00 | 6.35 | 1.00 | climatizzatore |
| 19:00 | 0.35 | 1.00 | lavastoviglie |
| 21:00 | 0.35 | 1.00 | lavastoviglie |
| 22:00 | 1.36 | 1.00 | climatizzatore |

Il device specifico vince **solo dentro la finestra che ha dichiarato**, e l'ordinamento si inverte
correttamente fuori.

---

## 11. Tabella dei parametri

Default della versione finale `cvxpy_survey_prior`.

| Parametro | Default | Significato |
|---|---:|---|
| `granularity_min` | 15 | ampiezza dello slot in minuti |
| `resample_method` | `mean` | aggregazione nel ricampionamento |
| `time_limit` | 60 | budget HiGHS per giorno (secondi) |
| `power_level_variation` (δ) | 0.15 | scarto dei livelli low/high da `p_typical` |
| `time_window_penalty` (λ_w) | 1.0 | peso del termine temporale |
| `window_penalty_ramp_min` | 60 | minuti di distanza che aggiungono 1 al fattore |
| `window_penalty_max_factor` | 6.0 | saturazione del termine di distanza |
| `activation_penalty` (λ_a) | 1.0 | costo per accensione |
| `duration_penalty_block` (λ_d) | 0.5 | costo per slot di sforamento durata |
| `duration_penalty_daily` (λ_D) | 0.0 | budget ore/giorno (disattivato) |
| `over_activation_factor` | 1.5 | moltiplicatore del break-even di quota |
| `window_vagueness_weight` (w) | 0.2 | peso del pavimento di vaghezza |
| `season_penalty` (λ_s) | 8.0 | costo per slot fuori stagione |

### Quali modelli attivano cosa

| Modello | Always-on | Livelli | Durata | Quota | Stagione | Vaghezza |
|---|---|---:|---|---|---|---|
| `cvxpy` | baseline | 1 | hard | — | — | — |
| `cvxpy_activation` | baseline | 3 | hard | — | — | — |
| `cvxpy_full` | variabili | 3 | hard | — | — | — |
| `cvxpy_soft_duration` | variabili | 3 | soft | — | — | — |
| `cvxpy_weekly_quota` | variabili | 3 | soft | ✓ | — | — |
| **`cvxpy_survey_prior`** | variabili | 3 | soft | ✓ | ✓ | ✓ |

Le varianti `_30min`, `_max`, `_median` cambiano solo `granularity_min` o `resample_method`, non la
formulazione.

---

## 12. Note di taratura

**`time_window_penalty` in L1.** Con `λ_w = 1.0` la penalità di finestra vale `p_i`, cioè
esattamente quanto costa lasciare quel watt non spiegato: la finestra diventa di fatto un vincolo
rigido e il device non si accende mai fuori. Per restare nella zona in cui la gradazione discrimina
davvero servono valori intorno a **0.2–0.4**.

**`window_vagueness_weight`.** Tarato su case reali. A `w = 1.0` il MAE su `86853106211173`
peggiorava da 118 a 220 perché device interi venivano eliminati; a `w = 0.2` l'attribuzione si
sposta verso i device dichiarati al costo di circa +8% di MAE:

| device | w = 0 | w = 0.2 | finestra |
|---|---:|---:|:--|
| Climatizzatore | 8.1 | 0.0 | — |
| Lavastoviglie | 0.0 | 1.2 | ✓ |
| Lavatrice | 1.2 | 4.8 | ✓ |
| Televisori | 21.0 | 18.1 | — |

Il trade-off è intrinseco: si compra plausibilità a scapito del fit. Senza ground truth il MAE non
è il criterio giusto per giudicare questi prior — serve a verificare che non stiano degradando la
ricostruzione oltre il ragionevole.

**Break-even delle penalità.** Per `λ_a`, `λ_d`, `λ_D` il break-even è a **1.0**: un'unità di
violazione costa quanto lasciare `p_i` watt non spiegati per uno slot. Per `λ_q` il break-even è
invece la **lunghezza in slot** dell'attivazione tipica, ed è per questo che viene calcolato per
device.
