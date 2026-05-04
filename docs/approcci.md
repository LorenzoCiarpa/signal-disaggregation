# Approcci di disaggregazione NILM — Dettaglio tecnico e matematico

## Indice

1. [Problema formale](#1-problema-formale)
2. [Preprocessing e dati in ingresso](#2-preprocessing-e-dati-in-ingresso)
3. [Approccio A — Event-based](#3-approccio-a--event-based)
4. [Approccio B — HMM Gaussiana per singolo dispositivo](#4-approccio-b--hmm-gaussiana-per-singolo-dispositivo)
5. [Approccio C — Factorial HMM semplificato (FHMM)](#5-approccio-c--factorial-hmm-semplificato-fhmm)
6. [Approccio D — Template matching](#6-approccio-d--template-matching)
7. [Approccio E — Event-based con prior bayesiano](#7-approccio-e--event-based-con-prior-bayesiano)
8. [Confronto e risultati reali](#8-confronto-e-risultati-reali)
9. [Limitazioni strutturali comuni](#9-limitazioni-strutturali-comuni)

---

## 1. Problema formale

Il segnale misurato `w_medio(t)` è la **somma** dei consumi di tutti i dispositivi attivi:

$$w(t) = \sum_{i=1}^{N} P_i \cdot x_i(t) + \varepsilon(t)$$

dove:
- $P_i$ è la potenza tipica del dispositivo $i$ (Watt), nota dalla knowledge base
- $x_i(t) \in \{0, 1\}$ è lo **stato latente** (acceso/spento) — quello che vogliamo stimare
- $\varepsilon(t)$ è rumore di misura e consumi non modellati
- $t$ è l'indice temporale campionato a **1 minuto**
- $N = 16$ dispositivi nella knowledge base

Il problema è **sottovincolato**: con un solo segnale scalare e 16 incognite binarie per ogni istante `t`, esistono moltissime soluzioni compatibili con le osservazioni. Ogni approccio introduce vincoli (e assunzioni) diversi per selezionarne una.

---

## 2. Preprocessing e dati in ingresso

Prima di qualsiasi approccio, il segnale grezzo viene preprocessato da `preprocessing.py`:

1. **Caricamento JSON**: ogni record ha `_id` (epoch Unix), `epoch_valid`, `w_medio`
2. **Filtraggio**: scartati i record con `epoch_valid=false` e i picchi con `w_medio > 10000W`
3. **Ricampionamento**: `resample('1min').mean()` porta tutto alla stessa frequenza
4. **Reindexing**: viene creato un indice completo minuto per minuto; i buchi sono `NaN`

Il risultato è una `pd.Series` con `DatetimeIndex` UTC:

```
2025-11-13 15:42:00+00:00    193.0
2025-11-13 15:43:00+00:00    192.0
2025-11-13 15:44:00+00:00    197.0
2025-11-13 15:45:00+00:00    NaN      ← gap nei dati
2025-11-13 15:46:00+00:00    185.0
```

Ogni approccio riceve questo segnale e la lista di `DeviceProfile` per l'IMEI, dove ogni profilo include:
- `p_typical_w`: potenza tipica (W)
- `dur_min_min`, `dur_typical_min`: durata minima e tipica dell'attivazione (minuti)
- `prior_weight`: 1.0 se presente nell'inventario dichiarato dal survey, 0.05 se assente
- `frequency_per_week`: frequenza settimanale d'uso dichiarata

---

## 3. Approccio A — Event-based

**File:** `scripts/nilm/approach_event_based.py`

### Idea

I dispositivi elettrici non si accendono gradualmente: producono **transizioni brusche** nel segnale aggregato. Rilevare queste transizioni permette di stimare quando ogni dispositivo si è acceso o spento.

### Matematica

**Step 1 — Derivata discreta:**

$$\Delta w(t) = w(t) - w(t-1)$$

Implementata come `signal.diff()`.

**Step 2 — Rilevamento eventi:**

Un evento di **accensione (ON)** avviene quando:
$$\Delta w(t) > \theta \quad \text{con } \theta = 50\,\text{W (default)}$$

Un evento di **spegnimento (OFF)** avviene quando:
$$\Delta w(t) < -\theta$$

**Step 3 — Assegnazione al dispositivo:**

Per ogni evento ON con salto $|\Delta w(t)|$, si seleziona il dispositivo $d^*$ tra quelli **presenti** (`prior_weight ≥ 1.0`):

$$d^* = \argmin_{d \in \text{presenti}} \left| P_d - |\Delta w(t)| \right|$$

È la **distanza euclidea** tra il salto osservato e la potenza tipica di ogni device.

**Step 4 — Pairing ON/OFF:**

Trovato il device $d^*$, si definisce la **finestra di ricerca** come 3 volte la sua durata tipica:

$$W = 3 \cdot \tau_{d^*}$$

dove $\tau_{d^*}$ è `dur_typical_min` del device assegnato (es. 270 minuti per la lavatrice con $\tau=90$ min).

Si scorre la lista degli eventi OFF già individuati nello Step 2 e si prende il **primo** che cade dopo $t_{\text{on}}$ entro la finestra $W$:

$$t_{\text{off}} = \min_{t} \left\{ t : t > t_{\text{on}},\quad \Delta w(t) < -\theta,\quad t - t_{\text{on}} \leq W \right\}$$

In italiano: "il primo spegnimento rilevato dopo l'accensione, purché non sia passato più di $W$ minuti."

Se nella finestra non esiste nessun evento OFF (es. il segnale termina, o non ci sono discese abbastanza brusche), si usa la durata tipica come fallback:

$$t_{\text{off}} = t_{\text{on}} + \tau_{d^*}$$

In questo caso il device viene marcato ON per esattamente la sua durata attesa, senza conferma da un evento di spegnimento reale.

**Step 5 — Costruzione output:**

$$\hat{P}_{d^*}(t) = \begin{cases} P_{d^*} & \text{se } t_{\text{on}} \leq t < t_{\text{off}} \\ 0 & \text{altrimenti} \end{cases}$$

### Esempio numerico

Segnale (estratto reale):

| t | w(t) | Δw(t) |
|---|------|--------|
| 08:00 | 180W | — |
| 08:01 | 185W | +5W |
| 08:02 | 1990W | **+1805W** ← evento ON |
| 08:03 | 1985W | -5W |
| ... | ... | ... |
| 09:32 | 1980W | -3W |
| 09:33 | 190W | **-1790W** ← evento OFF |

Evento ON a 08:02: `|Δw| = 1805W`. Device più vicino tra i presenti:
- Lavatrice: `|1800 - 1805| = 5W` ✓
- Boiler: `|2000 - 1805| = 195W`
- Forno: `|2500 - 1805| = 695W`

Vince la **Lavatrice**. La finestra massima è `90 × 3 = 270 min`. L'evento OFF a 09:33 è a 91 min: compatibile. La lavatrice viene marcata ON da 08:02 a 09:33.

### Limiti

- **Cieco alle co-attivazioni**: se lavatrice e boiler si accendono nello stesso minuto, `Δw = +3800W`. Il device più vicino è `Auto elettrica` (3700W), non la vera combinazione.
- **Un evento → un device**: non può spiegare salti composti.
- **Threshold fisso**: a 50W molti eventi reali (frigorifero che cicla) generano falsi positivi.

---

## 4. Approccio B — HMM Gaussiana per singolo dispositivo

**File:** `scripts/nilm/approach_hmm.py`

### Idea

Ogni dispositivo è modellato come un processo stocastico a 2 stati latenti (ON/OFF), dove la potenza osservata in ogni stato segue una distribuzione gaussiana. I parametri del modello vengono appresi dai dati tramite l'algoritmo di Baum-Welch (EM per HMM).

### Struttura del modello

Un HMM è definito da tre parametri $\lambda = (\pi, A, B)$:

**Distribuzione iniziale** $\pi$ (vettore 2×1):
$$\pi_i = P(q_1 = i), \quad i \in \{0=\text{OFF},\, 1=\text{ON}\}$$

**Matrice di transizione** $A$ (2×2):
$$A_{ij} = P(q_{t+1} = j \mid q_t = i)$$

Esempio per il frigorifero (cicla ogni ~20 min con duty cycle 30%):
```
        → OFF    → ON
da OFF [ 0.85   0.15 ]   ← rimane spento con prob 85%
da ON  [ 0.05   0.95 ]   ← rimane acceso con prob 95%
```

**Modello di emissione gaussiano** $B$:
$$P(y_t \mid q_t = i) = \mathcal{N}(y_t;\, \mu_i,\, \sigma_i^2)$$

Nel codice, le medie e varianze vengono **inizializzate dalla knowledge base** prima del fitting:

```python
# Esempio per la Lavatrice (p_typical = 1800W)
means_  = [[5.0],     [1800.0]]   # [μ_OFF, μ_ON]
covars_ = [[100.0],   [129600.0]] # [σ²_OFF, σ²_ON]
#           σ_OFF=10W               σ_ON = 0.2×1800 = 360W
```

La varianza ON viene impostata a $(0.2 \cdot P_{\text{typical}})^2$: si assume che il device non consumi esattamente `P_typical` ma con una deviazione standard del 20%.

### Algoritmo di Baum-Welch (E-step + M-step)

Il fitting avviene sul **segnale residuo** — il segnale aggregato meno tutti i device già stimati nei passi precedenti — per cercare di isolare il contributo del device corrente.

**E-step: Forward-Backward**

Calcola $\gamma_t(i) = P(q_t = i \mid \mathbf{y}, \lambda)$ tramite due passaggi ricorsivi:

*Forward* $\alpha_t(i) = P(y_1, \ldots, y_t, q_t=i \mid \lambda)$:
$$\alpha_1(i) = \pi_i \cdot \mathcal{N}(y_1; \mu_i, \sigma_i^2)$$
$$\alpha_t(j) = \mathcal{N}(y_t; \mu_j, \sigma_j^2) \cdot \sum_{i} \alpha_{t-1}(i) \cdot A_{ij}$$

*Backward* $\beta_t(i) = P(y_{t+1}, \ldots, y_T \mid q_t=i, \lambda)$:
$$\beta_T(i) = 1$$
$$\beta_t(i) = \sum_j A_{ij} \cdot \mathcal{N}(y_{t+1}; \mu_j, \sigma_j^2) \cdot \beta_{t+1}(j)$$

*Probabilità posteriori*:
$$\gamma_t(i) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_j \alpha_t(j) \cdot \beta_t(j)}$$

**M-step: aggiornamento parametri**

$$\mu_i^{\text{new}} = \frac{\sum_t \gamma_t(i) \cdot y_t}{\sum_t \gamma_t(i)}$$

$$(\sigma_i^2)^{\text{new}} = \frac{\sum_t \gamma_t(i) \cdot (y_t - \mu_i^{\text{new}})^2}{\sum_t \gamma_t(i)}$$

$$A_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

dove $\xi_t(i,j) = P(q_t=i, q_{t+1}=j \mid \mathbf{y}, \lambda) = \frac{\alpha_t(i) \cdot A_{ij} \cdot \mathcal{N}(y_{t+1};\mu_j,\sigma_j^2) \cdot \beta_{t+1}(j)}{P(\mathbf{y} \mid \lambda)}$

Vengono eseguite **50 iterazioni** (`n_iter=50`).

### Viterbi per la predizione

Dopo il fitting, `model.predict(obs)` usa l'algoritmo di **Viterbi** per trovare la sequenza di stati globalmente più probabile:

$$q_1^*, \ldots, q_T^* = \argmax_{q_{1:T}} P(q_{1:T} \mid \mathbf{y}, \lambda)$$

Programmazione dinamica con $v_t(j) = \max_{q_{1:t-1}} P(q_{1:t-1}, q_t=j, y_{1:t})$:

$$v_t(j) = \mathcal{N}(y_t;\mu_j,\sigma_j^2) \cdot \max_i \left[ v_{t-1}(i) \cdot A_{ij} \right]$$

### Esempio numerico

Segnale residuo (dopo aver sottratto il frigo):

```
t=1: 180W  → vicino a OFF (μ=5W)? No. Vicino a ON (μ=1800W)? No.
t=2: 1795W → vicino a ON (μ=1800W)? Sì! γ_ON(2) ≈ 0.97
t=3: 1810W → γ_ON(3) ≈ 0.98
...
t=91: 1750W → γ_ON(91) ≈ 0.91
t=92: 185W  → γ_OFF(92) ≈ 0.99
```

Il Viterbi percorre questo segnale e restituisce `[0,0,1,1,1,...,1,0,0]`. Lo stato 1 = ON viene identificato come quello con `μ_ON > μ_OFF` dopo il fitting.

### Problema critico: propagazione del residuo negativo

Il device viene processato **sequenzialmente**: prima il frigorifero, poi la lavatrice, poi il boiler, ecc. Se il frigorifero (HMM) classifica troppi campioni come ON, il residuo per la lavatrice diventa:

$$r_{\text{lavatrice}}(t) = w(t) - \hat{P}_{\text{frigo}}(t)$$

Se `w(t) = 200W` e il frigo ha classificato anche quel campione come ON (`P_frigo = 150W`), il residuo è `50W`. La lavatrice viene fittata su un segnale dove la maggior parte dei campioni è intorno a 50W — molto lontano da `μ_ON_init = 1800W`. Il Baum-Welch fa convergere `μ_ON` verso la media del residuo, e il risultato è che quasi tutti i campioni vengono classificati come ON alla potenza sbagliata.

Questo spiega il **MAE di 3011W** e `energy_error_pct > 1000%` osservati nel benchmark.

---

## 5. Approccio C — Factorial HMM semplificato (FHMM)

**File:** `scripts/nilm/approach_fhmm.py`

### Idea

Invece di stimare ogni dispositivo separatamente sul residuo, tutti i dispositivi vengono stimati **simultaneamente** minimizzando l'errore di ricostruzione globale tramite un algoritmo iterativo di coordinate-ascent.

### Modello

Lo stato del sistema al tempo $t$ è il vettore binario:

$$\mathbf{x}(t) = (x_1(t), x_2(t), \ldots, x_N(t)) \in \{0,1\}^N$$

La potenza stimata è:

$$\hat{w}(t) = \sum_{i=1}^{N} P_i \cdot x_i(t)$$

Si vuole minimizzare l'errore di ricostruzione quadratico su tutto il segnale:

$$\mathcal{L}(\mathbf{X}) = \sum_{t=1}^{T} \left( w(t) - \sum_{i=1}^{N} P_i \cdot x_i(t) \right)^2$$

### Algoritmo: Coordinate-Ascent Greedy

Minimizzazione alternata: si fissa tutto e si ottimizza un device alla volta.

**Inizializzazione**: $x_i(t) = 0 \;\forall i, t$ (tutti spenti)

**Iterazione** (ripetuta fino a convergenza o `max_iter=50`):

Per ogni device $i$, per ogni $t$:

1. Calcola la contribuzione di tutti gli altri device:
$$c_{-i}(t) = \sum_{j \neq i} P_j \cdot x_j(t)$$

2. Calcola il segnale residuo escludendo il device $i$:
$$r_i(t) = w(t) - c_{-i}(t)$$

3. Scegli lo stato che minimizza $|r_i(t) - P_i \cdot x_i(t)|$:

$$x_i(t) = \begin{cases} 1 & \text{se } |r_i(t) - P_i| < |r_i(t) - 0| \\ 0 & \text{altrimenti} \end{cases}$$

La condizione $|r_i - P_i| < |r_i|$ si semplifica algebricamente:

$$|r_i - P_i| < |r_i| \iff r_i > \frac{P_i}{2}$$

Nel codice:
```python
new_state_i = (residual > p_typicals[i] / 2.0).astype(float)
```

**Convergenza**: si ferma quando il numero di cambiamenti di stato in un'intera iterazione è inferiore a $\tau = 10^{-3} \cdot T$ (meno dello 0.1% dei campioni cambia).

### Smoothing temporale

Dopo la convergenza, si rimuovono i blocchi ON troppo brevi:

Per ogni device $i$, ogni run consecutivo di $x_i(t)=1$ di lunghezza $L < \lfloor d_i^{\min}/2 \rfloor$ viene azzerato.

Esempio: il frigorifero ha `dur_min_min = 5` min → vengono rimossi blocchi ON più brevi di 2.5 minuti (arrotondato a 2 campioni). Questo elimina spike spurii da rumore.

### Esempio numerico completo

Siano 3 device attivi (per semplicità): Frigo (150W), Lavatrice (1800W), Boiler (2000W).

**Segnale osservato:**
```
t=1: 150W   (solo frigo)
t=2: 1950W  (frigo + lavatrice)
t=3: 2000W  (solo boiler)
t=4: 2150W  (frigo + boiler)
```

**Iterazione 1**, aggiornamento Frigo ($P=150$W, soglia=75W):
```
t=1: r_frigo = 150-0-0=150 > 75 → x_frigo=1 ✓
t=2: r_frigo = 1950-0-0=1950 > 75 → x_frigo=1 (ma lavatrice non ancora ON)
t=3: r_frigo = 2000-0-0=2000 > 75 → x_frigo=1 (non corretto, boiler non ancora ON)
t=4: r_frigo = 2150-0-0=2150 > 75 → x_frigo=1 ✓
```

**Iterazione 1**, aggiornamento Lavatrice ($P=1800$W, soglia=900W):
```
t=1: r_lav = 150-150-0=0 < 900 → x_lav=0 ✓
t=2: r_lav = 1950-150-0=1800 > 900 → x_lav=1 ✓
t=3: r_lav = 2000-150-0=1850 > 900 → x_lav=1 (non corretto)
t=4: r_lav = 2150-150-0=2000 > 900 → x_lav=1 (non corretto)
```

**Iterazione 1**, aggiornamento Boiler ($P=2000$W, soglia=1000W):
```
t=1: r_boil = 150-150-0=0 < 1000 → x_boil=0 ✓
t=2: r_boil = 1950-150-1800=0 < 1000 → x_boil=0 ✓
t=3: r_boil = 2000-150-1800=50 < 1000 → x_boil=0 ✗ (dovrebbe essere 1)
t=4: r_boil = 2150-150-1800=200 < 1000 → x_boil=0 ✗
```

Continuando le iterazioni, il frigo viene corretto:
```
Iterazione 2, Frigo:
t=3: r_frigo = 2000-0-1800=200 > 75 → x_frigo=1... ancora sbagliato
     ma r_boil non è ancora 1

Iterazione 2, Boiler:
t=3: r_boil = 2000-150-1800=50 < 1000 → ancora sbagliato
```

Questo dimostra che il FHMM greedy può convergere a un **minimo locale** — in questo caso la soluzione `{frigo+lavatrice a t=3}` anziché `{boiler a t=3}` — perché l'ordine di aggiornamento (frigo prima del boiler) ha fissato una soluzione sub-ottimale. Il minimo locale dipende dall'ordine della lista `present_devices`.

### Perché è comunque il migliore

Il coordinamento simultaneo evita la doppia assegnazione sistematica: se il frigo è già ON a `t=1` (150W), la lavatrice vede residuo `0W` a `t=1` e non tenta di accendersi. Con l'HMM sequenziale invece, il modello della lavatrice non "sa" cosa ha già assegnato il frigo — sono indipendenti.

---

## 6. Approccio D — Template matching

**File:** `scripts/nilm/approach_template.py`

### Idea

Ogni dispositivo lascia nel segnale una "firma" riconoscibile: un blocco di potenza approssimativamente costante per una durata caratteristica. Si costruisce un **template rettangolare** per ogni device e si cerca dove nel segnale quel template si "adatta" meglio tramite cross-correlazione.

È l'unico approccio **senza fitting** — non esiste una fase di addestramento.

### Matematica

**Template per il device $i$:**

$$h_i = \underbrace{[P_i, P_i, \ldots, P_i]}_{D_i \text{ campioni}}$$

dove $D_i = \lfloor d_i^{\text{typical}} \rfloor$ (durata tipica in minuti = campioni a 1 min).

Esempio per la Lavatrice: `P=1800W`, `dur_typical=90 min` → vettore di 90 valori tutti a 1800.

**Cross-correlazione normalizzata:**

$$C_i(\tau) = \frac{\sum_{k=0}^{D_i-1} w(\tau+k) \cdot h_i(k)}{\|h_i\|}$$

Implementata con `scipy.signal.correlate(w, h_i, mode='same')` diviso $\|h_i\|_2 = P_i \cdot \sqrt{D_i}$.

`mode='same'` mantiene la lunghezza dell'output uguale alla lunghezza del segnale $T$, centrando il risultato.

Il valore $C_i(\tau)$ è alto quando il segnale intorno al campione $\tau$ assomiglia al template: un blocco piatto di potenza $P_i$ per $D_i$ minuti.

**Rilevamento picchi:**

Si usano `scipy.signal.find_peaks` con:
- `height=0.6` (soglia di correlazione normalizzata)
- `distance = \lfloor D_i / 2 \rfloor` (distanza minima tra picchi: evita che lo stesso evento generi più picchi adiacenti)

**Assegnazione greedy:**

I picchi vengono ordinati per $C_i(\tau)$ decrescente. Per ogni picco:
- Si calcola la finestra di attivazione: $[\tau - D_i/2,\; \tau - D_i/2 + D_i)$
- Se la finestra non si sovrappone a nessuna attivazione già assegnata (`assigned[]`), si marca come ON
- Altrimenti si scarta (first-come-first-served)

I device vengono processati in **ordine decrescente di potenza** (`P_i` più alto per primo): i device più grandi hanno template con valori assoluti più alti e quindi correlazioni più forti, quindi vengono assegnati con priorità.

### Esempio numerico

Segnale (8 campioni, Boiler da 2000W per 3 minuti, poi 78 minuti di rumore):

```
w = [50, 50, 2000, 2000, 2000, 60, 45, 55, ...]
```

Template Boiler: `h = [2000, 2000, 2000, ...]` (45 campioni), $\|h\| = 2000 \cdot \sqrt{45} \approx 13416$.

Cross-correlazione centrata a $\tau=4$ (centro del blocco ON):
$$C(\tau=4) = \frac{2000 \cdot 2000 \times 3}{13416} \approx \frac{12\,000\,000}{13416} \approx 894$$

Questo valore è alto rispetto al valore in zone piatte (50W × 3 / 13416 ≈ 0.011).

Dopo normalizzazione per il massimo teorico ($P_i \cdot \sqrt{D_i}$), il picco a $\tau=4$ supera la soglia 0.6 e viene rilevato come attivazione.

### Limiti

- Funziona **solo** per device con consumo piatto e durata fissa:
  - Lavatrice ✓, Boiler ✓, Auto elettrica ✓
  - Frigorifero ✗ (cicla con duty cycle variabile, forma non rettangolare)
  - Microonde ✗ (durata troppo breve: ~5 minuti, template troppo corto)
- Non può rilevare due attivazioni dello stesso device che si sovrappongono
- La soglia `corr_threshold=0.6` è critica: troppo alta → falsi negativi; troppo bassa → falsi positivi
- **Bug noto risolto**: `dev_array = result[dev.name].to_numpy(dtype=float)` restituiva un array read-only; corretta con `.copy()`

---

## 7. Approccio E — Event-based con prior bayesiano

**File:** `scripts/nilm/approach_event_prior.py`

### Idea

Come l'Approccio A, ma l'assegnazione di ogni evento a un device non avviene per minima distanza — avviene tramite uno **score bayesiano** che incorpora tre fattori:
1. Quanto è plausibile che quel salto provenga da quel device (likelihood gaussiana)
2. Se il device è presente in casa (prior dall'inventario)
3. Quanto spesso viene usato in una settimana (frequenza dal survey)

### Matematica

**Rilevamento eventi**: identico all'Approccio A.

**Score per ogni device $d$ dato l'evento $|\Delta w|$:**

$$\text{score}(d, |\Delta w|) = \mathcal{L}(d, |\Delta w|) \cdot \pi_d \cdot f_d$$

Dove:

**Likelihood gaussiana:**
$$\mathcal{L}(d, |\Delta w|) = \exp\left(-\frac{1}{2}\left(\frac{|\Delta w| - P_d}{\sigma_d}\right)^2\right)$$

con $\sigma_d = \sigma_{\text{factor}} \cdot P_d$ e $\sigma_{\text{factor}} = 0.3$ (default).

È la densità della gaussiana $\mathcal{N}(P_d,\, \sigma_d^2)$ valutata in $|\Delta w|$, senza la costante di normalizzazione (irrilevante per il confronto tra device). È proporzionale alla probabilità che un device di potenza $P_d$ generi un salto di $|\Delta w|$.

**Prior dall'inventario:**
$$\pi_d = \begin{cases} 1.0 & \text{se device presente in casa} \\ 0.05 & \text{se device assente} \end{cases}$$

Un device assente ha uno score 20 volte inferiore a parità di likelihood.

**Fattore di frequenza:**
$$f_d = \min\left(\frac{\text{freq\_per\_week}_d}{7}, 1.0\right)$$

Un device usato ogni giorno (freq=7) ha $f_d = 1.0$; uno usato 2 volte a settimana ha $f_d \approx 0.29$.

**Assegnazione:** vince il device con score massimo:
$$d^* = \argmax_d \;\text{score}(d, |\Delta w|)$$

**Pairing ON/OFF**: identico all'Approccio A.

### Esempio numerico

Evento: `|Δw| = 1600W`. Tre device presenti: Lavatrice (1800W), Piano cottura (1500W), Boiler (2000W). Tutti presenti (`π=1.0`).

**Lavatrice** ($P=1800$, $\sigma=540$, freq=3/week):
$$\mathcal{L} = \exp\left(-\frac{(1600-1800)^2}{2 \cdot 540^2}\right) = \exp(-0.068) \approx 0.934$$
$$f = 3/7 \approx 0.429 \quad \Rightarrow \quad \text{score} = 0.934 \cdot 1.0 \cdot 0.429 \approx 0.401$$

**Piano cottura** ($P=1500$, $\sigma=450$, freq=7/week):
$$\mathcal{L} = \exp\left(-\frac{(1600-1500)^2}{2 \cdot 450^2}\right) = \exp(-0.025) \approx 0.976$$
$$f = 7/7 = 1.0 \quad \Rightarrow \quad \text{score} = 0.976 \cdot 1.0 \cdot 1.0 \approx 0.976$$

**Boiler** ($P=2000$, $\sigma=600$, freq=7/week):
$$\mathcal{L} = \exp\left(-\frac{(1600-2000)^2}{2 \cdot 600^2}\right) = \exp(-0.222) \approx 0.801$$
$$f = 1.0 \quad \Rightarrow \quad \text{score} = 0.801$$

**Vincitore: Piano cottura** (score 0.976). Con l'Approccio A invece avrebbe vinto la **Lavatrice** (distanza 200W vs 100W del piano cottura) — lo stesso risultato in questo caso, ma il processo è diverso: l'A ignora la frequenza d'uso.

Ora il Piano cottura **assente** (`π=0.05`):
$$\text{score}_{\text{assente}} = 0.976 \cdot 0.05 \cdot 1.0 = 0.049$$

Con il device assente, vincerebbero Boiler (0.801) o Lavatrice (0.401).

### Differenza chiave rispetto all'Approccio A

| | Approccio A | Approccio E |
|---|---|---|
| Criterio assegnazione | Minima distanza $\|P_d - \|\Delta w\|\|$ | Score bayesiano |
| Device considerati | Solo presenti | Tutti (con penalità agli assenti) |
| Ruolo della frequenza | Ignorato | Modulato come $f_d$ |
| Forma della funzione | Distanza lineare | Gaussiana (decadimento esponenziale) |
| Tie-breaking stessa P | Primo nella lista | Primo nella lista (stesso limite) |

---

## 8. Confronto e risultati reali

I risultati seguenti sono calcolati su 6 IMEI con tutti e 5 gli approcci (run eseguita il 2026-04-07). Il template non compare per un bug risolto successivamente.

| Approccio | MAE (W) ↓ | Energy error % ↓ | Temporal consistency ↑ |
|-----------|-----------|------------------|------------------------|
| **fhmm** | **83.6** | **16.1%** | **0.81** |
| event_prior | 189.4 | 66.2% | 0.30 |
| event | 192.9 | 72.0% | 0.29 |
| hmm | 3011.1 | 1331.0% | 0.43 |

**MAE (W)**: errore medio assoluto di ricostruzione $\frac{1}{T}\sum_t |w(t) - \hat{w}(t)|$.

**Energy error %**: $\frac{|\sum_t w(t) - \sum_t \hat{w}(t)|}{\sum_t w(t)} \times 100$. L'HMM al 1331% significa che stima 14× più energia di quella reale.

**Temporal consistency**: frazione di blocchi ON con durata in $[d_{\min}/2,\; d_{\text{typical}} \times 3]$. Il FHMM a 0.81 produce attivazioni di durate plausibili; l'event a 0.29 genera molti blocchi troppo brevi o troppo lunghi.

---

## 9. Limitazioni strutturali comuni

### Il problema è sottodeterminato

Con un segnale scalare e $N=16$ device, l'equazione $w(t) = \sum_i P_i x_i(t)$ ha infinite soluzioni per ogni $t$. I vincoli introdotti (soglie, template, prior) selezionano **una** soluzione plausibile ma non necessariamente quella reale.

### Potenze generiche vs. potenze reali

I $P_i$ in `DEVICE_KNOWLEDGE_BASE` sono valori medi da datasheet, non misure dell'apparecchio specifico in quella casa. Un frigorifero può consumare tra 80W e 250W a seconda del modello, dell'età e della temperatura ambiente. Questo introduce un errore sistematico in tutti gli approcci.

### Assenza di ground truth

Non esistono etichette che indichino quando ogni device è effettivamente acceso. Le metriche del benchmark (MAE, energy error) misurano la **coerenza interna** della soluzione rispetto al segnale aggregato, non l'accuratezza rispetto alla realtà.

### Co-attivazioni

Se lavatrice e boiler si accendono insieme:
- **Event-based**: vede un unico salto di ~3800W e lo assegna come se fosse un singolo device
- **FHMM**: può in teoria gestirle, ma solo se converge alla soluzione corretta (dipende dall'ordine di aggiornamento)
- **HMM**: completamente cieco, lavora su residui sequenziali

### Device identici o confondibili

Frigorifero (150W), Televisori (100W), Congelatore (100W) hanno potenze simili o identiche. Con il solo segnale aggregato non è possibile distinguerli senza informazioni aggiuntive (es. pattern temporale, orari tipici di utilizzo).
