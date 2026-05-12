# Approccio Gurobi per la disaggregazione NILM

## Obiettivo

Questo documento spiega come è stato implementato l'approccio `gurobi` nel progetto e qual è l'idea generale del modello di ottimizzazione.

Il file di riferimento è [scripts/nilm/approach_gurobi.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/nilm/approach_gurobi.py).

L'approccio tratta la disaggregazione come un problema di ottimizzazione combinatoria: dato il segnale aggregato minuto per minuto, vogliamo stimare per ogni dispositivo se è acceso o spento in ciascun istante, imponendo vincoli temporali realistici.

In forma compatta, il modello cerca una decomposizione del segnale:

$$
y(t) \approx \sum_{i=1}^{N} P_i x_i(t)
$$

dove:

- $y(t)$ è la potenza aggregata osservata al minuto $t$
- $P_i$ è la potenza tipica del dispositivo $i$
- $x_i(t) \in \{0,1\}$ indica se il dispositivo $i$ è acceso o spento

La difficoltà è che il problema è fortemente ambiguo: molte combinazioni di dispositivi possono spiegare lo stesso valore aggregato. Per questo il modello introduce vincoli aggiuntivi su frequenza di accensione, durata minima, durata massima e plausibilità temporale.

## Dove entra nella pipeline

L'approccio viene richiamato da [scripts/run_disaggregation.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/run_disaggregation.py), dove `approach_gurobi` è registrato in `APPROACH_MAP` con chiave `gurobi`.

La funzione pubblica è:

```python
run(signal, devices, lambda_activation=10.0, prob_threshold=0.10,
    max_duration_factor=2.0, time_limit=60.0)
```

Input:

- `signal`: serie temporale `pandas` già preprocessata e ricampionata a 1 minuto
- `devices`: lista di `DeviceProfile` caricati da [scripts/nilm/devices.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/nilm/devices.py)

Output:

- un dizionario `device_name -> pd.Series` con la potenza stimata per ciascun dispositivo

## Dati e ipotesi usate dal modello

Ogni dispositivo è rappresentato da un profilo con informazioni statiche:

- `p_typical_w`: potenza tipica
- `dur_min_min`: durata minima di una singola attivazione
- `dur_typical_min`: durata tipica
- `frequency_per_week`: frequenza d'uso attesa
- `prior_weight`: indica se il dispositivo è dichiarato presente nell'abitazione
- `always_on` e `duty_cycle`: per dispositivi quasi sempre attivi come frigorifero e congelatore

Nel codice vengono ottimizzati solo i dispositivi con `prior_weight >= 1.0`, cioè quelli dichiarati presenti. I dispositivi assenti vengono restituiti direttamente come serie nulle.

Questa scelta riduce drasticamente la dimensione del problema e sfrutta l'inventario raccolto dal survey come vincolo forte.

## Segmentazione settimanale

Il modello non risolve l'intera serie temporale in un unico problema. La funzione `run()` divide il segnale per settimana di calendario e risolve un problema separato per ciascuna settimana.

Questo è importante per due motivi:

- un problema su tutto l'orizzonte temporale diventerebbe troppo grande per il solver
- il vincolo di tempo `time_limit` è applicato per singola settimana, quindi il runtime resta più controllabile

Per ogni settimana si estraggono:

- `y`: vettore numerico dei consumi aggregati
- `timestamps`: indice temporale della settimana
- `T`: numero di minuti validi nella settimana

La risoluzione vera e propria avviene in `_solve_week(...)`.

## Variabili decisionali

Per ogni dispositivo $i$ e minuto $t$ vengono introdotte le variabili:

- $x_{i,t} \in \{0,1\}$: dispositivo acceso/spento
- $u_{i,t} \in \{0,1\}$: evento di accensione al minuto $t$
- $d_{i,t} \in \{0,1\}$: evento di spegnimento al minuto $t$

Non vengono più introdotte variabili di slack per la ricostruzione. L'errore entra direttamente nella funzione obiettivo attraverso un termine quadratico.

## Funzione obiettivo

L'idea del modello è minimizzare l'errore di ricostruzione del segnale, con una penalità aggiuntiva per evitare troppe accensioni inutili.

La forma desiderata è:

$$
\min \sum_t \left(\sum_i P_i x_i(t) - y(t)\right)^2 + \sum_i \lambda_i \sum_t u_{i,t}
$$

dove il peso di attivazione per dispositivo è costruito come:

$$
\lambda_i = \frac{\text{lambda\_activation}}{\max(\text{freq\_week}_i, 0.5)/7}
$$

Quindi i dispositivi usati raramente vengono penalizzati di più a ogni nuova accensione.

### Nota sull'implementazione attuale

Nel codice corrente la parte quadratica dell'errore viene costruita con `gp.QuadExpr()` e l'obiettivo effettivamente passato a Gurobi è la somma di:

- errore quadratico di ricostruzione
- penalità sulle attivazioni

Quindi, a differenza della formulazione precedente, la penalità sulle attivazioni entra davvero nell'obiettivo e non sono necessarie variabili ausiliarie per rappresentare l'errore.

## Vincoli implementati

### 1. Esclusione nei punti con dato mancante

Se il segnale aggregato è `NaN` in un certo minuto, tutte le variabili `x[i,t]` vengono forzate a zero tramite upper bound.

Questo evita che il modello inventi assorbimenti in punti dove il dato non esiste.

### 2. Vincolo di plausibilità basato su schedule

Per ogni dispositivo viene calcolata una funzione $s_i(t) \in [0,1]$ che rappresenta quanto è plausibile che quel dispositivo sia acceso nel minuto $t$.

La funzione `_compute_schedule(...)` usa due casi:

- per dispositivi always-on con `duty_cycle`, restituisce una probabilità costante
- per gli altri, usa una probabilità piatta derivata da frequenza settimanale e durata tipica

La stima piatta è:

$$
p_i = \frac{\text{uses\_per\_day}_i \cdot \text{dur\_typical}_i}{1440}
$$

con:

$$
\text{uses\_per\_day}_i = \frac{\text{frequency\_per\_week}_i}{7}
$$

Se $s_i(t) < \text{prob\_threshold}$, allora il dispositivo non può essere acceso:

$$
x_{i,t} = 0
$$

Nel codice questo è implementato ponendo `x[i, t].UB = 0.0`.

Osservazione pratica: siccome per molti dispositivi lo schedule è piatto nel tempo, questo vincolo non modella davvero l'ora del giorno, ma solo una plausibilità media complessiva.

### 3. Vincolo di bilancio degli stati

Il modello collega lo stato acceso/spento ai fronti di accensione e spegnimento con:

$$
x_{i,t} - x_{i,t-1} = u_{i,t} - d_{i,t}
$$

Questo significa:

- se il dispositivo passa da 0 a 1, allora `u=1`
- se passa da 1 a 0, allora `d=1`
- se lo stato non cambia, allora `u=d=0`

Per $t=0$ il modello inizializza:

$$
u_{i,0} = x_{i,0}, \qquad d_{i,0} = 0
$$

quindi una accensione iniziale al primo campione viene contata come attivazione.

### 4. Mutua esclusione degli eventi

Per evitare che nello stesso istante un dispositivo si accenda e si spenga insieme, viene imposto:

$$
u_{i,t} + d_{i,t} \le 1
$$

### 5. Durata minima di attivazione

Se un dispositivo si accende al tempo $t$, deve rimanere acceso almeno `dur_min_min` minuti.

Definendo:

$$
a_i = \lceil \text{dur\_min\_min}_i \rceil
$$

il vincolo diventa:

$$
\sum_{\tau=t}^{t+a_i-1} x_{i,\tau} \ge a_i u_{i,t}
$$

Se `u[i,t]=1`, la somma dei successivi $a_i$ stati deve essere almeno $a_i$, quindi devono essere tutti accesi.

Questo vincolo è molto utile per eliminare accensioni spurie di 1 o 2 minuti su carichi che nella realtà hanno cicli molto più lunghi.

### 6. Durata massima tramite rolling window

Per evitare che un dispositivo resti acceso troppo a lungo, viene introdotto un vincolo su finestre scorrevoli.

Definendo:

$$
B_i = \left\lceil \text{dur\_typical\_min}_i \cdot \text{max\_duration\_factor} \right\rceil
$$

si impone:

$$
\sum_{\tau=t}^{t+B_i} x_{i,\tau} \le B_i
$$

Poiché la finestra contiene $B_i + 1$ istanti, il vincolo impedisce che tutti siano uguali a 1. In pratica vieta run ON più lunghe di circa $B_i$ minuti consecutivi.

Questo è un modo lineare e semplice per imporre una durata massima senza introdurre logica più complicata sul tempo di spegnimento.

## Interpretazione operativa del risultato

Una volta risolto il problema, il codice legge i valori di `x[i,t]`, li arrotonda e li converte in potenza stimata:

$$
\hat{p}_i(t) = P_i x_{i,t}
$$

Quindi l'output non è una potenza continua stimata liberamente, ma una serie a due livelli:

- `0`
- `p_typical_w`

Questo semplifica molto il problema ma introduce anche una limitazione forte: dispositivi con assorbimento modulato o multi-stato vengono approssimati come puramente ON/OFF a potenza costante.

## Perché l'approccio è un OR-MIQP

Questo approccio rientra nella categoria di metodi di Operations Research perché formula la NILM come un problema di ottimizzazione con:

- variabili binarie
- vincoli lineari
- obiettivo quadratico

Quindi il problema è un MIQP, più precisamente un Mixed-Integer Quadratic Program con variabili binarie e vincoli lineari. Gurobi esplora in modo intelligente lo spazio combinatorio delle soluzioni ammissibili, invece di usare regole euristiche locali come negli approcci event-based.

Il vantaggio principale è che il modello considera tutti i dispositivi simultaneamente e può spiegare il segnale come combinazione di più carichi attivi nello stesso minuto.

## Scelte implementative principali

### Uso della potenza tipica al posto di una banda di potenza

Nel modello entra solo `p_typical_w`. I campi `p_min_w` e `p_max_w` dei device profile non vengono usati nella formulazione attuale.

Questo rende il modello più semplice e più robusto numericamente, ma perde informazione quando i consumi reali oscillano molto attorno al valore medio.

### Ottimizzazione solo sui device presenti

La funzione `run()` elimina subito i device con `prior_weight < 1.0` dal set di variabili decisionali. È una scelta forte ma pragmatica: il survey household viene usato come filtro duro, non come prior morbido.

### Risoluzione settimana per settimana

Serve a mantenere il problema risolvibile in tempi ragionevoli. Un problema minuto per minuto con molti dispositivi e più settimane diventerebbe rapidamente pesante.

### Gestione robusta dei failure

Se Gurobi non è disponibile, se non trova una soluzione, o se genera un'eccezione, l'implementazione restituisce serie nulle per quel blocco temporale invece di far fallire l'intera pipeline.

Questo è utile in produzione o durante benchmark batch, perché permette di completare il run e analizzare comunque gli altri approcci.

## Limiti dell'implementazione attuale

### Schedule molto semplificato

Per quasi tutti i dispositivi lo schedule è costante nel tempo, quindi non stai usando davvero informazione su fasce orarie, solo una media giornaliera implicita.

### Modello binario a potenza costante

Un forno, una lavatrice o un climatizzatore spesso non assorbono potenza costante per tutta la durata. Qui invece ogni device è rappresentato da una singola potenza tipica.

### Nessuna dipendenza tra dispositivi

Il modello non codifica co-occorrenze o esclusioni tra device, per esempio forno e microonde, o climatizzatore e stagione, o ricarica auto solo di notte.

### Possibile costo computazionale elevato

Anche con la segmentazione settimanale, il numero di variabili e vincoli cresce rapidamente con il numero di dispositivi presenti e con la lunghezza della finestra osservata.

## Riassunto pratico

In termini semplici, l'approccio Gurobi fa questo:

1. prende il segnale aggregato di una settimana
2. decide per ogni device e per ogni minuto se il device è acceso o spento
3. sceglie la combinazione che minimizza un errore quadratico di ricostruzione, con penalità sulle accensioni
4. scarta soluzioni irrealistiche imponendo durata minima, durata massima e coerenza delle transizioni
5. restituisce una traccia di potenza per ogni dispositivo presente

È un approccio più globale e strutturato rispetto agli approcci euristici, ma richiede più tempo computazionale e una formulazione accurata del modello.

## Possibili estensioni

Se in futuro vuoi migliorarlo, le direzioni più naturali sono:

1. aggiungere davvero `obj_penalty` all'obiettivo
2. usare schedule dipendenti dall'ora del giorno invece che probabilità piatte
3. modellare device multi-stato o con bande di potenza
4. usare `prior_weight` come penalità morbida invece che come filtro duro
5. aggiungere vincoli specifici per classe di device

## File coinvolti

- [scripts/nilm/approach_gurobi.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/nilm/approach_gurobi.py): implementazione del modello
- [scripts/nilm/devices.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/nilm/devices.py): knowledge base dei dispositivi
- [scripts/run_disaggregation.py](/home/ciarpa/Desktop/github/work/signal-disaggregation/scripts/run_disaggregation.py): orchestrazione della pipeline