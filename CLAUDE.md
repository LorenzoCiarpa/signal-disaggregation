# Project Guidance

This file provides context and guidance for working with this project.

## Project Overview

This project is written primarily in python.

## Tools and Commands

- Use "python" or "python3" to run scripts
- Use "pip install -r requirements.txt" to install dependencies
- Use "pytest" for running tests (if configured)
- Use "black" or "autopep8" for code formatting (if configured)
- Use "pylint" or "flake8" for linting (if configured)

## Development Workflow

1. Make changes to .py files
2. Install/update dependencies if needed
3. Run linters if configured
4. Run tests with "pytest" or "python -m unittest"
5. Verify functionality with test scripts

## Coding Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Write docstrings for functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

## Testing Approach

- Write tests using pytest or unittest
- Place tests in test_*.py files
- Run "pytest" or "python -m unittest discover"
- Use fixtures for test setup
- Aim for good test coverage

## Environment Persistence

This project has a persistent environment configured via `CLAUDE_ENV_FILE` (`/etc/sandbox-persistent.sh`).

According to [Claude Code Documentation](https://code.claude.com/docs/en/settings#bash-tool-behavior):

> **CLAUDE_ENV_FILE**
>
> If set, this file will be sourced before each Bash command execution. This allows environment variables to persist across multiple Bash tool invocations.

The sandbox environment provides persistent environment variable storage across all shell sessions.

- Environment variables stored in `/etc/sandbox-persistent.sh` persist across all bash invocations
- The `CLAUDE_ENV_FILE` environment variable points to `/etc/sandbox-persistent.sh`
- Use `echo "export VAR_NAME=value" >> /etc/sandbox-persistent.sh` to add persistent variables
- Useful for tool installations (nvm, sdkman, etc.) that modify PATH or environment variables

## Critical: Shell Completions Must NOT Be in CLAUDE_ENV_FILE

**NEVER add shell completion scripts to the persistent environment file.**

Shell completion scripts (like `bash_completion` for NVM, SDKMAN, etc.) **will completely break the bash tool** when sourced via `CLAUDE_ENV_FILE`.

### Why Completions Break

`CLAUDE_ENV_FILE` is sourced **before every single bash command execution**, not just during shell initialization. Completion scripts rely on special variables (`COMP_WORDS`, `COMP_CWORD`, `COMPREPLY`) that only exist during tab-completion contexts, not during normal command execution.

### WRONG - Will Break Bash

```bash
# DO NOT ADD THESE TO /etc/sandbox-persistent.sh
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
[[ -s "$SDKMAN_DIR/etc/bash_completion.sh" ]] && source "$SDKMAN_DIR/etc/bash_completion.sh"
```

### CORRECT - Only Load Core Functionality

```bash
# ONLY add the main initialization scripts
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh"
```

### Symptoms of Broken Shell

When completion scripts are incorrectly added to `CLAUDE_ENV_FILE`:
- All bash commands return no output (silent failure)
- `echo`, `pwd`, and other basic commands produce no results
- The bash tool becomes completely unusable

### Solution

If you accidentally added completion scripts and broke the shell:
1. Remove the completion line(s) from `/etc/sandbox-persistent.sh`
2. Exit and restart the Claude Code session
3. Verify with `echo "test"` that bash works again

### IMPORTANT: Using the Bash Tool

**When using the Bash tool, in the case of not finding the tool in the PATH, try using a fresh login shell to ensure the persistent environment is properly loaded:**

- Use `bash -l -c "your-command"` instead of running commands directly
- This ensures `/etc/sandbox-persistent.sh` is sourced and PATH modifications are honored
- Example: `bash -l -c "java -version"` instead of `java -version`
- This is critical when tools like sdkman, nvm, or other environment managers modify PATH

**Why this is necessary:**
- Shell snapshots may contain cached environment state from before tools were installed
- Login shells always source the persistent environment file fresh, ensuring latest configuration
- This guarantees that agent's extension of the environment file, such as `CLAUDE_ENV_FILE` for Claude, is properly honored

Example - persisting nvm installation:
```bash
# After installing nvm
echo 'export NVM_DIR="$HOME/.nvm"' >> /etc/sandbox-persistent.sh
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /etc/sandbox-persistent.sh

# Then use login shells to access it
bash -l -c "node --version"
```

Example - persisting sdkman installation:
```bash
# After installing sdkman and Java
echo 'export SDKMAN_DIR="$HOME/.sdkman"' >> /etc/sandbox-persistent.sh
echo '[[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh"' >> /etc/sandbox-persistent.sh

# Then use login shells to access it
bash -l -c "java -version"
bash -l -c "sdk current"
```

## Network access

There is a firewall in place to restrict outbound network access. If you need http/https access to
an external service, request it, specifying the domain and port.

### Docker network access

You have access to a Docker daemon in this environment. You can access published ports on "localhost"
because it is included in the shell's "no proxy" configuration. For direct access to container ports,
you must add the container's network to the "no proxy" configuration.

## Additional Notes

- Always read relevant files before making changes
- Run tests after making modifications
- Follow the existing code structure and patterns
- Ask for clarification if project requirements are unclear
- You have sudo permissions, so you can install necessary packages
- npm, pip and uv are already available for package management

---

## NILM Signal Disaggregation — Descrizione del Task

### Obiettivo generale

Il progetto affronta il problema della **Signal Disaggregation** (o NILM — Non-Intrusive Load
Monitoring): dato il consumo energetico aggregato di un'abitazione misurato da un unico
dispositivo IoT, si vuole capire **quali elettrodomestici sono accesi in ogni istante di tempo**,
senza etichette di ground truth.

### Struttura dei dati

I dati si trovano nella cartella `json/` come file JSON per ogni abitazione, identificata dall'IMEI
del dispositivo di misura installato. Ogni file contiene un array di record, uno per minuto circa:

```json
{
  "_id":          { "$numberLong": "1763018881" },   // epoch Unix (secondi)
  "epoch_valid":  true,                              // flag qualità dato
  "w_minimo":     50,                                // watt minimi nell'intervallo
  "t_min_offset": 48,                                // offset secondi del minimo
  "t_min_absolute": { "$numberLong": "1763018929" }, // timestamp assoluto del minimo
  "w_massimo":    50,                                // watt massimi nell'intervallo
  "t_max_offset": 48,
  "t_max_absolute": { "$numberLong": "1763018929" },
  "w_medio":      50,                                // watt medi (metrica principale)
  "w_istantaneo": 50,                                // watt istantanei
  "granularity":  61                                 // durata intervallo in secondi
}
```

Grandezze chiave:
- **`w_medio`**: consumo medio nell'intervallo → metrica principale per la disaggregazione
- **`w_massimo`** / **`w_minimo`**: utili per rilevare picchi e fluttuazioni rapide
- **`epoch_valid`**: se `false` il record va scartato
- **`granularity`**: normalmente 60-61 s; variazioni indicano anomalie

### Dispositivi presenti nelle abitazioni

Per ogni IMEI è disponibile un file `device_usage_by_imei/{imei}.json` con l'inventario degli
elettrodomestici dichiarato dagli utenti tramite survey:
- Frigorifero, Congelatore, Lavatrice, Asciugatrice, Lavastoviglie
- Forno, Piano cottura, Microonde, Boiler
- Climatizzatore, Televisori, Computer, Console
- Auto elettrica, ecc.

Non esistono etichette temporali: **non sappiamo quando ogni dispositivo è acceso**.

### Task di disaggregazione (obiettivo principale)

Dato il segnale aggregato `w_medio(t)`, si vuole stimare i contributi individuali dei dispositivi:

```
w_medio(t) ≈ Σ_i  P_i · x_i(t)
```

dove `P_i` è la potenza tipica del dispositivo i-esimo e `x_i(t) ∈ {0,1}` indica se è acceso.

Approcci possibili (da considerare in base alla qualità dei dati disponibili):
- **Metodi basati su eventi**: rilevamento di transizioni (accensione/spegnimento) tramite
  variazioni brusche di `w_medio`
- **Clustering / HMM**: modelli a stati nascosti per caratterizzare i pattern di consumo
- **Template matching**: confronto con firme energetiche note degli elettrodomestici
- **Factorial HMM**: modello con variabili latenti indipendenti per ogni dispositivo
- **Deep learning (seq2point, seq2seq)**: reti neurali addestrate su dataset pubblici (UK-DALE,
  REDD) e adattate ai dati locali tramite transfer learning

### Fasi del progetto

1. **Verifica qualità dati** (`analysis/verification_report.txt`)
   - Identificazione di spike anomali (> 10 000 W)
   - Rilevamento di buchi temporali
   - Confronto con le note qualitative in `NILM_dati.rtf`

2. **Pre-processing**
   - Filtraggio spike (> 10 000 W)
   - Riempimento/interpolazione dei buchi
   - Normalizzazione e allineamento temporale

3. **Feature engineering**
   - Derivata del segnale (δw/δt)
   - Statistiche su finestre scorrevoli (media, varianza)
   - Trasformata di Fourier per rilevare periodicità

4. **Disaggregazione**
   - Applicazione di uno o più degli approcci elencati sopra
   - Validazione qualitativa (confronto con orari tipici degli elettrodomestici)

5. **Visualizzazione e reporting**
   - Grafici per abitazione e per giorno
   - Stima della quota energetica per dispositivo

### Struttura cartelle output

```
analysis/
  {imei}/
    daily_plots/        <- grafici giornalieri w_max + w_medio (dati filtrati)
    spike_plots/        <- grafici giorni con picchi > 10 000 W
    spikes_{imei}.txt   <- lista istanti anomali
  verification_report.txt   <- verifica delle note in NILM_dati.rtf
```

### Script principali

| Script | Descrizione |
|--------|-------------|
| `scripts/analysis_nilm.py` | Analisi qualità dati, grafici, spike, report di verifica |
| `scripts/compute_stats.py` | Statistiche CSV per ogni file JSON |
| `scripts/plot_daily_traces.py` | Grafici giornalieri (versione precedente) |
| `scripts/plot_tracciato.py` | Visualizzazione singolo tracciato |

Eseguire l'analisi principale con:
```bash
python3 scripts/analysis_nilm.py
```
