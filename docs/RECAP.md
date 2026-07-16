## Panoramica del progetto

Il progetto affronta il problema del **NILM (Non-Intrusive Load Monitoring)**: dato il consumo
energetico aggregato di un'abitazione misurato da un singolo dispositivo IoT, si stima **quali
elettrodomestici sono accesi in ogni istante**, senza etichette di ground truth.

Il modello matematico è:

```
w_medio(t) ≈ Σ_i  P_i · x_i(t)
```

dove `P_i` è la potenza tipica del dispositivo i-esimo e `x_i(t) ∈ {0,1}` indica se è acceso.