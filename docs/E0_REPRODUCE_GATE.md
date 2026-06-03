# E0 — reproduce-gate: `electricity_nips` + CRPS-sum

> **In una riga.** Rifacciamo M3 TimeGrad sullo split **ufficiale** del benchmark
> (`electricity_nips`) e misuriamo il **CRPS-sum**, così possiamo mettere il nostro
> numero accanto alla tabella pubblicata di TimeGrad/CSDI e chiudere onestamente il
> limite «non usiamo lo split del benchmark».

## Perché serve

Tutti i nostri numeri headline su Electricity girano sul file grezzo **LSTNet**
(`electricity.txt.gz`, 321 serie) con il **nostro** split a rapporti `[0.7/0.1/0.2]`.
Ottimo per iterare in fretta e per l'EDA, ma i numeri **CRPS-sum** dei paper di
diffusione (TimeGrad, CSDI, ScoreGrad) sono riportati sul dataset GluonTS
**`electricity_nips`**: **370** serie, il **confine train/test ufficiale**, e un test a
**7 finestre rolling** con `prediction_length = 24`. Confrontare il nostro `CRPS` (scala
~10²) con il loro `CRPS_sum` (scala ~10⁻²) sarebbe un errore di categoria. E0 elimina
l'ambiguità: stesso modello, **split del paper**, **metrica del paper**.

L'idea chiave è isolare **una sola variabile**. La config E0 tiene `H = 168` e
`τ = 24` **identici** alla config LSTNet: cambia *solo* la sorgente dati e lo split.
Così il `CRPS_sum` di E0 risponde a una domanda netta — *quanta parte della distanza dai
numeri pubblicati era semplicemente lo split, e quanta resta da imputare a tuning /
architettura / numero di seed?*

## Cosa cambia (e cosa no)

|                | LSTNet (headline)            | `electricity_nips` (E0)              |
|----------------|------------------------------|--------------------------------------|
| serie (D)      | 321                          | **370**                              |
| split          | nostri rapporti `0.7/0.1/0.2`| **confine ufficiale** del benchmark  |
| protocollo test| finestre sliding `stride=1`  | **7 finestre rolling**, `stride = τ` |
| sorgente       | `laiguokun .txt.gz`          | `gluonts.get_dataset("electricity_nips")` |
| metrica extra  | —                            | **`CRPS_sum`** (`--crps-sum`)        |
| `H`, `τ`       | 168, 24                      | **168, 24** (identici)               |
| modello, seed  | TimeGrad, 42                 | TimeGrad, 42 (identici)              |

**Invariato a valle.** `src/data/loader.py` vede `source.loader:
gluonts_electricity_nips` e delega a `src/data/electricity_nips.py`, che restituisce lo
**stesso** contratto `ForecastDataset`. Quindi `run_timegrad.py` non cambia di una riga:
`to_gluonts_multivariate("train")` per il fit, `iter_windows("test", stride=τ)` per le 7
finestre. È la promessa del contratto dati (plan §4.5): *cambiare dataset = cambiare la
config*.

## La metrica: `CRPS_sum`

`CRPS_sum` è la metrica multivariata standard della letteratura, calcolata **esattamente**
come `gluonts.evaluation.MultivariateEvaluator` con `target_agg_funcs={"sum": np.sum}`:

```
CRPS_sum = mean_q [ ( Σ_t 2·ρ_q( y_t^Σ , q-esimo quantile di F_t^Σ ) ) / Σ_t |y_t^Σ| ]
```

dove `y_t^Σ = Σ_d y_{t,d}` è la **somma sui canali** (si valuta il forecast *congiunto*
dell'aggregato, non le marginali), `ρ_q` è la pinball loss, `q` scorre i **decili**
`0.1…0.9`. Due proprietà la rendono confrontabile tra paper: la somma-sui-canali e la
**normalizzazione per `Σ|y^Σ|`**, che la rende *scale-free* (Electricity cade intorno a
~0.02, non a ~10² come il nostro `CRPS` per-posizione).

> ⚠️ **`CRPS_sum` e `CRPS` non sono sulla stessa scala** e non vanno mai confrontati
> direttamente né messi sullo stesso asse. Sono due metriche diverse: `CRPS` per-posizione
> (la nostra colonna headline) e `CRPS_sum` normalizzato (per il confronto col paper).

Implementazione: `src/eval/metrics.py::crps_sum` (eager) + accumulatori in
`ForecastEvaluator` (streaming). È **opt-in** (`evaluate_forecast(..., crps_sum=True)`,
`--crps-sum` sul runner): di default ogni run produce il dict identico a prima, quindi
nessun numero già in banca viene toccato. Coperta dai test in `tests/test_metrics.py`
(eager == reference GluonTS-style, streaming == eager, forecast perfetto → 0).

## Come si lancia

E0 vive **solo** nell'ambiente pesante (gluonts): in locale non è eseguibile (FASE B —
gluonts non installato). Sul box GPU / Colab:

```bash
# 0) smoke del contratto (scarica electricity_nips la prima volta, ~1 min):
python -m src.data.electricity_nips configs/data_electricity_nips.yaml
#    atteso: test (336, 370) -> 7 finestre (H=168 + 7·24), stride=24

# 1) M3 TimeGrad sullo split ufficiale + metrica pubblicata:
python -m experiments.run_timegrad \
    --config configs/data_electricity_nips.yaml \
    --chunk 1 --crps-sum --epochs 50 --samples 100 --diff-steps 100 --device cuda
```

In Colab è la sezione **P2.6** di `notebooks/colab_electricity_all.ipynb` (dentro la
PARTE 2, che installa già lo stack gluonts), con BLOCCO DI RECUPERO e backup su Drive come
le altre gambe. La riga finisce nella registry con `dataset = electricity_nips` e la
colonna `CRPS_sum`.

## Come riproduciamo le 7 finestre rolling (e l'unico caveat)

GluonTS valuta 7 finestre il cui target di 24 passi *tassella* gli **ultimi 7·τ passi**
della serie completa, ciascuna condizionata sulla storia che la precede. Lo riproduciamo
esponendo come split di test la **coda di lunghezza `H + 7·τ`** e forzando `stride = τ`:
`iter_windows` produce così **esattamente 7** finestre non sovrapposte, target-per-target
identiche al benchmark, ciascuna con `H` passi di contesto immediatamente precedente.

**Unico caveat, documentato.** GluonTS dà a ogni finestra la sua storia *intera* e il
modello la tronca a `context_length` tramite i lag; il nostro contratto le dà *esattamente*
`H` passi di contesto. Con `H = 168` ≥ span dei lag orari, per TimeGrad le due cose
coincidono. È una differenza di protocollo di secondo ordine, non un confronto sbagliato.

## Numeri di riferimento (per leggere il risultato)

Sul `electricity_nips` ufficiale, dai paper (370-dim, split ufficiale, `CRPS_sum`):

| Modello   | `CRPS_sum` pubblicato (≈) | fonte                         |
|-----------|---------------------------|-------------------------------|
| TimeGrad  | **~0.021**                | Rasul et al. 2021 (Tab. 1)    |
| CSDI      | ~0.017–0.018              | Tashiro et al. 2021           |
| GP-Copula | ~0.024                    | Salinas et al. 2019           |

> I valori vanno ri-verificati sulla tabella esatta del paper prima di citarli nel
> report; qui servono solo come ordine di grandezza atteso (~10⁻²).

**Come interpretare.** Se il nostro `CRPS_sum` E0 si avvicina a ~0.02, gran parte della
distanza headline dai numeri pubblicati era **lo split** (321 vs 370 serie + sliding vs
rolling), non un difetto del modello. Se resta molto più alto, la differenza è imputabile
a **budget/tuning/seed** (noi: 1 seed, 50 epoche, niente tuning esteso) — anch'esso un
limite onesto da dichiarare. In entrambi i casi E0 trasforma un limite vago («split
diverso») in un numero.

## Cosa E0 *non* dimostra

- Non è un confronto a parità di **budget** col paper (epoche, seed, search): un singolo
  seed e 50 epoche non sono il loro protocollo completo.
- Non tocca M0–M2/M4: è il leg M3 sullo split ufficiale. Estenderlo a M2 DeepAR è banale
  (stessa config, `run_deepar.py`) se serve un secondo punto di ancoraggio.
- Il caveat del contesto (sopra) resta: fedele per TimeGrad con `H` ampio, non una replica
  bit-per-bit del rolling di GluonTS.
