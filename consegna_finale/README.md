# Probabilistic Time-Series Forecasting with Diffusion Models — pacchetto di consegna

> Progetto d'esame **PML** (Probabilistic Machine Learning) — Università di Trieste, Prof. Luca Bortolussi.
> Gruppo: **Giovanni Mason (SM3800158) · Lorenzo Di Bernardo (SM3800132) · Lorenzo Karol Gobbo (SM28A00018)** · Appello **10 giugno 2026**.
> Branch di consegna: `feature/port-ladder-exchange` (non ancora unito a `main`).

Questo è il pacchetto autosufficiente da consegnare. **Punto di partenza:** [`LEGGIMI_CONSEGNA.md`](LEGGIMI_CONSEGNA.md), che indicizza tutti i file. La mappa completa del repository è nel README in cima al repository.

---

## Domanda di ricerca

Nel nostro setting, un modello di diffusione condizionato produce forecast probabilistici migliori — per accuratezza e calibrazione — di una baseline ingenua, un modello classico (ARIMA) e un modello deep autoregressivo (DeepAR), e l'eventuale miglioramento ripaga il costo di campionamento? È il tema dell'ultimo capitolo del corso (generazione incondizionata) esteso al forecasting **condizionato** `p(futuro | passato)`.

## Idea

Inquadriamo il forecasting come stima della distribuzione condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) del corso. Abbiamo confrontato una scala di modelli a complessità crescente — **M0** seasonal-naive, **M1** ARIMA, **M2** DeepAR, **M3** TimeGrad (diffusione autoregressiva), **M4** TimeDiff (diffusione non-autoregressiva, con ablazione **M4ε**) — con le stesse metriche: accuratezza puntuale (MASE), qualità probabilistica (**CRPS**), copertura/calibrazione e costo.

## Risultati principali (Electricity, CRPS più basso è meglio)

| Modello | CRPS ↓ | MASE ↓ | cov50 | cov90 |
|---|---:|---:|---:|---:|
| **M0 seasonal-naive** | **160.5** | 1.00 | 0.50 | 0.89 |
| M1 ARIMA | 867.5 | 3.66 | 0.59 | 0.93 |
| M2 DeepAR | 253.7 | 1.83 | 0.47 | 0.83 |
| M3 TimeGrad | 241.6 | 1.39 | 0.28 | 0.65 |
| M4 TimeDiff (x0) | 287.3 | 1.40 | 0.00 | 0.01 |
| M4ε TimeDiff (ε) | 1376.3 | 4.02 | 1.00 | 1.00 |

- Su Electricity il **seasonal-naive resta la baseline più forte sul CRPS** (160.5); nessun modello addestrato lo batte.
- **TimeGrad migliora DeepAR sul CRPS** (241.6 vs 253.7) ma **non batte M0**, ed è costoso (~4h19 di campionamento).
- **DeepAR è il modello più calibrato tra quelli addestrati** (copertura 0.47 / 0.83, la più vicina ai target 0.50 / 0.90).
- **TimeDiff x0 collassa la varianza predittiva** (copertura ≈ 0); **TimeDiff ε sovra-disperde** (copertura ≈ 1, MASE 4.0). Cambiare parametrizzazione non lo calibra.

Più che incoronare un modello, il lavoro serve a rendere il confronto equo e a misurare il compromesso tra qualità della distribuzione, calibrazione e costo. La tabella è **generata** da `results/registry.csv`, non scritta a mano: [`tables/comparison_electricity.md`](tables/comparison_electricity.md).

## Percorso di lettura consigliato

1. [`SINTESI_PROGETTO.md`](SINTESI_PROGETTO.md) — una pagina per inquadrare in fretta.
2. [`slides/deck_electricity_it.pdf`](slides/deck_electricity_it.pdf) — la stessa storia in 8 minuti.
3. [`REPORT.md`](REPORT.md) — la trattazione completa (teoria DDPM, scala M0–M4, valutazione, limiti).
4. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — come re-derivare ogni numero/artefatto.

## Note di onestà

Split **nostro** (LSTNet 70/10/20), **non** `electricity_nips`: CRPS **interno**, non confrontabile coi paper (E0 predisposto ma non eseguito, serve GPU). Un solo seed (42), budget modesto (50 epoche, `diff_steps=100`). E2/E3 in corso, E4 pianificato, toy DDPM didattico, E6 (valore economico) è una demo su M0. Dettaglio in [`CHECKLIST_CONSEGNA_IT.md`](CHECKLIST_CONSEGNA_IT.md).
