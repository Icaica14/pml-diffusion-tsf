# Comparativa M0–M3 — Exchange

> Generato da `experiments/make_tables.py` da `results/registry.csv`. **Non modificare a mano** — rigenera il file.

_test split · n_windows=1430 · H=60 · τ=30 · D=8 · S=100 · seed=42 · m=1_

## Qualità

| Modello | MASE | CRPS | pinball | cov50 | cov90 | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| M0 seasonal-naive | 4.526 | **0.007175** | **0.00379** | **0.445** | 0.862 | **0.009812** | 0.01659 |
| M1 ARIMA | **4.525** | 0.007249 | 0.003831 | 0.633 | **0.932** | 0.009817 | **0.01658** |
| M2 DeepAR | 7.407 | 0.01025 | 0.005416 | 0.392 | 0.828 | 0.01413 | 0.02185 |
| M3 TimeGrad | 7.478 | 0.01134 | 0.005958 | 0.384 | 0.729 | 0.01504 | 0.02383 |

**Barra da battere (CRPS):** M0 = 0.007175 — il riferimento di persistenza che i modelli deep (e in particolare M3) devono superare. _Grassetto = miglior modello per colonna; coverage premia la vicinanza al nominale (0.50 / 0.90), non il valore più alto._

## Costo & setup

| Modello | fit | predict | epochs | diff_steps | piattaforma |
|---|---|---|---|---|---|
| M0 seasonal-naive | — | **0.5** | — | — | macOS (locale) |
| M1 ARIMA | **17.1** | 45.2 | — | — | macOS (locale) |
| M2 DeepAR | 58.0 | 28.1 | 20 | — | Linux (Colab) |
| M3 TimeGrad | 132.8 | 744.0 | 20 | 100 | Linux (Colab) |

_MASE/CRPS dipendono da scala e denominatore del dataset: **confrontabili solo entro lo stesso dataset**, mai tra Exchange ed Electricity._
