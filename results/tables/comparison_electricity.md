# Comparativa M0–M4 — Electricity

> Generato da `experiments/make_tables.py` da `results/registry.csv`. **Non modificare a mano** — rigenera il file.

_test split · n_windows=5071 · H=168 · τ=24 · D=321 · S=100 · seed=42 · m=24_

## Qualità

| Modello | MASE | CRPS | pinball | cov50 | cov90 | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| M0 seasonal-naive | **1.003** | **160.5** | **84.6** | **0.497** | **0.885** | **203.0** | **1,759.7** |
| M1 ARIMA | 3.664 | 867.5 | 459.0 | 0.592 | 0.930 | 1,131.9 | 12,870.6 |
| M2 DeepAR | 1.830 | 253.7 | 133.9 | 0.466 | 0.831 | 356.0 | 2,448.8 |
| M3 TimeGrad | 1.391 | 241.6 | 126.6 | 0.279 | 0.647 | 312.8 | 3,025.3 |
| M4 TimeDiff | 1.399 | 287.3 | 143.8 | 0.003 | 0.008 | 288.5 | 2,551.5 |
| M4ε TimeDiff (ε) | 4.017 | 1,376.3 | 729.0 | 0.998 | 1.000 | 1,044.8 | 9,370.4 |

**Barra da battere (CRPS):** M0 = 160.5 — il riferimento di persistenza che i modelli deep (e in particolare M3) devono superare. _Grassetto = miglior modello per colonna; coverage premia la vicinanza al nominale (0.50 / 0.90), non il valore più alto._

## Costo & setup

| Modello | fit | predict | epochs | diff_steps | piattaforma |
|---|---|---|---|---|---|
| M0 seasonal-naive | — | 3,520.5 | — | — | macOS (locale) |
| M1 ARIMA | 1,758.5 | 9,719.5 | — | — | macOS (locale) |
| M2 DeepAR | 608.8 | 4,930.0 | 50 | — | Linux (Colab) |
| M3 TimeGrad | 2,331.9 | 15,526.0 | 50 | 100 | Linux (Colab) |
| M4 TimeDiff | 48.0 | 2,484.7 | 50 | 100 | Linux (Colab) |
| M4ε TimeDiff (ε) | **43.3** | **2,439.2** | 50 | 100 | Linux (Colab) |

_MASE/CRPS dipendono da scala e denominatore del dataset: **confrontabili solo entro lo stesso dataset**, mai tra Exchange ed Electricity._
