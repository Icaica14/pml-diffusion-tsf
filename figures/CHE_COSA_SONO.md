# CHE_COSA_SONO — figures

> Figure PNG **generate dagli script** (mai disegnate a mano). Qui stanno quelle della
> **sandbox Exchange** e dell'**EDA**. Le figure della presentazione sono in
> `figures/presentation/`.

## Confronto modelli sulla sandbox Exchange (da `experiments/plot_results.py`)
| File | Che cos'è |
|---|---|
| `cmp_crps.png` | CRPS per modello (qualità probabilistica). |
| `cmp_mase.png` | MASE per modello (accuratezza puntuale). |
| `cmp_calibration.png` | Calibrazione: copertura empirica vs nominale. |
| `cmp_intervals.png` | Copertura + ampiezza delle bande (sharpness). |
| `cmp_cost.png` | Costi di fit e di predict. |
| `cmp_quality_cost.png` | Qualità vs costo (la "tesi" in un grafico). |

## EDA Exchange (dal notebook `notebooks/01_eda_exchange.ipynb`)
| File | Che cos'è |
|---|---|
| `eda_exchange_series.png` | Le 8 serie dei tassi di cambio nel tempo. |
| `eda_exchange_acf.png` | Autocorrelazione (lag-1 ~0.999 → near-random-walk). |
| `eda_exchange_returns_dist.png` | Distribuzione dei rendimenti giornalieri. |
| `eda_exchange_volatility.png` | Volatilità nel tempo. |
| `eda_exchange_regimes.png` | Regimi / finestre rappresentative della serie. |
