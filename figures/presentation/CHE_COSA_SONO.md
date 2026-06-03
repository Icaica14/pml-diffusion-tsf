# CHE_COSA_SONO — figures/presentation

> Le figure che vanno **nella presentazione** (dataset Electricity), più gli esperimenti
> E6/E2/E3. Tutte generate da script che leggono `results/` (mai disegnate a mano).

## Confronto modelli su Electricity (da `experiments/plot_presentation.py`)
| File | Che cos'è |
|---|---|
| `fig_cmp_crps.png` | CRPS per modello + la "barra da battere" (M0). La slide-risultato centrale. |
| `fig_cmp_mase.png` | MASE per modello (soglia naive = 1.0). |
| `fig_cmp_quality_cost.png` | Qualità (CRPS) vs costo (predict_s): **la tesi** della presentazione. |
| `fig_cmp_calibration.png` | Calibrazione: copertura empirica vs nominale @50/90. |
| `fig_cmp_intervals.png` | Ampiezza delle bande (sharpness) @50/90. |
| `fig_cmp_cost.png` | Tempi di fit/predict (scala log). |

## Esperimenti aggiuntivi
| File | Che cos'è |
|---|---|
| `fig_e6_money.png` | **E6**: valore economico (€) per modello dal dispatch della batteria. Da `run_economic.py`. |
| `fig_e2_horizon.png` | **E2**: CRPS e copertura vs orizzonte τ. Da `plot_sweeps.py`. |
| `fig_e3_steps.png` | **E3**: elbow CRPS-vs-passi + costo-vs-passi del toy DDPM. Da `plot_sweeps.py`. |
