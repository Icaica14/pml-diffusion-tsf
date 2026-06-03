# CHE_COSA_SONO — experiments

> Script **eseguibili** (CLI). Ognuno fa girare un pezzo della pipeline: allena/valuta un
> modello, oppure genera tabelle e figure **dai numeri già nel registry** (non
> ricalcola). Si lanciano dalla radice con `python -m experiments.<nome>`.

## Run dei modelli — allenano, valutano e appendono **una riga** a `results/registry.csv`
| File | Che cos'è |
|---|---|
| `run_naive.py` | **M0** seasonal-naive (l'ancora di onestà). Locale, niente GPU. |
| `run_arima.py` | **M1** ARIMA per-canale. Locale (statsmodels). |
| `run_deepar.py` | **M2** DeepAR (baseline deep probabilistico). Pesante: Colab/GPU. |
| `run_timegrad.py` | **M3** TimeGrad (diffusione condizionale autoregressiva, il centro del progetto). Pesante: Colab/GPU. |
| `run_timediff.py` | **M4** TimeDiff (diffusione condizionale non-autoregressiva, torch puro). Ha un `--smoke` per validare la pipeline in pochi secondi prima del run lungo. Pesante: Colab/GPU. |

## Esperimenti trasversali
| File | Che cos'è |
|---|---|
| `run_economic.py` | **E6**: trasforma le previsioni in decisioni (dispatch di una batteria, LP) e misura il valore in €. |
| `run_horizon_sweep.py` | **E2**: ripete un modello su una griglia di orizzonti τ → `results/sweeps/horizon.csv`. Gambe leggere (M0/M1) in locale, deep su Colab. |
| `run_toy_ddpm_sweep.py` | **E3**: sweep dei passi di denoising sul toy DDPM (puro NumPy) → `results/sweeps/steps.csv`. Tutto in locale. |

## Generazione di tabelle e figure (leggono il registry, **non** ricalcolano)
| File | Che cos'è |
|---|---|
| `make_tables.py` | Dal registry alle tabelle di confronto in `results/tables/` (CSV + Markdown). |
| `plot_results.py` | Figure della **sandbox Exchange** → `figures/` (`cmp_*.png`). |
| `plot_presentation.py` | Le 6 figure della presentazione **Electricity** → `figures/presentation/` (`fig_cmp_*.png`). |
| `plot_sweeps.py` | Le figure degli sweep → `figures/presentation/fig_e2_horizon.png` e `fig_e3_steps.png`. |

> Promemoria: i numeri **non** si scrivono mai a mano. Si lancia un `run_*`, che appende
> al registry; poi `make_tables`/`plot_*` rigenerano tutto da lì.
