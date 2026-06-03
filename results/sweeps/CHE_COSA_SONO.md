# CHE_COSA_SONO — results/sweeps

> Output degli **sweep** parametrici. Stessa logica del registry: una riga per ogni punto
> della curva. Le figure le disegna `experiments/plot_sweeps.py`.

| File | Che cos'è |
|---|---|
| `horizon.csv` | **E2** — una riga per (modello, orizzonte τ): come cambiano CRPS, copertura e tempi al crescere di τ. |
| `steps.csv` | **E3** — una riga per (eta, numero di passi di denoising) del toy DDPM: CRPS, MAE e tempo di campionamento. Mostra l'elbow qualità/costo. |

> `horizon.csv` ha le stesse colonne del registry più una colonna `sweep`; `steps.csv`
> ha colonne proprie del toy (`n_sample_steps`, `eta`, `ms_per_sample`…).
