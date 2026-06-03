# CHE_COSA_SONO — src/eval

> La **valutazione**: come si trasformano le previsioni in numeri confrontabili, dove quei
> numeri vengono salvati, e (E6) come si convertono in valore economico.

| File | Che cos'è |
|---|---|
| `metrics.py` | Metriche **puntuali** (MAE/RMSE/MASE) e **probabilistiche** (CRPS, copertura, calibrazione, pinball). Include la valutazione "eager" e quella "a chunk" per dataset larghi. |
| `registry.py` | `append_result`: appende **una riga** a `results/registry.csv` (la fonte di verità). Schema che può crescere senza rompere le righe vecchie. |
| `economic.py` | **E6**: il problema di ottimizzazione lineare (dispatch di una batteria) che trasforma le previsioni in valore €. |
| `__init__.py` | API pubblica del package eval. |
