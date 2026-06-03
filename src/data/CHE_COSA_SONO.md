# CHE_COSA_SONO — src/data

> Il "data layer": dal file grezzo alle **finestre** `(contesto, target)` pronte per i
> modelli, con split **temporale** (niente shuffle) e scaling **fittato solo sul train**
> (così non c'è leakage dal futuro).

| File | Che cos'è |
|---|---|
| `loader.py` | Builder generico (`build_dataset(cfg)`): scarica se manca, taglia gli split contigui, costruisce le finestre scorrevoli. È il punto d'ingresso usato da tutti i runner. |
| `contract.py` | Il **contratto dati** (`ForecastDataset`): l'oggetto condiviso che ogni modello consuma — finestre per split, serie grezze, ponti verso GluonTS (uni/multivariato), metadati (freq, start, D, H, τ). |
| `exchange.py` | Loader specifico del dataset **Exchange**. |
| `electricity.py` | Loader specifico del dataset **Electricity** (il primario). |
| `scaling.py` | Standardizzazione per-canale (z-score), **fit su TRAIN**, applicata a val/test. |
| `__init__.py` | API pubblica del package data. |
