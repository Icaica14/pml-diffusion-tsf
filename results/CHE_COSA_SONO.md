# CHE_COSA_SONO — results

> I **numeri** prodotti dagli esperimenti. Il cuore è `registry.csv`: ogni run appende
> **una riga**; tabelle e figure si **generano** da lì, non si scrivono mai a mano.

| File | Che cos'è |
|---|---|
| `registry.csv` | **La fonte di verità.** Una riga per (dataset, modello, run) con tutte le metriche (CRPS, MASE, coperture, ampiezze, tempi) e i parametri che definiscono il confronto (H, τ, S, seed…). |
| `manifest_exchange.json` | Manifest del dataset Exchange: hash/shape/split, per riproducibilità. |
| `manifest_electricity.json` | Idem per Electricity. |
| `arima_orders_exchange.json` | Ordini ARIMA `(p,d,q)` per canale, in cache per non rifare la selezione a ogni run. |
| `arima_orders_electricity.json` | Idem per Electricity. |
| `eda_exchange_stats.json` | Statistiche numeriche dell'EDA Exchange (autocorrelazioni, ecc.), prodotte dal notebook EDA. |

## Sottocartelle (ognuna col suo `CHE_COSA_SONO.md`)
| Cartella | Contenuto |
|---|---|
| `tables/` | Tabelle di confronto generate dal registry (CSV + Markdown). |
| `economic/` | Output dell'esperimento E6 (valore economico). |
| `sweeps/` | Output degli sweep E2 (orizzonte) ed E3 (passi di denoising). |

> ⚠️ I numeri **non sono confrontabili tra dataset diversi** (scale diverse): si confronta
> solo all'interno dello stesso dataset.
