# CHE_COSA_SONO — results/tables

> Tabelle di confronto **generate** dal registry da `experiments/make_tables.py` (mai
> scritte a mano). Una coppia CSV + Markdown per dataset.

| File | Che cos'è |
|---|---|
| `comparison_exchange.csv` | Confronto M0–M3 su Exchange (formato macchina). |
| `comparison_exchange.md` | La stessa tabella in Markdown. |
| `comparison_electricity.csv` | Confronto M0–M3 su Electricity (formato macchina). |
| `comparison_electricity.md` | La stessa tabella in Markdown. |

> ⚠️ I numeri **non sono confrontabili tra dataset diversi** (scale diverse): si confronta
> solo dentro la stessa tabella. Per rigenerarle: `python -m experiments.make_tables`.
