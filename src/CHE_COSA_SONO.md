# CHE_COSA_SONO — src

> Il **codice sorgente**, organizzato in package. Qui non c'è niente di eseguibile: gli
> script da lanciare stanno in `experiments/`. `src/` fornisce le **librerie** che quelli
> usano (caricare dati, costruire modelli, calcolare metriche).

| Package | Che cos'è |
|---|---|
| `data/` | Dal file grezzo alle finestre: caricamento, split temporale, scaling (fit su train), windowing, manifest. |
| `models/` | I quattro modelli della ladder (M0–M3) + il toy DDPM didattico. |
| `eval/` | Metriche (puntuali e probabilistiche), registry dei risultati, valore economico E6. |
| `utils/` | Utility trasversali: seeding, caricamento config, compatibilità frequenze. |
| `viz/` | Spazio per le funzioni di plotting condivise (per ora i grafici vivono in `experiments/`). |
| `__init__.py` | Marca `src` come package Python. |

> Ogni sotto-package ha il suo `CHE_COSA_SONO.md` con il dettaglio file per file.
