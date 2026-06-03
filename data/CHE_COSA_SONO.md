# CHE_COSA_SONO — data

> Dati locali. **Il contenuto NON è nella repo** (`.gitignore`): i file grezzi/processati
> sono grandi e/o si riscaricano. Questa cartella si **popola da sola** al primo run: il
> loader (`src/data/loader.py`) scarica il file se manca e lo mette in cache.

| Sottocartella | Che cos'è |
|---|---|
| `raw/` | Download grezzi dei dataset (es. `exchange_rate.txt.gz`, `electricity.txt.gz`), presi dal benchmark LSTNet. **Gitignored.** |
| `processed/` | Eventuali artefatti intermedi/cache (`*.parquet`). Creata al bisogno. **Gitignored.** |

> URL sorgente, cadenza e numero di canali stanno in `configs/data_*.yaml`. Per
> (ri)creare i dati basta lanciare un qualsiasi `experiments/run_*.py`: il loader scarica,
> poi taglia gli split temporali e costruisce le finestre.
