# CHE_COSA_SONO — radice del progetto

> `pml-diffusion-tsf`: **forecasting probabilistico** di serie temporali con **modelli di
> diffusione** (esame PML, Università di Trieste). Questa è la cartella radice. Qui stanno
> i file di progetto; il codice e i dati sono nelle sottocartelle.
>
> 📌 **Convenzione:** ogni cartella della repo contiene un file `CHE_COSA_SONO.md` come
> questo, che spiega in italiano cosa sono i file lì dentro. Apri quello della cartella
> che ti interessa.

## File in questa cartella
| File | Che cos'è |
|---|---|
| `README.md` | Presentazione del progetto: stato, ladder dei modelli (M0→M3), dataset, come si parte. È il punto d'ingresso per chi apre la repo. |
| `CONTRIBUTING.md` | Regole di collaborazione: branch-per-task + Pull Request, mai commit diretti su `main`, niente dati o checkpoint nei commit. |
| `requirements.txt` | Dipendenze: il gruppo "leggero" (NumPy/pandas/statsmodels…) gira in locale; il gruppo "pesante" (torch/gluonts/pytorchts) è pinnato e si installa su Colab. |
| `LICENSE` | Licenza del codice. |
| `.gitignore` | Cosa NON committare: dati grezzi/processati, checkpoint, cache, PDF (le note del corso sono protette da copyright). |
| `STATO_PROGETTO.md` | Diario di stato in italiano (file **locale, non committato**): avanzamento, risultati, prossimi passi. |

## Sottocartelle (ognuna col suo `CHE_COSA_SONO.md`)
| Cartella | Contenuto |
|---|---|
| `src/` | Codice sorgente (librerie): dati, modelli, valutazione, utilità. |
| `experiments/` | Script eseguibili: run dei modelli, generazione tabelle/figure, sweep E2/E3. |
| `configs/` | Configurazioni YAML dei due dataset (Exchange, Electricity). |
| `data/` | Dati grezzi/processati (gitignored: non finiscono nella repo). |
| `results/` | I numeri prodotti dagli esperimenti (registry, tabelle, sweep, manifest). |
| `figures/` | Figure PNG (sandbox Exchange + EDA). |
| `figures/presentation/` | Figure della presentazione (Electricity + E6 + E2/E3). |
| `docs/` | Documentazione: piano di implementazione, EDA, materiale per la presentazione. |
| `notebooks/` | Jupyter: EDA locale + notebook Colab per i modelli pesanti M2/M3. |
| `tests/` | Test automatici (contratto dati, LP economico E6). |
