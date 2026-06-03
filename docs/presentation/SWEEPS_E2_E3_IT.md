# E2 ed E3 — sweep di orizzonte e di passi di denoising

> **A cosa servono, in una frase ciascuno.**
> - **E2 (orizzonte τ):** come si degradano la **qualità probabilistica** (CRPS) e la
>   **calibrazione** (copertura) man mano che si prevede **più lontano**.
> - **E3 (passi di denoising):** **quanti passi di reverse servono davvero** — la curva
>   ha un *elbow*: quasi tutta la qualità si compra nei primi pochi passi, mentre il
>   **costo cresce ~linearmente** coi passi.
>
> Entrambi gli sweep scrivono un CSV in `results/sweeps/` e le figure le disegna
> **`experiments/plot_sweeps.py`** in `figures/presentation/`.
>
> | Sweep | Runner | CSV | Figura |
> |---|---|---|---|
> | E2 orizzonte | `experiments/run_horizon_sweep.py` | `results/sweeps/horizon.csv` | `fig_e2_horizon.png` |
> | E3 passi | `experiments/run_toy_ddpm_sweep.py` (locale) · `run_timegrad.py` (Colab) | `results/sweeps/steps.csv` | `fig_e3_steps.png` |

---

## E2 — sweep di orizzonte (CRPS / copertura vs τ)

Tutto è **tenuto fisso tranne τ** (contesto H, lag stagionale m, n. campioni S, seed), così
l'unica cosa che muove la curva è *quanto avanti* prevediamo. Il denominatore MASE è
indipendente da τ, quindi anche MASE resta confrontabile lungo la curva.

### In locale — gambe leggere (M0 seasonal-naive, M1 ARIMA), sandbox Exchange

Nessuna GPU, niente GluonTS: gira in pochi minuti.

```bash
python -m experiments.run_horizon_sweep \
    --models seasonal_naive,arima \
    --config configs/data_exchange.yaml \
    --taus 7,15,30,60 --chunk 256 --samples 100
```

> Risultato già prodotto (8 righe in `results/sweeps/horizon.csv`): su Exchange — che è
> quasi un *random walk* — ARIMA ≈ seasonal-naive e il CRPS **cresce monotòno** con τ.
> È esattamente la storia da "negative control" del dataset sandbox.

### Su Colab — gambe pesanti (M2 DeepAR, M3 TimeGrad), dataset Electricity

**Stesso identico comando**, solo con i modelli deep e il config Electricity. Gli import
dei modelli pesanti sono *lazy*, quindi il runner è lo stesso file. Prima esegui il
setup di ambiente pinnato del notebook `notebooks/colab_m2_m3.ipynb` (install di
`pytorchts` + `gluonts` compatibili), poi:

```bash
# Colab, GPU attiva. τ piccoli/medi per Electricity (oraria): 12,24,48,96.
python -m experiments.run_horizon_sweep \
    --models deepar,timegrad \
    --config configs/data_electricity.yaml \
    --taus 12,24,48,96 \
    --chunk 256 --samples 100 \
    --epochs 50 --device cuda --accelerator gpu
```

> ⚠️ **Costo.** Ogni punto τ **riallena** il modello a quell'orizzonte. Con 4 valori di τ
> e 2 modelli deep sono **8 training**. TimeGrad ad Electricity costa da solo ~39 min di
> fit + ~4 h di sampling (vedi `RESULTS_PLACEHOLDERS_ELECTRICITY.md`): per lo sweep
> conviene **ridurre il budget** (`--epochs`, `--samples`, magari `--taus 24,96`) e/o
> usare `--predict-batch-size` piccolo. Lo scopo è la *forma della curva*, non i numeri
> headline (quelli sono in E1).

### Figura e lettura

```bash
python3 -m experiments.plot_sweeps   # legge horizon.csv (+ steps.csv) → PNG
```

`fig_e2_horizon.png` ha due pannelli:
- **sinistra — CRPS vs τ**, una linea per modello (più basso = meglio). Si guarda se e
  *dove* il ranking fra modelli cambia: l'ipotesi è che il vantaggio della diffusione
  cresca con l'orizzonte.
- **destra — copertura vs τ**: 90% (linea piena) e 50% (puntinata) contro i nominali
  tratteggiati. Dice se la **calibrazione regge** allungando τ o se le bande diventano
  troppo strette (sotto-copertura).

---

## E3 — sweep dei passi di denoising (l'elbow qualità/costo)

### In locale — il **toy DDPM** in puro NumPy (vehicle principale)

```bash
python -m experiments.run_toy_ddpm_sweep      # ~2-3 min, nessuna dipendenza pesante
```

Allena **un** `ToyConditionalDDPM` (`src/models/toy_ddpm.py`) sul target bimodale
`p(x|c)`, poi per ogni `(eta, n_sample_steps)` campiona, **cronometra** e calcola il
**CRPS** (stesso stimatore dei modelli veri). Scrive `results/sweeps/steps.csv`.

> **Perché il toy è il vehicle giusto per E3.** Il toy usa un **sampler DDIM con
> *respacing***: il numero di passi di reverse è un knob **al momento del campionamento**,
> su **un solo modello allenato**. Così la curva CRPS-vs-passi è fine e onesta (stesso
> peso, cambia solo il sampler). In più sweepa due `eta`, e il contrasto è didattico:
> - `eta=0` → **DDIM deterministico**: satura *subito* (~3 passi) ma a un **floor di CRPS
>   più alto** (~0.65) — il campionatore deterministico fatica a coprire **entrambe** le
>   mode del target;
> - `eta=1` → **ancestrale / DDPM-like**: serve qualche passo in più (~10–12) ma arriva a
>   un **CRPS migliore** (~0.58), perché il **rumore iniettato ripristina la dispersione**
>   che la distribuzione bimodale richiede.
>
> *È la tesi del progetto in miniatura:* la stocasticità della diffusione serve proprio a
> rappresentare distribuzioni **non-gaussiane** (qui bimodali) — un modello a testa
> gaussiana non può, e infatti la versione deterministica resta indietro.

### Su Colab — TimeGrad reale (punti di conferma)

> ⚠️ **Caveat importante e da dire all'orale.** In PyTorchTS il numero di passi di
> diffusione (`diff_steps`) è **fissato all'allenamento**: non esiste un knob per
> ridurre i passi *all'inferenza* (niente respacing DDIM pronto). Quindi sul modello
> **vero** ogni punto della curva E3 è un **training separato**. Per questo l'elbow fine
> lo diamo col toy (respacing gratis), e su TimeGrad mettiamo **2–3 punti di conferma**.

```bash
# Colab. Riallena TimeGrad a diversi diff_steps e logga una riga per ciascuno.
for T in 10 25 50 100; do
  python -m experiments.run_timegrad \
      --config configs/data_electricity.yaml \
      --diff-steps $T --epochs 50 --samples 100 \
      --device cuda
done
# poi una vista CRPS/tempo vs diff_steps dalle righe loggate (registry o steps.csv).
```

> Se vuoi che questi punti finiscano nello **stesso** `steps.csv` del toy (così la figura
> li sovrappone), aggiungi a `run_timegrad.py` un append con `sweep="steps"`,
> `model="timegrad"`, `n_sample_steps=diff_steps`. Altrimenti restano in `registry.csv`
> come righe E1 a `diff_steps` variabile e si confrontano a mano.

### Figura e lettura

`fig_e3_steps.png` ha due pannelli (toy):
- **sinistra — CRPS vs passi (log-x): l'elbow.** Una curva per `eta`; la linea
  tratteggiata è il *floor* (CRPS minimo di quella curva) e il cerchio annota l'**elbow** =
  il numero minimo di passi entro il 2% del floor. Messaggio: *la qualità satura presto*.
  Si nota anche che la curva `eta=1` (rossa) **sta sotto** la `eta=0` (verde): l'ancestrale
  paga qualche passo in più ma raggiunge un CRPS migliore (vedi sopra, il punto sulla
  bimodalità).
- **destra — tempo vs passi**: il costo cresce **~linearmente** (retta `∝ passi` con
  ms/passo). Messaggio: *ogni passo in più si paga, ma non rende*.

---

## Nota onesta (riproducibilità / confrontabilità)

- **toy DDPM ≠ TimeGrad.** Il toy dimostra il *meccanismo* (forward/reverse, ε-pred,
  DDIM) in modo trasparente e CPU-only; TimeGrad è il modello dei risultati. La curva E3
  del toy mostra la *forma* (elbow + costo lineare); i punti TimeGrad la *confermano*.
- I numeri **non sono mai confrontabili fra dataset diversi** (scale diverse) né fra toy e
  modelli veri: dentro una figura si confronta solo ciò che condivide asse e dato.
- Seed e griglie sono negli args dei runner; ogni riga CSV porta `seed`, `platform` e i
  parametri che definiscono il punto, quindi è ricostruibile.
```
