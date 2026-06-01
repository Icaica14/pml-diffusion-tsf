# Piano figure — Electricity (presentazione PML)

> **Mappa di tutte le figure della presentazione.** Per ciascuna: cosa mostra,
> perché serve, da quali dati nasce, se **esiste già** o è **da generare**, e con
> quale **script**. Convenzione colori stabile in tutto il progetto:
>
> | Modello | Colore | Hex |
> |---|---|---|
> | M0 seasonal-naive | grigio | `#7f7f7f` |
> | M1 ARIMA | blu | `#1f77b4` |
> | M2 DeepAR | arancione | `#ff7f0e` |
> | M3 TimeGrad 🔴 | rosso | `#d62728` |
>
> 🔴 **Le figure che includono M3 usano i valori-placeholder** (centro intervalli da
> `RESULTS_PLACEHOLDERS_ELECTRICITY.md`) e mostrano un **marcatore "PLACEHOLDER"**
> finché la riga reale non entra in `results/registry.csv`.
> `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
>
> **Output attesi in:** `figures/presentation/` (PNG, ~150 dpi, sfondo bianco).

---

## Tabella sintetica

| # | File | Slide | Stato | Generata da |
|---|---|---|---|---|
| 1 | `fig_cmp_crps.png` | 8 | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 2 | `fig_cmp_mase.png` | 8 (backup) | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 3 | `fig_cmp_quality_cost.png` | 9, 10 | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 4 | `fig_cmp_calibration.png` | B1 | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 5 | `fig_cmp_intervals.png` | B1 | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 6 | `fig_cmp_cost.png` | 9 (supp.) | 🔵 M3 placeholder | `experiments/plot_presentation.py` |
| 7 | `fig_fanchart_example.png` | 2 | 🟡 da generare | manuale / notebook EDA |
| 8 | `fig_dataset_overview.png` | 3 | 🟡 da generare | manuale / notebook EDA |
| 9 | `fig_exchange_vs_electricity.png` | 4 | 🟡 da generare | manuale / notebook EDA |
| 10 | `fig_model_ladder.png` | 5 | 🟡 schema | disegno/slide |
| 11 | `fig_diffusion_intuition.png` | 6 | 🟡 schema | disegno/toy DDPM |
| 12 | `fig_teaser_fanchart.png` | 1 (opz.) | 🟡 opzionale | come #7 |

> Le figure **1–6** sono **auto-generate** dallo script di presentazione (leggono la
> registry, gestiscono il placeholder M3). Le **7–12** sono illustrative/didattiche e
> si possono fare a mano o riusando i notebook EDA.

---

## Dettaglio figura per figura

### 1. `fig_cmp_crps.png` — barre CRPS (la metrica principale)
- **Cosa mostra:** una barra di CRPS per modello (M0, M1, M2, M3), più bassa = meglio.
  Linea orizzontale tratteggiata sulla **barra da battere** = 160.5 (M0).
- **Perché:** è la slide-risultato centrale (S8). Rende visibile che DeepAR non
  raggiunge il naive e pone la domanda su M3.
- **Dati:** colonna `CRPS` da `results/registry.csv`, righe `dataset=electricity`.
- **M3:** 🔴 se assente, barra rossa **tratteggiata** a CRPS≈210 (placeholder) con
  etichetta "PLACEHOLDER".
- **Stato:** 🔵 da generare ora (placeholder), auto-aggiorna a M3 reale.

### 2. `fig_cmp_mase.png` — barre MASE (accuratezza puntuale)
- **Cosa mostra:** MASE per modello; linea a **1.0** = soglia seasonal-naive.
- **Perché:** complemento puntuale del CRPS; backup di S8.
- **Dati:** colonna `MASE`, righe electricity.
- **M3:** 🔴 placeholder MASE≈1.53 (barra tratteggiata).
- **Stato:** 🔵 da generare ora (placeholder).

### 3. `fig_cmp_quality_cost.png` — scatter qualità vs costo ⭐
- **Cosa mostra:** asse X = `predict_s` (costo, log), asse Y = `CRPS` (qualità); un
  punto per modello; ideale in basso-a-sinistra. Etichette accanto ai punti.
- **Perché:** **è la tesi della presentazione** (S9, richiamo S10). Sposta il discorso
  da "chi vince" a "quanto costa".
- **Dati:** `predict_s` + `CRPS`, righe electricity.
- **M3:** 🔴 punto rosso con bordo tratteggiato a (~10 800 s, ~210) + "PLACEHOLDER".
- **Stato:** 🔵 da generare ora (placeholder).

### 4. `fig_cmp_calibration.png` — calibrazione (backup B1)
- **Cosa mostra:** per ogni modello, cov50 e cov90 vs i nominali 0.50 e 0.90 (due
  pannelli o barre raggruppate); vicino alla linea nominale = ben calibrato.
- **Perché:** risponde a "come fai a dire che è calibrato?".
- **Dati:** `cov50`, `cov90`, righe electricity.
- **M3:** 🔴 placeholder cov50≈0.49, cov90≈0.87.
- **Stato:** 🔵 da generare ora (placeholder).

### 5. `fig_cmp_intervals.png` — ampiezza bande / sharpness (backup B1)
- **Cosa mostra:** width50 e width90 per modello; a parità di copertura, più stretto =
  più informativo.
- **Perché:** la calibrazione da sola non basta; mostra il compromesso copertura↔ampiezza.
- **Dati:** `width50`, `width90`, righe electricity.
- **M3:** 🔴 placeholder (se non in registry, lasciare barra tratteggiata o omettere M3).
- **Stato:** 🔵 da generare ora (placeholder).

### 6. `fig_cmp_cost.png` — costi fit/predict (supplemento S9)
- **Cosa mostra:** barre dei tempi `fit_s` e `predict_s` per modello (scala log).
- **Perché:** quantifica il "prezzo" di cui parla C; utile se chiedono i tempi.
- **Dati:** `fit_s`, `predict_s`, righe electricity.
- **M3:** 🔴 placeholder fit≈1800 s, predict≈10 800 s.
- **Stato:** 🔵 da generare ora (placeholder).

### 7. `fig_fanchart_example.png` — esempio di previsione probabilistica (S2)
- **Cosa mostra:** una finestra reale di Electricity: storia, mediana prevista, banda
  50% e 90% attorno; il vero futuro sovrapposto.
- **Perché:** rende concreto "prevediamo una distribuzione, non un punto".
- **Dati:** un campione qualunque del test (anche da M0 o M2); serve solo a illustrare.
- **M3:** non necessario.
- **Stato:** 🟡 da generare (manuale o da notebook EDA). Non bloccante.

### 8. `fig_dataset_overview.png` — panoramica Electricity (S3)
- **Cosa mostra:** qualche serie campione + heatmap ora-del-giorno × giorno-settimana
  che evidenzia la doppia stagionalità.
- **Perché:** giustifica perché il seasonal-naive è forte e perché il dataset è
  interessante.
- **Dati:** `data/processed` Electricity (non committato).
- **Stato:** 🟡 da generare. Non bloccante.

### 9. `fig_exchange_vs_electricity.png` — confronto di regime (S4)
- **Cosa mostra:** due mini-serie affiancate: un tasso di cambio (quasi random walk,
  senza stagionalità) vs un consumo elettrico (stagionale).
- **Perché:** spiega il ruolo dei due dataset e la frase "il vincitore dipende dal
  segnale".
- **Dati:** un canale Exchange + un canale Electricity.
- **Stato:** 🟡 da generare. Non bloccante.

### 10. `fig_model_ladder.png` — la scaletta dei modelli (S5)
- **Cosa mostra:** schema M0→M1→M2→M3 con due frecce "espressività ↑" e "costo ↑".
- **Perché:** dà la mappa mentale dei quattro modelli in un colpo d'occhio.
- **Dati:** nessuno (schema).
- **Stato:** 🟡 da disegnare (anche direttamente in slide). Non bloccante.

### 11. `fig_diffusion_intuition.png` — rumore↔segnale (S6)
- **Cosa mostra:** striscia in due righe: forward (dato→rumore) in alto, reverse
  (rumore→dato) in basso; opzionale un toy DDPM 1-D che "ripulisce" una sinusoide.
- **Perché:** spiega la diffusione senza math.
- **Dati:** nessuno o un toy 1-D generato al volo.
- **Stato:** 🟡 da disegnare / toy opzionale. Non bloccante.

### 12. `fig_teaser_fanchart.png` — teaser di copertina (S1, opzionale)
- **Cosa mostra:** un fan-chart elegante, solo decorativo, per la slide titolo.
- **Stato:** 🟡 opzionale (può essere lo stesso di #7).

---

## Come si generano le figure auto (1–6)

```bash
# dalla root del repo
python3 -m experiments.plot_presentation
```

Lo script `experiments/plot_presentation.py`:
1. legge `results/registry.csv` e tiene **solo** `dataset=electricity`;
2. se la riga `model=timegrad` (Electricity) **manca**, usa i **valori-placeholder**
   da `RESULTS_PLACEHOLDERS_ELECTRICITY.md` e disegna M3 **tratteggiato** con etichetta
   "PLACEHOLDER"; se **c'è**, usa i numeri reali e toglie il marcatore;
3. salva i PNG in `figures/presentation/`.

> Dopo M3 (Fase B) basta **rieseguire lo stesso comando**: trovata la riga reale, le
> figure 1–6 si aggiornano da sole e perdono il marcatore placeholder. Vedi
> [`NEXT_PROMPT_AFTER_M3_IT.md`](NEXT_PROMPT_AFTER_M3_IT.md).
