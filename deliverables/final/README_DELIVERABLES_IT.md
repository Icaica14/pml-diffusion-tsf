# Pacchetto di consegna — esame PML (UniTS, Prof. Bortolussi)

> **Forecasting probabilistico di serie temporali con modelli di diffusione — un confronto
> controllato su consumi elettrici (M0 → M4).**
> Gruppo: **Giovanni Mason (SM3800158) · Lorenzo Di Bernardo (SM3800132) · Lorenzo Karol Gobbo (SM28A00018)** · Appello **10 giugno 2026**.

Questa cartella `deliverables/final/` è il **pacchetto autosufficiente** da consegnare/zippare.
Tutto ciò che serve al professore è qui dentro; il resto del repo è il codice che lo produce.

---

## Il risultato in un paragrafo (la "storia")

Su **Electricity** (321 serie orarie, forte stagionalità) confrontiamo una scala di modelli a
complessità crescente — **M0** seasonal-naive, **M1** ARIMA, **M2** DeepAR, **M3** TimeGrad
(diffusione autoregressiva), **M4** TimeDiff (diffusione non-autoregressiva) — con le stesse
metriche (MASE, **CRPS**, copertura/calibrazione, costo). Quello che troviamo: nel nostro
setting nessun modello addestrato batte il seasonal-naive sul CRPS (la barra è **160.5**;
DeepAR 253.7, TimeGrad 241.6, TimeDiff 287.3, ARIMA 867.5). Il modello meglio calibrato della
scala è **DeepAR**, non una diffusione. Le due diffusioni si comportano in modo opposto a
seconda della parametrizzazione: TimeDiff in forma **x0** sotto-disperde (copertura ≈ 0,
CRPS ≈ MAE), mentre la variante **ε** non corregge il problema ma lo ribalta nell'eccesso
opposto (copertura ≈ 1). Più che incoronare un modello, il lavoro serve a rendere il confronto
equo e a misurare il compromesso tra qualità della distribuzione, calibrazione e costo.

---

## Cosa c'è in questa cartella

| # | File | Cos'è |
|---|---|---|
| 1 | [`REPORT.md`](REPORT.md) | **Report scientifico completo** (IT): teoria DDPM, scala M0–M4, i quattro pilastri di valutazione, esperimenti, risultati reali, limiti, riproducibilità. |
| 2 | [`README.md`](README.md) | Panoramica del progetto (EN + IT), tabella modello×dataset, risultati principali, mappa repo. |
| 3 | [`slides/deck_electricity_it.pdf`](slides/deck_electricity_it.pdf) | **Le slide in PDF** (autosufficienti) — da proiettare. ~10 slide + backup B1–B7. |
| 4 | [`slides/deck_electricity_it.pptx`](slides/deck_electricity_it.pptx) | Le slide in PowerPoint (note del relatore nel pannello note). |
| 5 | [`slides/deck_electricity_it.md`](slides/deck_electricity_it.md) | Il **sorgente** delle slide (Markdown/pandoc). *Le immagini sono incorporate nel PDF/PPTX; il `.md` referenzia `figures/presentation/` dal root del repo — rigenera con `build_deck.sh`.* |
| 6 | [`SPEAKER_SCRIPT_8_MIN_IT.md`](SPEAKER_SCRIPT_8_MIN_IT.md) | **Copione 8–10 min**, diviso A (Giovanni) · B (Lorenzo Di Bernardo) · C (Lorenzo Karol Gobbo). |
| 7 | [`QA_ORALE_ELECTRICITY_IT.md`](QA_ORALE_ELECTRICITY_IT.md) | **Banca Q&A orale** (37 domande+risposte) con i numeri a memoria. |
| 8 | [`tables/`](tables/) | Tabelle finali **generate** dal registry: `comparison_electricity.{md,csv}`, `comparison_exchange.{md,csv}`. |
| 9 | [`figures/`](figures/) | Le 6 figure di confronto: CRPS, MASE, calibrazione, intervalli, costo, qualità-vs-costo. |
| 10 | [`economic/`](economic/) | **E6** (valore economico, demo M0): `value_electricity.{md,csv}` + `fig_e6_money.png`. |
| 11 | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Come re-derivare ogni numero/artefatto (ambiente, dati, protocollo, comandi). |
| 12 | [`CHECKLIST_CONSEGNA_IT.md`](CHECKLIST_CONSEGNA_IT.md) | **Checklist finale** prima di consegnare (cosa mandare, controlli, limiti). |
| 13 | `README_DELIVERABLES_IT.md` | *(questo file)* l'indice del pacchetto. |
| 14 | [`ONE_PAGE_SUMMARY_IT.md`](ONE_PAGE_SUMMARY_IT.md) | **Sintesi di una pagina** (IT): domanda, dati, modelli, metriche, tre risultati, limiti — per capire in fretta cosa abbiamo fatto. |

---

## Per il professore: il percorso di lettura consigliato

1. **`ONE_PAGE_SUMMARY_IT.md`** — una pagina per inquadrare in fretta cosa abbiamo fatto.
2. **Slide PDF** (`slides/deck_electricity_it.pdf`) — la stessa storia in 8 minuti.
3. **`REPORT.md`** — la trattazione completa (§0 sintesi esecutiva, §6 esperimenti, §8 limiti).
4. **`tables/comparison_electricity.md`** — i numeri principali in una tabella.

## Per chi vuole verificare / rieseguire

→ **`REPRODUCIBILITY.md`**. In breve: ogni numero vive in `results/registry.csv`; tabelle e
figure si rigenerano in pochi secondi su CPU
(`python3 -m experiments.make_tables`, `… plot_presentation`); i modelli profondi (M2–M4)
sono stati addestrati su GPU Colab L4 e le loro righe sono già in banca.

---

## Note di onestà (da non nascondere)

- Split **nostro** (LSTNet 70/10/20), **non** `electricity_nips`: CRPS **interno**, non
  confrontabile coi paper. E0 è predisposto ma **non** eseguito (serve GPU).
- **Un solo seed** (42); budget di calcolo modesto (50 epoche, `diff_steps=100`).
- **E2/E3 in corso**, **E4** pianificato, **toy DDPM** è didattico, **E6** è una demo su M0.
  Dettaglio completo in [`CHECKLIST_CONSEGNA_IT.md`](CHECKLIST_CONSEGNA_IT.md) §3.
