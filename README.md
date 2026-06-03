# Probabilistic Time-Series Forecasting with Diffusion Models

> In our setting, does a *conditional diffusion model* produce probabilistic forecasts that beat classical and deep-learning baselines on accuracy and calibration — and at what sampling cost?

![status](https://img.shields.io/badge/status-pipeline%20complete-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![course](https://img.shields.io/badge/course-PML%20%C2%B7%20UniTS-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Exam project for the *Probabilistic Machine Learning* (PML) course — University of Trieste, Prof. Luca Bortolussi.**
**Delivery branch:** `feature/port-ladder-exchange` (not yet merged into `main` — please use this branch).

🇬🇧 English (below) · 🇮🇹 [Versione italiana](#-in-italiano)

---

## Overview

Most forecasts give a single number — *"tomorrow's electricity demand will be 100."* The future is uncertain, and one number hides that. We treat forecasting as estimating the conditional distribution `p(future | past)` and implement it with a **conditional diffusion model** — the generative family the PML course covers in its final chapter, here made conditional on the past. We compare it against a seasonal-naive baseline, a classical model (ARIMA), and a deep probabilistic model (DeepAR), measuring four things: point accuracy (MAE/RMSE/MASE), probabilistic quality (CRPS, coverage, calibration), sampling/training cost, and a small economic-value demo (scheduling a battery on each forecast).

The result is **not** a win for diffusion. In our setting no trained model beats the seasonal-naive on CRPS on Electricity; whatever diffusion adds is conditional on the horizon, the dataset, the target parameterization (x0 vs ε), and a sampling cost we measure. We read this as a controlled comparison, not a model coronation.

## Research question

Can a conditional diffusion model forecast `p(future | past)` better — on accuracy and calibration — than a naive baseline, a classical model, and a deep autoregressive model, and is any improvement worth its sampling cost? This is the PML course's final-chapter topic (unconditional generation) extended to **conditional** forecasting.

## Project status

🟢 **Pipeline complete; results banked in `results/registry.csv`.** The model ladder M0→M3 runs on both datasets; the fifth rung **M4 TimeDiff** (and its **M4ε** ε-ablation) runs on the primary Electricity dataset. Tables and figures are *generated* from the registry, never hand-typed.

| | M0 naive | M1 ARIMA | M2 DeepAR | M3 TimeGrad | M4 TimeDiff |
|---|:---:|:---:|:---:|:---:|:---:|
| **Exchange** (D=8, sandbox) | ✅ | ✅ | ✅ | ✅ | — |
| **Electricity** (D=321, primary) | ✅ | ✅ | ✅ | ✅ | ✅ + ε-ablation |

**Experiments:** `E1` main comparison and `E6` economic-value demo are done; `E2` horizon sweep and `E3` denoising-steps sweep are in progress; `E0` reproduce-gate and `E4` regime-shift are planned (see [Limitations & future work](#limitations--future-work)).

## Final deliverables

The self-contained package for the professor lives in **[`consegna_finale/`](consegna_finale/)** — start from **[`LEGGIMI_CONSEGNA.md`](consegna_finale/LEGGIMI_CONSEGNA.md)**, which indexes it.

| What | File |
|---|---|
| Slides (8-min talk) | [`consegna_finale/slides/deck_electricity_it.pdf`](consegna_finale/slides/deck_electricity_it.pdf) |
| Full scientific report (PDF) | [`consegna_finale/REPORT.pdf`](consegna_finale/REPORT.pdf) |
| One-page summary (PDF) | [`consegna_finale/SINTESI_PROGETTO.pdf`](consegna_finale/SINTESI_PROGETTO.pdf) |
| Package index | [`consegna_finale/LEGGIMI_CONSEGNA.md`](consegna_finale/LEGGIMI_CONSEGNA.md) |

The report source (Markdown, Italian) is also at the repo root: [`REPORT.md`](REPORT.md).

## Main results

Electricity, identical protocol for every model (test split, H=168, τ=24, S=100 samples, seed=42). Lower is better for CRPS and MASE; coverage targets are 0.50 and 0.90.

| Model | CRPS ↓ | MASE ↓ | cov50 | cov90 |
|---|---:|---:|---:|---:|
| **M0 seasonal-naive** | **160.5** | 1.00 | 0.50 | 0.89 |
| M1 ARIMA (per-channel) | 867.5 | 3.66 | 0.59 | 0.93 |
| M2 DeepAR | 253.7 | 1.83 | 0.47 | 0.83 |
| M3 TimeGrad | 241.6 | 1.39 | 0.28 | 0.65 |
| M4 TimeDiff (x0) | 287.3 | 1.40 | 0.00 | 0.01 |
| M4ε TimeDiff (ε) | 1376.3 | 4.02 | 1.00 | 1.00 |

What the numbers say, in our setting:

- On Electricity the **seasonal-naive stays the strongest baseline on CRPS** (160.5); no trained model beats it.
- **TimeGrad improves on DeepAR on CRPS** (241.6 vs 253.7) but does not beat M0, and it is expensive — sampling takes ~4 h 19 min.
- **DeepAR is the most calibrated of the trained models** (coverage 0.47 / 0.83, closest to the 0.50 / 0.90 targets).
- **TimeDiff x0 collapses the predictive variance** (coverage ≈ 0) despite a competitive point error; the **ε-ablation over-disperses** (coverage ≈ 1, MASE 4.0). x0 under-disperses, ε over-disperses — a parameterization swap does not calibrate it.

## Datasets

| Dataset | Role | Shape | Notes |
|---|---|---|---|
| **Exchange** | sandbox / negative control | D=8, daily | near random-walk; M0 is hard to beat — a sanity check |
| **Electricity** | primary | D=321, hourly | strong daily/weekly seasonality |

Both come from the LSTNet multivariate time-series collection: <https://github.com/laiguokun/multivariate-time-series-data>. We use the raw LSTNet file with **our own 70/10/20 split**, **not** the published `electricity_nips` split — so our CRPS is an **internal** comparison across our own models, not directly comparable to published paper tables (see [Limitations](#limitations--future-work)). Raw data is not committed; the loaders fetch/build it.

## Models compared

| Model | Role | Library |
|---|---|---|
| **M0 — Seasonal-naive** | the baseline to beat | NumPy |
| **M1 — ARIMA** | classical statistical baseline (one model per channel) | statsforecast |
| **M2 — DeepAR** | deep autoregressive probabilistic baseline | GluonTS |
| **M3 — TimeGrad** | *autoregressive* conditional diffusion (ε-prediction) | PyTorchTS |
| **M4 — TimeDiff** | *non-autoregressive* conditional diffusion (x0-prediction) | PyTorch (self-contained) |
| *M4ε* | TimeDiff ε-prediction ablation | PyTorch |
| *+ toy DDPM* | a ~150-line from-scratch conditional DDPM, didactic only | PyTorch |

## Repository structure

Every folder also carries a `CHE_COSA_SONO.md` (Italian) describing its files.

```
pml-diffusion-tsf/
├── README.md · CONTRIBUTING.md · requirements.txt
├── docs/                     # implementation plan (EN + IT); presentation/ = the 8-min deck source
├── configs/                  # one YAML per dataset (data_exchange.yaml, data_electricity.yaml)
├── src/
│   ├── data/                 # data contract: loaders, temporal split, train-only scaling, windowing
│   ├── models/               # ladder wrappers: naive (M0), arima (M1), deepar (M2), timegrad (M3), timediff (M4)
│   ├── eval/                 # metrics (CRPS, coverage, calibration), registry, economic.py (E6)
│   └── utils/                # seeds, config loading, GluonTS freq-compat shim
├── experiments/              # run_{naive,arima,deepar,timegrad,timediff}.py + make_tables.py + plot_*.py + run_economic.py
├── notebooks/                # Exchange EDA + the Colab GPU notebooks (M2/M3/M4)
├── tests/                    # data-contract + economic-LP unit tests
├── results/                  # registry.csv (single source of truth) + tables/ + economic/
├── figures/                  # generated plots; presentation/ = the deck figures
└── consegna_finale/          # delivery package: slides (PDF/PPTX), REPORT, one-page summary, speaker script, Q&A, tables, figures
```

## Reproduce the main tables & figures

The deep rows (M2–M4) were trained once on a Colab GPU and banked in the registry; everything below regenerates locally on CPU in seconds.

```bash
pip install -r requirements.txt

# tables + figures, parsed from results/registry.csv (idempotent):
python -m experiments.make_tables          # -> results/tables/comparison_*.md
python -m experiments.plot_presentation    # -> figures/presentation/fig_cmp_*.png

# the economic-value demo (E6, M0, no training):
python -m experiments.run_economic         # -> results/economic/ + fig_e6_money.png

pytest -q                                  # light unit tests (no GPU, no data download)
```

Re-training the deep models needs a CUDA GPU and is optional — full protocol in [`consegna_finale/REPRODUCIBILITY.md`](consegna_finale/REPRODUCIBILITY.md). The PyTorchTS ↔ GluonTS combo (M2/M3) is fragile and pinned in `requirements.txt`; M4 TimeDiff is self-contained PyTorch.

## Limitations & future work

- **Internal split.** Our 70/10/20 split is not `electricity_nips`, so CRPS is not paper-comparable. The `E0` reproduce-gate is prepared but **not run** (needs a GPU).
- **One seed (42), modest budget** (50 epochs, 100 diffusion steps, little tuning) — a plausible reason a diffusion model does not shine here, disclosed as such.
- **Calibration is unsolved for TimeDiff:** neither x0 nor ε is well-calibrated; learned-variance or conformal post-hoc calibration is future work.
- **Economic value (E6) is a demo on M0** only; the cross-model money comparison is future work.
- **E2/E3** (horizon and denoising-step sweeps) are in progress; **E4** (regime-shift) is planned; the toy DDPM is a didactic artifact, not a result.

Full detail in the report ([`REPORT.md`](REPORT.md) / [`consegna_finale/REPORT.pdf`](consegna_finale/REPORT.pdf)).

## Team

| Member | Matricola | Owns |
|---|---|---|
| Giovanni Mason | SM3800158 | M0, M1, classical/statistics narrative, point metrics |
| Lorenzo Di Bernardo | SM3800132 | M2, M3, M4, the toy DDPM, Colab/training, configs |
| Lorenzo Karol Gobbo | SM28A00018 | CRPS/coverage/calibration, figures, slide narrative, oral Q&A |

## Course context

The PML course ends on diffusion models for unconditional generation. This project extends that final chapter to **conditional** forecasting, `p(future | past)`, and discusses which uncertainty the model captures — **aleatoric** (the spread of plausible futures) vs **epistemic** (model/parameter uncertainty, which a point-estimated diffusion model does not capture). The implementation plan ([`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`_IT`](docs/IMPLEMENTATION_PLAN_IT.md)) maps each component back to a section of the course notes.

## License & course materials

Released under the [MIT License](LICENSE). The repository **intentionally does not include** the professor's lecture notes or course textbooks (copyrighted, and the notes ask not to be redistributed); they are cited by section and page only. `*.pdf` files are gitignored as a safeguard. Colleagues: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch conventions.

---

## 🇮🇹 In italiano

**Progetto d'esame per il corso di *Probabilistic Machine Learning* (PML) — Università di Trieste, Prof. Luca Bortolussi. Gruppo di 3.**
**Branch di consegna:** `feature/port-ladder-exchange` (non ancora unito a `main`: fate riferimento a questo branch).

**Domanda di ricerca.** Nel nostro setting, un modello di diffusione condizionato produce forecast probabilistici migliori — per accuratezza e calibrazione — di una baseline ingenua, un modello classico (ARIMA) e un modello deep autoregressivo (DeepAR), e l'eventuale miglioramento ripaga il costo di campionamento? È il tema dell'ultimo capitolo del corso (generazione incondizionata) esteso al forecasting **condizionato** `p(futuro | passato)`.

**Idea.** Inquadriamo il forecasting come stima della distribuzione condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) del corso. Abbiamo confrontato quattro cose: accuratezza puntuale (MAE/RMSE/MASE), qualità probabilistica (CRPS, copertura, calibrazione), costo (tempo di addestramento e di campionamento) e una piccola demo di valore economico (programmare una batteria sul forecast di ciascun modello, esperimento E6).

**Risultati principali (Electricity, CRPS più basso è meglio):**

- Su Electricity il **seasonal-naive resta la baseline più forte sul CRPS** (160.5); nessun modello addestrato lo batte.
- **TimeGrad migliora DeepAR sul CRPS** (241.6 vs 253.7) ma **non batte M0**, ed è costoso (~4h19 di campionamento).
- **DeepAR è il modello più calibrato tra quelli addestrati** (copertura 0.47 / 0.83, la più vicina ai target 0.50 / 0.90).
- **TimeDiff x0 collassa la varianza predittiva** (copertura ≈ 0), mentre **TimeDiff ε sovra-disperde** (copertura ≈ 1, MASE 4.0): cambiare parametrizzazione non lo calibra.

Numeri completi: M0 160.5 · M1 867.5 · M2 253.7 · M3 241.6 · M4 (x0) 287.3 · M4ε (ε) 1376.3.

**Dataset.** *Exchange* (D=8, giornaliero) come sandbox/controllo negativo (quasi random walk, M0 difficile da battere); *Electricity* (D=321, orario) come dataset principale, forte stagionalità. Entrambi dalla collezione LSTNet (<https://github.com/laiguokun/multivariate-time-series-data>). Usiamo lo split **nostro** 70/10/20, non `electricity_nips`: il CRPS è **interno**, non confrontabile coi paper.

**Materiale finale.** Il pacchetto autosufficiente è in [`consegna_finale/`](consegna_finale/): [slide PDF](consegna_finale/slides/deck_electricity_it.pdf), [REPORT.pdf](consegna_finale/REPORT.pdf), [sintesi di una pagina](consegna_finale/SINTESI_PROGETTO.pdf), indice in [`LEGGIMI_CONSEGNA.md`](consegna_finale/LEGGIMI_CONSEGNA.md). Il report completo (sorgente Markdown) è anche in [`REPORT.md`](REPORT.md).

**Limiti e lavori futuri.** Split interno (E0 predisposto ma non eseguito, serve GPU); un solo seed e budget modesto; calibrazione di TimeDiff irrisolta (né x0 né ε); E6 è una demo su M0 (confronto cross-model futuro); E2/E3 in corso, E4 pianificato, toy DDPM didattico.

**Per i colleghi:** leggete prima il piano (Parti 1–3 di [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md)), poi la parte del vostro ruolo. Regola d'oro: far girare tutta la pipeline sul piccolo dataset **Exchange** prima di salire a Electricity.
