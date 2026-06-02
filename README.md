# Probabilistic Time-Series Forecasting with Diffusion Models

> **Can a diffusion model produce *probabilistic* forecasts of a future time series — a distribution of plausible futures rather than a single number — that are more accurate, better calibrated, or more informative than classical and deep-learning baselines, and at what computational cost?**

![status](https://img.shields.io/badge/status-results%20in-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![course](https://img.shields.io/badge/course-PML%20%C2%B7%20UniTS-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Exam project for the *Probabilistic Machine Learning* (PML) course — University of Trieste, Prof. Luca Bortolussi.**

🇬🇧 English (below) · 🇮🇹 [Versione italiana](#-in-italiano)

---

## Overview

Most forecasts give you a single number — *"tomorrow's electricity demand will be 100."* But the future is uncertain, and one number hides that uncertainty entirely. A more useful forecast says: *"here are the plausible tomorrows, and how likely each one is"* — a whole **distribution** of futures instead of one guess.

That is what this project builds. We treat forecasting as one question: *given everything seen so far (the past), what does the distribution of possible futures look like?* — written `p(future | past)`. To answer it we use a **diffusion model**: the same family of generative models behind modern image generators, retrained here to *generate plausible future trajectories of a time series* instead of pictures. It learns from the exact objective the PML course derives in its final chapter — our twist is to make it **conditional on the past**.

We compare it head-to-head against a naive baseline, a classical statistical model (ARIMA/ETS), and a deep autoregressive probabilistic model (DeepAR), and we measure **four** things the exam rewards:

- **Point accuracy** — MAE, RMSE, MASE
- **Probabilistic quality** — CRPS, interval coverage, calibration
- **Cost** — training/inference time, and the quality-vs-denoising-steps trade-off
- **Economic value** — the *money saved* when a real decision (scheduling a battery against a time-of-use price) is driven by each model's forecast, because a better-calibrated distribution makes cheaper, more robust decisions (see [Economic value](#economic-value--turning-forecasts-into-money) below)

Our thesis is deliberately non-triumphalist: diffusion buys *richer, better-calibrated uncertainty*, but its advantage depends on horizon, dataset, and a real sampling-cost penalty we can measure and tune.

## Status

🟢 **Pipeline complete; full results in.** The whole model ladder runs on both datasets (M0→M3 on each; the fifth rung, **M4 TimeDiff**, on the primary Electricity dataset), the head-to-head tables and figures are *generated* from the registry (never hand-typed), and the economic-value pillar (E6) is built. Every number below is real, from `results/registry.csv`.

**Model × dataset — all rungs evaluated:**

| | M0 naive | M1 ARIMA | M2 DeepAR | M3 TimeGrad | M4 TimeDiff |
|---|:---:|:---:|:---:|:---:|:---:|
| **Exchange** (D=8, sandbox) | ✅ | ✅ | ✅ | ✅ | — |
| **Electricity** (D=321, primary) | ✅ | ✅ | ✅ | ✅ | ✅ |

**Headline (Electricity, CRPS ↓ — the probabilistic-quality metric):** the seasonal-naive bar is **160.5**, and *no trained model beats it* — DeepAR 253.7, **TimeGrad 241.6** (better than DeepAR, but not the naive), **TimeDiff 287.3**, ARIMA 867.5. This is the project's honest, non-triumphalist finding: on a strongly seasonal signal a simple baseline is a serious opponent, and diffusion's richer uncertainty does not pay for its sampling cost *here*. The two diffusion models make the point from opposite ends: **M3 TimeGrad** (autoregressive) is the better-calibrated but slow one (~4 h to sample), while **M4 TimeDiff** (non-autoregressive) samples in ~41 min and has the lowest point error of any deep model, yet in its x0-prediction form the predictive **collapses** (coverage ≈ 0, CRPS ≈ MAE) — an ε-prediction ablation to recalibrate it is in progress.

**Experiments:** `E1` main comparison ✅ · `E6` economic value ✅ · `E2` horizon sweep / `E3` denoising-steps 🟡 in progress · `E0` reproduce-gate, `E4` regime-shift ⏳ planned.

| Document | Description |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · [`_IT`](docs/IMPLEMENTATION_PLAN_IT.md) | The complete, execution-ready implementation plan (EN + IT), with a deep appendix (E) on the experimental phase. |
| [`docs/presentation/`](docs/presentation/) | The 8-minute Electricity deck (FASE B, real M3 + M4 numbers): slide template, speaker script, figure plan, oral-Q&A bank. |
| [`docs/EDA_EXCHANGE.md`](docs/EDA_EXCHANGE.md) | Figure-backed exploratory analysis of the Exchange sandbox dataset (random-walk levels, volatility clustering, heavy tails). |
| `results/tables/comparison_*.md` | Auto-generated head-to-head tables (one per dataset). |

## Approach at a glance

**The model ladder** (each rung a stronger opponent):

| Model | Role | Library |
|---|---|---|
| **M0 — Seasonal-naive** | the honesty anchor — must be beaten | NumPy / Darts |
| **M1 — ARIMA / ETS** | the classical statistical baseline | statsforecast |
| **M2 — DeepAR** | the *fair* deep probabilistic baseline | GluonTS |
| **M3 — TimeGrad** | the frontier: *autoregressive* conditional diffusion *(centerpiece)* | PyTorchTS |
| **M4 — TimeDiff** | the *non-autoregressive* diffusion variant — one reverse chain over the whole horizon, fast sampling | PyTorch (self-contained) |
| *+ toy DDPM* | a ~150-line from-scratch conditional DDPM — the "understanding artifact" | PyTorch |

**The experiments** (each numbered, falsifiable, producing one artifact):

`E0` reproduce-a-published-result gate · `E1` main comparison · `E2` horizon sweep · `E3` denoising-steps vs quality & cost · `E4` regime-shift robustness · `E6` **economic value (battery dispatch)** · `E5` generality (stretch).

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 5–7 for the full specification.

## Economic value — turning forecasts into money

A forecast only matters if it changes a **decision**. Our fourth evaluation pillar makes that concrete: we use each model's forecast to **schedule a battery** (charge when power is cheap, discharge when it's expensive) against a time-of-use electricity price, then price the result.

- A **point** forecast plans against a single guessed future; a **distribution** (diffusion / DeepAR samples) plans against the whole spread of plausible futures, hedging its bets.
- The schedule is a small **linear program**; with a distribution we minimize the *expected* bill over the model's sampled trajectories (sample-average approximation).
- We apply every schedule to the **true** future and read off the realized bill, then report **money saved** vs a naive baseline (ceiling) and a perfect-foresight **oracle** (lower bound) — so euros are always shown as a fraction of what was actually achievable.
- The punchline ties the pillars together: optimal storage decisions use a **quantile** of the predictive distribution (a *newsvendor* structure), and **CRPS is the average decision regret over all cost ratios** — so a better-calibrated forecast should literally save more money. `E6` tests whether it does.

This is a **bolt-on module** (`src/eval/economic.py`) that runs *over the forecasts E1 already produces* — **no extra training** — so it adds a headline result without enlarging the core project. Full protocol in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §3.5, §6.4, and experiment E6.

## Repository structure

Every folder also carries a `CHE_COSA_SONO.md` (Italian) describing its files, so the repo is self-documenting.

```
pml-diffusion-tsf/
├── README.md · CONTRIBUTING.md · requirements.txt
├── docs/                     # implementation plan (EN + IT); presentation/ = the 8-min deck
├── configs/                  # one YAML per dataset (data_exchange.yaml, data_electricity.yaml)
├── src/
│   ├── data/                 # the data contract: loaders, temporal split, train-only scaling, windowing
│   ├── models/               # the ladder wrappers: naive (M0), arima (M1), deepar (M2), timegrad (M3), timediff (M4)
│   ├── eval/                 # metrics (CRPS, coverage, calibration), registry, economic.py (E6 battery dispatch)
│   ├── viz/                  # (placeholder) plotting currently lives in experiments/
│   └── utils/                # seeds, config loading, GluonTS freq-compat shim
├── experiments/              # run_{naive,arima,deepar,timegrad,timediff}.py + make_tables.py + plot_*.py + run_economic.py
├── notebooks/                # Exchange EDA + the Colab GPU notebooks (M2/M3/M4)
├── tests/                    # data-contract + economic-LP unit tests (18 green)
├── results/                  # registry.csv (single source of truth) + tables/ + economic/ (committed)
└── figures/                  # generated plots; presentation/ = the deck figures (committed)
```

## Getting started

```bash
pip install -r requirements.txt          # light local env (data, M0/M1, eval, plots, E6)

# run the ladder's local rungs on the sandbox dataset (Exchange):
python -m experiments.run_naive   --config configs/data_exchange.yaml   # M0
python -m experiments.run_arima   --config configs/data_exchange.yaml   # M1

# assemble the head-to-head tables and the deck figures from the registry:
python -m experiments.make_tables                 # -> results/tables/comparison_*.md
python -m experiments.plot_presentation           # -> figures/presentation/fig_cmp_*.png

# the economic-value pillar (E6), no training needed:
python -m experiments.run_economic                # -> results/economic/ + fig_e6_money.png
pytest -q                                         # 18 tests
```

**Two environments.** A light **local** one (above: data, classical baselines M0/M1, evaluation, plots, the E6 LP) and a **Colab/GPU** one for the deep models M2 (DeepAR), M3 (TimeGrad), and M4 (TimeDiff) — see [`notebooks/`](notebooks/). The PyTorchTS ↔ GluonTS combo (M2/M3) is fragile and is pinned in `requirements.txt`; **M4 TimeDiff is self-contained PyTorch** (stock Colab torch, no pinning).

**Golden rule:** get the whole pipeline green on the small **Exchange** dataset first; *only then* scale up to the primary **Electricity** dataset (`--config configs/data_electricity.yaml`).

## Team & roles

Each member owns one vertical slice end-to-end (so each can defend it in the individual oral exam):

| Role | Owns | Member |
|---|---|---|
| **A — Baselines & Statistics** | M0, M1, classical/PML-statistics narrative, point metrics | _add name · GitHub handle_ |
| **B — Diffusion & Infrastructure** | M2, M3, M4, the toy DDPM, Colab/training, configs | _add name · GitHub handle_ |
| **C — Evaluation, Viz & Story** | CRPS/coverage/calibration, all figures, slide narrative, oral Q&A bank | _add name · GitHub handle_ |

## Course context

The PML course *ends* on diffusion models (unconditional generation). This project's originality is to **extend that final chapter from unconditional generation to *conditional* forecasting**, `p(future | past)`, and to discuss explicitly *which* uncertainty the model captures — **aleatoric** (the spread of plausible futures) vs **epistemic** (model/parameter uncertainty, which a point-estimated diffusion model does *not* capture). The implementation plan maps every component back to a specific chapter/page of the course notes.

## Contributing

Colleagues: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch conventions, how to pick up a role, and the definition of "done" for each experiment.

## A note on course materials

This repository **intentionally does not include** the professor's lecture notes or any course textbooks. Those are copyrighted, and the notes carry an explicit request not to be redistributed. The plan cites them by section and page number only. `*.pdf` files are gitignored as a safeguard.

## License

Released under the [MIT License](LICENSE). If your university's coursework policy requires otherwise, change this before making the work widely public.

---

## 🇮🇹 In italiano

**Progetto d'esame per il corso di *Probabilistic Machine Learning* (PML, apprendimento automatico probabilistico) — Università di Trieste, Prof. Luca Bortolussi. Gruppo di 3.**

**Domanda di ricerca.** Un *diffusion model* può produrre **forecast probabilistici** di una serie temporale — una *distribuzione* di futuri plausibili invece di un singolo numero — meglio calibrati delle baseline classiche e di deep learning, e a quale costo computazionale?

**Idea.** Inquadriamo il forecasting come l'apprendimento della distribuzione generativa condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) che il corso ricava nell'ultimo capitolo. Lo confrontiamo con una baseline ingenua, un modello classico (ARIMA/ETS) e un modello probabilistico di deep learning (DeepAR), misurando **quattro** cose: **accuratezza puntuale** (MAE/RMSE/MASE), **qualità probabilistica** (CRPS, copertura, calibrazione), **costo** e **valore economico** — il denaro risparmiato quando il forecast di ciascun modello programma una batteria contro un prezzo a fasce orarie (esperimento E6, un modulo *sopra i forecast già prodotti*, senza addestramento aggiuntivo).

**Stato:** 🟢 pipeline completa, risultati in cassaforte. La scala gira su entrambi i dataset (M0→M3 su ciascuno; il quinto gradino, **M4 TimeDiff**, sul dataset principale Electricity); tabelle e figure di confronto sono **generate dalla registry** (mai scritte a mano); il pilastro del valore economico (E6) è costruito. **Risultato chiave (Electricity, CRPS):** la barra del seasonal-naive è **160.5** e **nessun modello addestrato la batte** — DeepAR 253.7, **TimeGrad 241.6** (meglio di DeepAR, non del naive), **TimeDiff 287.3**, ARIMA 867.5. Finding onesto e non-trionfalista: su un segnale fortemente stagionale una baseline semplice è un avversario serio. I due diffusion model lo mostrano da estremi opposti: **M3 TimeGrad** (autoregressivo) è il più calibrato ma lento (~4h di sampling), mentre **M4 TimeDiff** (non-autoregressivo) campiona in ~41 min ed ha l'errore puntuale più basso tra i deep, ma nella forma x0 la predittiva **collassa** (copertura ≈ 0, CRPS ≈ MAE) — un'ablazione ε per ricalibrarla è in corso. Il deck di presentazione (8 minuti, FASE B) è in [`docs/presentation/`](docs/presentation/); il piano completo in [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md).

**Per i colleghi:** leggete prima il piano (Parti 1–3), poi la parte del vostro ruolo (A / B / C, vedi tabella sopra). La regola d'oro: far girare tutta la pipeline sul piccolo dataset **Exchange** prima di salire di scala.
