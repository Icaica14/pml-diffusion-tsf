# Probabilistic Time-Series Forecasting with Diffusion Models

> **Can a diffusion model produce *probabilistic* forecasts of a future time series — a distribution of plausible futures rather than a single number — that are more accurate, better calibrated, or more informative than classical and deep-learning baselines, and at what computational cost?**

![status](https://img.shields.io/badge/status-planning-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)
![course](https://img.shields.io/badge/course-PML%20%C2%B7%20UniTS-8A2BE2)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)

**Exam project for the *Probabilistic Machine Learning* (PML) course — University of Trieste, Prof. Luca Bortolussi. Team of 3.**

🇬🇧 English (below) · 🇮🇹 [Versione italiana](#-in-italiano)

---

## Overview

Real forecasting is not *"tomorrow's electricity demand will be 100"*; it is *"here is the distribution of plausible tomorrows."* We frame multi-step forecasting as learning the **conditional generative distribution** `p(future | past)` and instantiate it with a **conditional denoising diffusion model**, trained by the same ELBO / noise-prediction objective the PML course derives in its final chapter.

We compare it head-to-head against a naive baseline, a classical statistical model (ARIMA/ETS), and a deep autoregressive probabilistic model (DeepAR), and we measure three things the exam rewards:

- **Point accuracy** — MAE, RMSE, MASE
- **Probabilistic quality** — CRPS, interval coverage, calibration
- **Cost** — training/inference time, and the quality-vs-denoising-steps trade-off

Our thesis is deliberately non-triumphalist: diffusion buys *richer, better-calibrated uncertainty*, but its advantage depends on horizon, dataset, and a real sampling-cost penalty we can measure and tune.

## Status

🟡 **Planning phase — no code yet.** The full blueprint is written and lives in [`docs/`](docs/). Everything below will be built *from* that plan.

| Document | Description |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | The complete, execution-ready implementation plan (English). |
| [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md) | Full Italian translation, with every acronym expanded on first use and a deep appendix on the experimental phase. |

## Approach at a glance

**The model ladder** (each rung a stronger opponent):

| Model | Role | Library |
|---|---|---|
| **M0 — Seasonal-naive** | the honesty anchor — must be beaten | NumPy / Darts |
| **M1 — ARIMA / ETS** | the classical statistical baseline | statsforecast |
| **M2 — DeepAR** | the *fair* deep probabilistic baseline | GluonTS |
| **M3 — TimeGrad** | the frontier: conditional diffusion *(centerpiece)* | PyTorchTS |
| *+ toy DDPM* | a ~150-line from-scratch conditional DDPM — the "understanding artifact" | PyTorch |

**The experiments** (each numbered, falsifiable, producing one artifact):

`E0` reproduce-a-published-result gate · `E1` main comparison · `E2` horizon sweep · `E3` denoising-steps vs quality & cost · `E4` regime-shift robustness · `E5` generality (stretch).

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 5–7 for the full specification.

## Planned repository structure

```
pml-diffusion-tsf/
├── README.md
├── docs/                     # the implementation plan (EN + IT)
├── configs/                  # one YAML per experiment (model + data + seed)
├── src/
│   ├── data/                 # loaders, splitting, scaling, windowing, manifest
│   ├── models/               # wrappers: naive, arima, deepar, timegrad, toy_ddpm
│   ├── eval/                 # metrics (CRPS, coverage, calibration), runners
│   ├── viz/                  # consistent plotting (forecasts, intervals, curves)
│   └── utils/                # seeds, logging, timing, config loading
├── experiments/              # entry scripts: run_E0.py ... run_E4.py
├── notebooks/                # exploration + the toy-DDPM teaching notebook
├── results/                  # CSV results registry (committed)
└── figures/                  # generated plots for the slides (committed)
```

## Getting started

> Code is not here yet (planning phase). When it lands, this section will hold the exact setup. For now:

1. **Read the plan.** Start with [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) Parts 1–3 (the story), then your owned parts.
2. **Two environments are planned:** a light local one (data, classical baselines, evaluation, plots) and a Colab/GPU one (DeepAR, TimeGrad). The PyTorchTS ↔ GluonTS version combo is fragile and will be pinned in `requirements.txt`.
3. **Golden rule:** get the whole pipeline green on the small **Exchange** dataset first; *only then* scale up to the primary dataset.

## Team & roles

Each member owns one vertical slice end-to-end (so each can defend it in the individual oral exam):

| Role | Owns | Member |
|---|---|---|
| **A — Baselines & Statistics** | M0, M1, classical/PML-statistics narrative, point metrics | _add name · GitHub handle_ |
| **B — Diffusion & Infrastructure** | M2, M3, the toy DDPM, Colab/training, configs | _add name · GitHub handle_ |
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

**Idea.** Inquadriamo il forecasting come l'apprendimento della distribuzione generativa condizionata `p(futuro | passato)` e la realizziamo con un diffusion model condizionato, addestrato con lo stesso obiettivo (ELBO / predizione del rumore) che il corso ricava nell'ultimo capitolo. Lo confrontiamo con una baseline ingenua, un modello classico (ARIMA/ETS) e un modello probabilistico di deep learning (DeepAR), misurando: **accuratezza puntuale** (MAE/RMSE/MASE), **qualità probabilistica** (CRPS, copertura, calibrazione) e **costo**.

**Stato:** 🟡 fase di pianificazione, nessun codice ancora. Il piano completo è in [`docs/IMPLEMENTATION_PLAN_IT.md`](docs/IMPLEMENTATION_PLAN_IT.md) — include un'appendice che spiega a fondo la fase sperimentale (prima la metodologia, poi le istruzioni operative passo-passo).

**Per i colleghi:** leggete prima il piano (Parti 1–3), poi la parte del vostro ruolo (A / B / C, vedi tabella sopra). La regola d'oro: far girare tutta la pipeline sul piccolo dataset **Exchange** prima di salire di scala.
