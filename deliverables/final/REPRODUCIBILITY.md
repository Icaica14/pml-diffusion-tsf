# Reproducibility

> **One sentence.** Every headline number lives in `results/registry.csv`; the tables
> and figures are *generated* from it (never hand-typed); the deep models (M2–M4) were
> trained once on a Colab GPU and their result rows banked into that registry, while
> M0/M1, all tables, all figures and the E6 economic demo regenerate locally on CPU in
> seconds.

This file tells a reader exactly how to re-derive each deliverable in
`deliverables/final/`. Paths below are **repo-relative** — run every command from the
repository root unless noted.

---

## 1. Environment

- **Python** 3.10+ (developed and run on 3.12).
- **Local stack** (CPU, for M0/M1, tables, figures, the E6 demo, and all tests):
  `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `statsmodels`/`pmdarima` (ARIMA),
  `torch` (TimeDiff is **pure PyTorch**, no extra framework). See `requirements.txt`.
- **GPU stack** (only to *re-train* the deep legs M2/M3): `gluonts` + a PyTorch backend,
  on CUDA. These produced the banked rows on a Colab **L4** GPU; you do **not** need them
  to read the results or rebuild the deck/tables/figures.
- **Determinism**: a single global seed `seed = 42` is set via `src/utils/seeds.py`
  (`set_seed`) for every run. **One seed only** — this is a declared limitation
  (no variance-from-initialization estimate), not an oversight.

Quick sanity check of the metric and decision code (no GPU, no data download):

```bash
python3 -m pytest tests/                 # if pytest is installed
# or, plain-python fallback per file:
python3 tests/test_metrics.py            # CRPS / CRPS_sum invariants
python3 tests/test_economic.py           # battery-dispatch LP correctness
python3 tests/test_data_contract.py      # ForecastDataset contract
```

---

## 2. Data

| Dataset | Source | Shape | Split |
|---|---|---|---|
| **Exchange** (sandbox / negative control) | repo loader | D=8, daily | our 70/10/20 |
| **Electricity** (headline) | LSTNet `electricity.txt.gz` (321 series, hourly) | D=321 | our 70/10/20 |
| `electricity_nips` (E0, *planned* GPU run) | `gluonts.get_dataset("electricity_nips")` | D=370 | **official** + 7 rolling windows |

Raw data is **not committed** (gitignored); the loaders fetch/build it. The headline
results use the **raw LSTNet file with our split**, *not* the published `electricity_nips`
split — so our CRPS is an **internal**, apples-to-apples comparison across our own models,
**not** directly comparable to the TimeGrad/CSDI paper tables. E0 (`configs/data_electricity_nips.yaml`,
`docs/E0_REPRODUCE_GATE.md`) is the prepared-but-not-yet-run bridge to the published metric.

---

## 3. Protocol (identical for every model)

| | Electricity (headline) | Exchange (control) |
|---|---|---|
| context `H` | 168 (one week) | 60 |
| horizon `τ` | 24 (one day) | 30 |
| season `m` | 24 | 1 |
| samples `S` | 100 | 100 |
| seed | 42 | 42 |
| test windows | 5071 | 1430 |

Metrics are implemented once in `src/eval/metrics.py` (MAE, RMSE, MASE, CRPS via the
almost-unbiased ensemble estimator, interval coverage/width, pinball; `crps_sum` is
opt-in for E0). The same evaluator scores every model, so the head-to-head is fair.

---

## 4. The single source of truth

`results/registry.csv` — **10 banked rows**: Exchange {M0, M1, M2, M3} and Electricity
{M0, M1, M2, M3, M4 (x0), M4ε (eps)}. Nothing downstream re-types a number; the tables and
figures parse this file. If you trust one artifact, trust this one.

| Model | Where it ran | CRPS (Electricity) |
|---|---|---|
| M0 seasonal-naive | local CPU | **160.5** ← the bar to beat |
| M1 ARIMA (per-channel) | local CPU | 867.5 |
| M2 DeepAR | Colab L4 GPU | 253.7 |
| M3 TimeGrad | Colab L4 GPU | 241.6 |
| M4 TimeDiff (x0) | Colab L4 GPU | 287.3 (cov ≈ 0) |
| M4ε TimeDiff (eps) | Colab L4 GPU | 1376 (cov ≈ 1) |

**Honest finding:** no trained model beats the seasonal-naive on CRPS here, and the
best-calibrated model on the whole ladder is **DeepAR**, not a diffusion.

---

## 5. Regenerate the deliverables locally (CPU, seconds)

```bash
# 5.1 tables  → results/tables/comparison_{electricity,exchange}.{md,csv}
python3 -m experiments.make_tables

# 5.2 figures → figures/presentation/fig_cmp_*.png
python3 -m experiments.plot_presentation

# 5.3 slide deck → docs/presentation/deck_electricity_it.{pptx,pdf}
#     deps: pandoc (>=3); for the PDF also xelatex (MacTeX/TeX Live) + "Arial Unicode MS"
bash docs/presentation/build_deck.sh

# 5.4 E6 economic demo (M0, local) → results/economic/value_electricity.{md,csv}
python3 -m experiments.run_economic --config configs/data_electricity.yaml \
        --channels 6 --channel-select topvar --windows 120 --samples 100   # see --help
```

Tables and figures are **idempotent** functions of the registry: rerun them anytime and
they reproduce the committed artifacts bit-for-bit (figures up to backend rendering).

---

## 6. Re-train the deep models (GPU required — optional)

These rows are already banked; re-run only if you want to reproduce them from scratch on a
CUDA GPU. The resilient end-to-end recipe (with checkpointing and Drive backup) is
`notebooks/colab_electricity_all.ipynb`, the expedition that produced the banked rows.

```bash
python -m experiments.run_naive    --config configs/data_electricity.yaml --samples 100        # M0 (CPU ok)
python -m experiments.run_arima    --config configs/data_electricity.yaml                       # M1 (CPU, slow)
python -m experiments.run_deepar   --config configs/data_electricity.yaml --epochs 50           # M2 (GPU)
python -m experiments.run_timegrad --config configs/data_electricity.yaml --epochs 50 \
        --diff-steps 100 --samples 100 --device cuda                                            # M3 (GPU)
python -m experiments.run_timediff --config configs/data_electricity.yaml --param x0  --device cuda   # M4
python -m experiments.run_timediff --config configs/data_electricity.yaml --param eps --device cuda   # M4ε ablation
```

Each runner appends a row to `results/registry.csv`; then rerun §5.1–5.2 to refresh the
tables and figures. Use `--help` on any runner for the full flag set (epochs, batch size,
schedule, etc.).

---

## 7. Hardware actually used

- **Local**: macOS, CPU — M0, M1, all tables, all figures, the deck build, the E6 demo,
  every test.
- **Colab**: one **L4** GPU — M2 DeepAR, M3 TimeGrad, M4 TimeDiff, M4ε. Sampling cost is
  real and reported (TimeGrad predict ≈ 4 h 19 min; TimeDiff predict ≈ 41 min).

---

## 8. Caveats that bound reproducibility

1. **One seed (42).** No initialization-variance bars; for publication you'd want ≥3 seeds.
2. **Our split ≠ `electricity_nips`.** Headline CRPS is internal, not paper-comparable;
   E0 is prepared to close this on a GPU but was not run.
3. **Modest budget.** 50 epochs, `diff_steps = 100`, minimal tuning — a plausible reason a
   diffusion model does not shine, disclosed as such.
4. **Deep rows need a GPU.** Re-deriving M2–M4 locally on CPU is impractical; read them from
   the registry instead.
