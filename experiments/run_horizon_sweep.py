"""E2 — horizon sweep: how probabilistic forecast quality degrades with the horizon τ.

For each ``(model, τ)`` on the grid this rebuilds the dataset at that horizon, fits the
model, forecasts the **test** windows on the **original** scale, scores them with the
shared metrics module, and appends one row to ``results/sweeps/horizon.csv``. The curve
CRPS / coverage vs τ is then drawn by ``experiments/plot_sweeps.py``
(``figures/presentation/fig_e2_horizon.png``) — the clean, ideally-monotone story the
plan asks for in E2.

Everything **except τ is held fixed** (context length H, seasonal lag m, sample count S,
seed), so the only thing changing along the curve is *how far ahead we predict*. The
MASE denominator is τ-independent (a data-level constant), so MASE stays comparable
across the sweep too.

Local legs (no GPU, no GluonTS):  ``seasonal_naive`` (M0), ``arima`` (M1).
Deep legs (Colab GPU):            ``deepar`` (M2), ``timegrad`` (M3) — *same command*,
                                  run on the box with the heavy stack. The deep-model
                                  imports are **lazy**, so M0/M1 run with neither the
                                  heavy group nor a GPU installed.

Scoring goes through the exact same ``evaluate_forecast`` / ``evaluate_streaming`` path
as the headline ``run_*.py`` scripts, so a sweep point is comparable cell-for-cell with
the E1 row at the matching τ.

Usage::

    # local, sandbox dataset — a real 2-model curve in a few minutes
    python -m experiments.run_horizon_sweep --models seasonal_naive,arima \
        --config configs/data_exchange.yaml --taus 7,15,30,60 --chunk 256

    # Colab, primary dataset — add the deep rungs (chunked scoring for D=321)
    python -m experiments.run_horizon_sweep --models deepar,timegrad \
        --config configs/data_electricity.yaml --taus 12,24,48,96 \
        --chunk 256 --epochs 50 --samples 100
"""

from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import build_dataset  # noqa: E402
from src.eval.metrics import evaluate_forecast, evaluate_streaming, seasonal_naive_scale  # noqa: E402
from src.eval.registry import append_result  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def _score(ds, model, scale, chunk: int):
    """Forecast + score the **test** split (eager or chunked) for an already-fit model.

    Mirrors the scoring branch of every ``run_*.py`` so the sweep numbers are produced
    by the identical code path as the headline E1 rows. Returns
    ``(metrics, n_windows, predict_s)``.
    """
    if chunk and chunk > 0:
        t0 = time.perf_counter()
        metrics, n_windows = evaluate_streaming(
            ds, model, "test", scale, chunk, scaled=False, levels=(0.5, 0.9)
        )
        predict_s = time.perf_counter() - t0
    else:
        te_ctx, te_tgt = ds.windows("test", scaled=False)
        t0 = time.perf_counter()
        point, samples = model.predict(te_ctx)
        predict_s = time.perf_counter() - t0
        metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))
        n_windows = int(te_tgt.shape[0])
    return metrics, n_windows, predict_s


def _build_and_fit(name: str, ds, season: int, samples: int, seed: int, args):
    """Construct + fit one model at the dataset's current τ. Returns ``(model, fit_s)``.

    M0/M1 are local and torch-free. M2/M3 **lazy-import** the heavy stack (GluonTS /
    PyTorchTS) and mirror the constructors in ``run_deepar.py`` / ``run_timegrad.py``
    exactly, so a Colab sweep reuses the validated model code unchanged.
    """
    t0 = time.perf_counter()
    if name == "seasonal_naive":
        from src.models.naive import SeasonalNaiveForecaster

        model = SeasonalNaiveForecaster(
            season_length=season, horizon=ds.tau, n_samples=samples, seed=seed
        )
        if args.chunk and args.chunk > 0:
            model.fit_chunked(ds.iter_windows("train", args.chunk, scaled=False))
        else:
            tr_ctx, tr_tgt = ds.windows("train", scaled=False)
            model.fit(tr_ctx, tr_tgt)
    elif name == "arima":
        from src.models.classical import ClassicalForecaster

        model = ClassicalForecaster(
            horizon=ds.tau, order=None, n_samples=samples, seed=seed
        ).fit(ds.raw_splits["train"])
    elif name in ("deepar", "deepar_notf"):
        from src.models.deepar import DeepARForecaster  # heavy: Colab only

        model = DeepARForecaster(
            horizon=ds.tau, context_length=ds.H, freq=ds.meta["freq"],
            start=ds.meta.get("start_date"), num_layers=args.layers,
            hidden_size=args.hidden, max_epochs=args.epochs,
            num_batches_per_epoch=args.batches, batch_size=args.batch_size,
            n_samples=samples, disable_time_features=(name == "deepar_notf"),
            accelerator=args.accelerator, seed=seed,
        )
        model.fit(ds.to_gluonts("train"))
    elif name == "timegrad":
        from src.models.timegrad import TimeGradForecaster  # heavy: Colab only

        model = TimeGradForecaster(
            target_dim=ds.D, horizon=ds.tau, context_length=ds.H,
            freq=ds.meta["freq"], start=ds.meta.get("start_date"),
            num_cells=args.num_cells, num_layers=args.layers, cell_type=args.cell_type,
            diff_steps=args.diff_steps, beta_end=args.beta_end,
            beta_schedule=args.beta_schedule, max_epochs=args.epochs,
            num_batches_per_epoch=args.batches, batch_size=args.batch_size,
            n_samples=samples, predict_batch_size=args.predict_batch_size,
            input_size=args.input_size, device=args.device, seed=seed,
        )
        model.fit(ds.to_gluonts_multivariate("train"))
    else:
        raise ValueError(f"unknown model '{name}'")
    return model, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="E2 — horizon sweep (CRPS / coverage vs τ).")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    p.add_argument("--models", default="seasonal_naive,arima",
                   help="comma list of: seasonal_naive,arima,deepar,deepar_notf,timegrad")
    p.add_argument("--taus", default="7,15,30,60", help="comma list of horizons to sweep")
    p.add_argument("--samples", type=int, default=100, help="predictive samples S per window")
    p.add_argument("--chunk", type=int, default=0,
                   help="score the test split in chunks of this many windows (0 = eager). "
                        "Use a few hundred for wide data or large τ to bound memory.")
    p.add_argument("--season", type=int, default=None,
                   help="seasonal lag m for MASE (default: eval.season_length from config).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--registry", default=str(REPO_ROOT / "results" / "sweeps" / "horizon.csv"))
    # --- deep-model passthrough (only read for M2/M3 on Colab) ---
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=40, help="DeepAR LSTM hidden size.")
    p.add_argument("--layers", type=int, default=2, help="RNN layers (M2/M3).")
    p.add_argument("--num-cells", type=int, default=64, help="TimeGrad RNN width.")
    p.add_argument("--cell-type", default="GRU")
    p.add_argument("--diff-steps", type=int, default=100)
    p.add_argument("--beta-end", type=float, default=0.1)
    p.add_argument("--beta-schedule", default="linear")
    p.add_argument("--predict-batch-size", type=int, default=16)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--accelerator", default="auto", help="DeepAR/Lightning accelerator.")
    p.add_argument("--device", default="auto", help="TimeGrad device.")
    args = p.parse_args()

    base_cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(base_cfg.get("seed", 42))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    taus = [int(t) for t in args.taus.split(",") if t.strip()]
    season = (args.season if args.season is not None
              else int(base_cfg.get("eval", {}).get("season_length", 1)))

    print(f"[E2 horizon sweep] config={Path(args.config).name}  models={models}  "
          f"taus={taus}  m={season}  ->  {Path(args.registry).name}")

    for tau in taus:
        cfg = copy.deepcopy(base_cfg)
        cfg["window"]["horizon"] = tau
        ds = build_dataset(cfg)
        # τ-independent MASE denominator — identical to the E1 runs, shared by all models.
        scale = seasonal_naive_scale(ds.raw_splits["train"], season_length=season)
        for name in models:
            set_seed(seed)  # same seed at every (τ, model) so only τ moves the curve
            model, fit_s = _build_and_fit(name, ds, season, args.samples, seed, args)
            metrics, n_windows, predict_s = _score(ds, model, scale, args.chunk)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sweep": "horizon",
                "dataset": ds.name,
                "model": name,
                "season_length": season,
                "split": "test",
                "n_windows": n_windows,
                "H": ds.H,
                "tau": tau,
                "D": ds.D,
                "n_samples": args.samples,
                "seed": seed,
                **{k: round(v, 6) for k, v in metrics.items()},
                "fit_s": round(fit_s, 4),
                "predict_s": round(predict_s, 4),
                "platform": platform.platform(),
            }
            append_result(args.registry, row)
            print(f"  τ={tau:>3}  {name:<15}  CRPS={metrics['CRPS']:.5g}  "
                  f"cov50={metrics['cov50']:.3f}  cov90={metrics['cov90']:.3f}  "
                  f"MASE={metrics['MASE']:.4f}   (fit {fit_s:.1f}s · predict {predict_s:.1f}s)")

    print(f"done — rows appended to {args.registry}")


if __name__ == "__main__":
    main()
