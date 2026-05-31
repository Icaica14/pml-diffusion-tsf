"""Run the M0 seasonal-naive baseline and append its row to the results registry.

This is the first rung of the model ladder and the first real entry in
``results/registry.csv``. It establishes the conventions every later experiment
follows: build the :class:`ForecastDataset` from a config, forecast the **test**
windows on the **original** scale, score with the shared metrics module, and append
exactly one provenance-tagged row.

Usage::

    python -m experiments.run_naive                                         # Exchange, m=1 (from config)
    python -m experiments.run_naive --config configs/data_electricity.yaml  # Electricity, m=24 (from config)
    python -m experiments.run_naive --season 168 --samples 200              # weekly-naive ablation
"""

from __future__ import annotations

import argparse
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
from src.models.naive import SeasonalNaiveForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M0 seasonal-naive baseline.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Seasonal lag m. Default: read eval.season_length from the config "
        "(1 for Exchange's near-random-walk; 24 for hourly Electricity). Pass an "
        "explicit value to override for an ablation (e.g. 168 = weekly naive).",
    )
    parser.add_argument("--samples", type=int, default=100, help="Bootstrap samples per window.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Score the test split in chunks of this many windows (0 = eager, build "
        "all windows at once). Use a few hundred for wide data like Electricity "
        "(D=321), whose full (N, S, tau, D) sample tensor is hundreds of GB.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    ds = build_dataset(cfg)

    # Seasonal lag m: CLI override, else the dataset's eval.season_length (the value
    # M0's forecast AND every model's MASE denominator share, for comparability).
    season = args.season if args.season is not None else int(cfg.get("eval", {}).get("season_length", 1))

    # MASE denominator (data-level constant, original scale) — shared by every model.
    scale = seasonal_naive_scale(ds.raw_splits["train"], season)

    model = SeasonalNaiveForecaster(
        season_length=season,
        horizon=ds.tau,
        n_samples=args.samples,
        seed=seed,
    )

    if args.chunk and args.chunk > 0:
        # Memory-safe path for wide data (Electricity): stream train + test windows so
        # neither the (M, H, D) train tensor nor the (N, S, tau, D) test samples are
        # ever held whole. Numerically identical to the eager path (see ForecastEvaluator).
        model.fit_chunked(ds.iter_windows("train", args.chunk, scaled=False))
        t0 = time.perf_counter()
        metrics, n_windows = evaluate_streaming(
            ds, model, "test", scale, args.chunk, scaled=False, levels=(0.5, 0.9)
        )
        predict_s = time.perf_counter() - t0
    else:
        # Eager path: small/narrow data (Exchange) — metrics on the ORIGINAL scale.
        tr_ctx, tr_tgt = ds.windows("train", scaled=False)
        te_ctx, te_tgt = ds.windows("test", scaled=False)
        model.fit(tr_ctx, tr_tgt)
        t0 = time.perf_counter()
        point, samples = model.predict(te_ctx)
        predict_s = time.perf_counter() - t0
        metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))
        n_windows = int(te_tgt.shape[0])

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": "seasonal_naive",
        "season_length": season,
        "split": "test",
        "n_windows": n_windows,
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Provenance: pin exactly how the data was built alongside the numbers.
    ds.save_manifest(REPO_ROOT / "results" / f"manifest_{ds.name}.json")

    # Console summary.
    print(f"M0 seasonal-naive (m={season})  ·  {ds.name}  ·  test split")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}")
    print("  point   : "
          f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
    print("  prob    : "
          f"CRPS={metrics['CRPS']:.4f}  pinball={metrics['pinball']:.4f}")
    print("  calib   : "
          f"cov50={metrics['cov50']:.3f} (width {metrics['width50']:.4f})  "
          f"cov90={metrics['cov90']:.3f} (width {metrics['width90']:.4f})")
    print(f"  predict : {predict_s:.3f}s  ->  appended to {Path(args.registry).name}")


if __name__ == "__main__":
    main()
