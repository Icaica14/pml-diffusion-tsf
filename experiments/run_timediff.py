"""Run the M4 TimeDiff model and append its row to the results registry.

The **second diffusion rung** and the non-autoregressive counterpart to M3 TimeGrad.
It follows the exact conventions of ``run_timegrad.py`` — build the
:class:`ForecastDataset` from a config, train on the **train** split, forecast the
**test** windows on the **original** scale, score with the shared metrics module, append
one provenance-tagged row — so the M0..M4 numbers are comparable cell-for-cell.

Unlike M3, TimeDiff is **self-contained PyTorch** (no GluonTS/PyTorchTS): it trains by
random-cropping ``H+tau`` windows straight from ``ds.raw_splits["train"]`` and
standardizes internally, so the runner hands it the raw train series rather than a
GluonTS dataset. Per-window forecasts reuse the trained network with no refit
(leakage-free, identical discipline to M1/M2/M3).

Usage (on the Colab/GPU box; ``--smoke`` runs anywhere torch imports)::

    python -m experiments.run_timediff                                         # Exchange, default budget
    python -m experiments.run_timediff --config configs/data_electricity.yaml --chunk 256  # Electricity
    python -m experiments.run_timediff --epochs 50 --samples 100 --device cuda
    python -m experiments.run_timediff --param eps --config configs/data_electricity.yaml --chunk 256  # eps ablation -> 'timediff_eps'
    python -m experiments.run_timediff --smoke --config configs/data_electricity.yaml      # fast plumbing check
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
from src.models.timediff import TimeDiffForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def _resolve_device(choice: str) -> str:
    """Map ``--device auto`` to 'cuda' when a GPU is visible, else 'cpu'."""
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_model(ds, args, device, seed) -> TimeDiffForecaster:
    return TimeDiffForecaster(
        target_dim=ds.D,
        horizon=ds.tau,
        context_length=ds.H,
        freq=ds.meta["freq"],
        start=ds.meta.get("start_date"),
        diff_steps=args.diff_steps,
        beta_schedule=args.beta_schedule,
        beta_end=args.beta_end,
        parameterization=args.param,
        hidden=args.hidden,
        n_res_blocks=args.blocks,
        kernel=args.kernel,
        mixup_prob=args.mixup,
        max_epochs=args.epochs,
        num_batches_per_epoch=args.batches,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.samples,
        n_sample_steps=args.sample_steps,
        eta=args.eta,
        predict_batch_size=args.predict_batch_size,
        clip_x0=args.clip_x0,
        device=device,
        seed=seed,
    )


def _smoke(ds, args, device, seed, season) -> None:
    """Tiny CPU/GPU-friendly fit→predict→score on a few test windows.

    Validates the whole plumbing (shapes, metric finiteness, registry-free) in seconds
    before committing a multi-hour GPU run. Does **not** append to the registry.
    """
    print(f"SMOKE  ·  {ds.name}  ·  tiny net, {args.smoke_windows} test windows, device={device}")
    model = TimeDiffForecaster(
        target_dim=ds.D, horizon=ds.tau, context_length=ds.H,
        freq=ds.meta["freq"], start=ds.meta.get("start_date"),
        diff_steps=20, beta_schedule=args.beta_schedule,
        parameterization=args.param,
        hidden=16, n_res_blocks=2, mixup_prob=args.mixup,
        max_epochs=2, num_batches_per_epoch=5, batch_size=16, lr=args.lr,
        n_samples=8, n_sample_steps=20, eta=args.eta, predict_batch_size=4,
        clip_x0=args.clip_x0, device=device, seed=seed,
    )
    model.fit(ds.raw_splits["train"])
    ctx, tgt = next(ds.iter_windows("test", chunk_size=args.smoke_windows, scaled=False))
    point, samples = model.predict(ctx)
    scale = seasonal_naive_scale(ds.raw_splits["train"], season_length=season)
    metrics = evaluate_forecast(tgt, point, samples, scale, levels=(0.5, 0.9))
    import numpy as np

    ok = point.shape == (ctx.shape[0], ds.tau, ds.D) and samples.shape == (ctx.shape[0], model.n_samples, ds.tau, ds.D)
    finite = bool(np.isfinite(list(metrics.values())).all())
    print(f"  point {point.shape}  samples {samples.shape}  shapes_ok={ok}")
    print(f"  MASE={metrics['MASE']:.3f}  CRPS={metrics['CRPS']:.3f}  "
          f"cov50={metrics['cov50']:.3f}  cov90={metrics['cov90']:.3f}  finite={finite}")
    print("SMOKE OK" if (ok and finite) else "SMOKE FAILED — investigate before the real run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M4 TimeDiff model.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_exchange.yaml"))
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batches", type=int, default=100, help="num_batches_per_epoch.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64, help="Denoiser channel width.")
    parser.add_argument("--blocks", type=int, default=4, help="Residual conv blocks.")
    parser.add_argument("--kernel", type=int, default=3, help="Conv kernel size.")
    parser.add_argument("--diff-steps", type=int, default=100, help="Diffusion steps T.")
    parser.add_argument("--beta-schedule", default="cosine", help="'cosine' or 'linear'.")
    parser.add_argument("--beta-end", type=float, default=0.1, help="beta_end for the linear schedule.")
    parser.add_argument("--param", default="x0", choices=["x0", "eps"],
                        help="Denoiser target: 'x0' (default, the banked M4 row) or 'eps' "
                             "(noise-prediction calibration ablation; logs as model 'timediff_eps').")
    parser.add_argument("--mixup", type=float, default=0.5,
                        help="Future-mixup keep-prob for x_ar (1.0 disables; inference uses x_ar).")
    parser.add_argument("--samples", type=int, default=100, help="Sampled trajectories per window.")
    parser.add_argument("--sample-steps", type=int, default=100,
                        help="Reverse steps walked (>= diff_steps = full chain; less = DDIM respacing).")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM stochasticity (1.0 ~ ancestral spread, 0.0 deterministic).")
    parser.add_argument("--predict-batch-size", type=int, default=16,
                        help="Windows per GPU batch at sampling time (memory cap; each "
                             "window expands to --samples chains). Lower it if the sampler OOMs.")
    parser.add_argument("--clip-x0", type=float, default=5.0,
                        help="Clamp predicted x0 to +/- this (standardized) each step; <0 disables.")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--season", type=int, default=None,
                        help="Seasonal lag m for the MASE denominator. Default: eval.season_length.")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Score the test split in chunks of this many windows (0 = eager). "
                             "Required for Electricity (D=321): the full (N,S,tau,D) tensor is huge.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "results" / "registry.csv"))
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny fast fit→predict→score on a few windows; no registry write.")
    parser.add_argument("--smoke-windows", type=int, default=16, help="Test windows scored in --smoke.")
    args = parser.parse_args()
    if args.clip_x0 is not None and args.clip_x0 < 0:
        args.clip_x0 = None

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = _resolve_device(args.device)
    ds = build_dataset(cfg)
    season = args.season if args.season is not None else int(cfg.get("eval", {}).get("season_length", 1))

    if args.smoke:
        _smoke(ds, args, device, seed, season)
        return

    model = _build_model(ds, args, device, seed)

    t_fit = time.perf_counter()
    model.fit(ds.raw_splits["train"])
    fit_s = time.perf_counter() - t_fit

    # MASE denominator is a data-level constant: identical to M0..M3 for comparability.
    scale = seasonal_naive_scale(ds.raw_splits["train"], season_length=season)

    if args.chunk and args.chunk > 0:
        t0 = time.perf_counter()
        metrics, n_windows = evaluate_streaming(
            ds, model, "test", scale, args.chunk, scaled=False, levels=(0.5, 0.9)
        )
        predict_s = time.perf_counter() - t0
    else:
        te_ctx, te_tgt = ds.windows("test", scaled=False)
        t0 = time.perf_counter()
        point, samples = model.predict(te_ctx)
        predict_s = time.perf_counter() - t0
        metrics = evaluate_forecast(te_tgt, point, samples, scale, levels=(0.5, 0.9))
        n_windows = int(te_tgt.shape[0])

    model_name = "timediff_eps" if args.param == "eps" else "timediff"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": ds.name,
        "model": model_name,
        "season_length": season,
        "split": "test",
        "n_windows": n_windows,
        "H": ds.H,
        "tau": ds.tau,
        "D": ds.D,
        "n_samples": args.samples,
        "epochs": args.epochs,
        "diff_steps": args.diff_steps,
        "beta_schedule": args.beta_schedule,
        "n_res_blocks": args.blocks,
        "hidden": args.hidden,
        "n_sample_steps": args.sample_steps,
        "eta": args.eta,
        "seed": seed,
        **{k: round(v, 6) for k, v in metrics.items()},
        "fit_s": round(fit_s, 4),
        "predict_s": round(predict_s, 4),
        "platform": platform.platform(),
    }
    append_result(args.registry, row)

    # Console summary.
    print(f"M4 TimeDiff [{args.param}]  ·  {ds.name}  ·  test split  ·  model={model_name}")
    print(f"  windows={row['n_windows']}  H={ds.H}  tau={ds.tau}  D={ds.D}  "
          f"S={args.samples}  seed={seed}  device={device}")
    print(f"  model   : hidden={args.hidden} blocks={args.blocks} kernel={args.kernel}  "
          f"diff_steps={args.diff_steps} schedule={args.beta_schedule} mixup={args.mixup}")
    print(f"  sample  : steps={args.sample_steps} eta={args.eta}  train_loss={model.train_loss_:.5f}")
    print(f"  train   : epochs={args.epochs} x {args.batches} batches of {args.batch_size}")
    print("  point   : "
          f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  MASE={metrics['MASE']:.4f}")
    print("  prob    : "
          f"CRPS={metrics['CRPS']:.4f}  pinball={metrics['pinball']:.4f}")
    print("  calib   : "
          f"cov50={metrics['cov50']:.3f} (width {metrics['width50']:.4f})  "
          f"cov90={metrics['cov90']:.3f} (width {metrics['width90']:.4f})")
    print(f"  time    : fit {fit_s:.1f}s  predict {predict_s:.1f}s  "
          f"->  appended to {Path(args.registry).name}")


if __name__ == "__main__":
    main()
