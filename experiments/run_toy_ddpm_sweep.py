"""E3 — denoising-steps sweep, run **locally** on the pure-NumPy toy DDPM.

E3 asks the question every diffusion practitioner faces at deploy time: *how many
reverse (denoising) steps do I actually need?* Sampling cost is **linear** in the number
of steps, but quality typically **plateaus** after a handful — so there is an "elbow"
where you buy almost all the quality for a fraction of the cost. The real experiment runs
on TimeGrad's ``diff_steps`` on a GPU (see the Colab cells in
``docs/presentation/SWEEPS_E2_E3_IT.md``); this script reproduces the *same curve*
cheaply, transparently, and with zero heavy dependencies using
``src/models/toy_ddpm.py`` — so the mechanism is defensible at the oral even without a
GPU in the room.

What it does:
  1. trains **one** ``ToyConditionalDDPM`` on the bimodal toy target ``p(x | c)``;
  2. for each sampler setting in the grid (number of DDIM steps × ``eta``) it draws
     ``S`` samples per test condition, **times** the sampling, and scores **CRPS**
     (sample-based, the same probabilistic metric used on the real models) and MAE of
     the predictive median;
  3. appends one row per ``(eta, n_sample_steps)`` to ``results/sweeps/steps.csv``.

Two ``eta`` values are swept by default so the figure can contrast the **deterministic
DDIM** path (``eta=0`` — DDIM's selling point: fewer steps for the same quality) with
the **ancestral / DDPM-like** path (``eta=1``). The cost (time) depends on the step
count, not on ``eta``.

``experiments/plot_sweeps.py`` turns ``steps.csv`` into
``figures/presentation/fig_e3_steps.png`` (CRPS-vs-steps elbow + time-vs-steps line).

Usage::

    # the default local sweep (a couple of minutes, pure NumPy, no GPU)
    python -m experiments.run_toy_ddpm_sweep

    # a quicker pass for iterating on the figure
    python -m experiments.run_toy_ddpm_sweep --n-test 300 --samples 50 \
        --steps-grid 5,10,25,50,100
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.registry import append_result  # noqa: E402
from src.models.toy_ddpm import (  # noqa: E402
    ToyConditionalDDPM,
    crps_samples,
    make_conditional_data,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="E3 — denoising-steps sweep on the pure-NumPy toy DDPM "
                    "(CRPS & time vs number of reverse steps)."
    )
    p.add_argument("--steps-grid", default="2,3,5,8,12,20,35,60,100",
                   help="comma list of reverse-step counts to sweep (the E3 knob).")
    p.add_argument("--eta", default="0.0,1.0",
                   help="comma list of DDIM eta values (0 = deterministic DDIM, "
                        "1 = ancestral/DDPM-like). Each is a separate curve.")
    p.add_argument("--n-diffusion", type=int, default=100,
                   help="training diffusion steps T (caps the sweep grid).")
    p.add_argument("--train-steps", type=int, default=4000, help="optimiser steps.")
    p.add_argument("--batch", type=int, default=256, help="training minibatch size.")
    p.add_argument("--n-train", type=int, default=8000, help="toy training pairs.")
    p.add_argument("--n-test", type=int, default=500, help="toy test conditions.")
    p.add_argument("--samples", type=int, default=100, help="predictive samples S.")
    p.add_argument("--hidden", default="64,64", help="MLP hidden widths.")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--registry",
                   default=str(REPO_ROOT / "results" / "sweeps" / "steps.csv"))
    args = p.parse_args()

    grid = [int(s) for s in args.steps_grid.split(",") if s.strip()]
    etas = [float(e) for e in args.eta.split(",") if e.strip()]
    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())

    # One dataset, one trained model — shared across the whole sweep so that only the
    # *sampler* changes along the curve (the same discipline as E2's "only τ moves").
    data_rng = np.random.default_rng(args.seed)
    c_tr, x_tr = make_conditional_data(args.n_train, data_rng)
    c_te, y_te = make_conditional_data(args.n_test, data_rng)

    print(f"[E3 steps sweep] toy DDPM  T={args.n_diffusion}  hidden={hidden}  "
          f"train_steps={args.train_steps}  n_test={args.n_test}  S={args.samples}")
    model = ToyConditionalDDPM(
        n_steps=args.n_diffusion, hidden=hidden, lr=args.lr, seed=args.seed
    )
    t0 = time.perf_counter()
    model.train(c_tr, x_tr, steps=args.train_steps, batch=args.batch)
    fit_s = time.perf_counter() - t0
    print(f"  trained in {fit_s:.1f}s  ->  sweeping {len(etas)}×{len(grid)} settings "
          f"->  {Path(args.registry).name}")

    for eta in etas:
        for k in grid:
            t0 = time.perf_counter()
            samples = model.sample(c_te, n_samples=args.samples,
                                   n_sample_steps=k, eta=eta)
            predict_s = time.perf_counter() - t0
            crps = crps_samples(samples, y_te)
            point = np.median(samples, axis=1)
            mae = float(np.mean(np.abs(point - y_te)))
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sweep": "steps",
                "model": "toy_ddpm",
                "dataset": "toy_bimodal",
                "n_diffusion": args.n_diffusion,
                "eta": eta,
                "n_sample_steps": k,
                "n_test": args.n_test,
                "n_samples": args.samples,
                "seed": args.seed,
                "CRPS": round(crps, 6),
                "MAE": round(mae, 6),
                "fit_s": round(fit_s, 4),
                "predict_s": round(predict_s, 4),
                "ms_per_sample": round(1e3 * predict_s / (args.n_test * args.samples), 5),
                "platform": platform.platform(),
            }
            append_result(args.registry, row)
            print(f"  eta={eta:<3}  steps={k:>3}  CRPS={crps:.4f}  MAE={mae:.4f}  "
                  f"time={predict_s:6.2f}s")

    print(f"done — rows appended to {args.registry}")


if __name__ == "__main__":
    main()
