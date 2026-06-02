"""E6 — turn M0's forecast into money on Electricity (plan §3.5, §6.4).

This is the runnable demonstration of the economic-value pillar. It needs **no GPU**:
the seasonal-naive M0 forecaster runs locally (pure NumPy + a residual bootstrap), so
we can exercise the whole forecast -> decision -> euros path today, before the L4-trained
M1/M2/M3 forecast arrays are available. The dispatch maths lives in
``src.eval.economic``; this script only wires the dataset to it.

What it does
------------
1. Build the Electricity dataset (cached raw file, offline) and pick the ``--channels``
   client meters with the **strongest daily load swing** — the regime where a battery
   actually has something to arbitrage (an honest, reproducible selection, not
   cherry-picking the result; see ``--channel-select``).
2. Fit M0 and forecast a subsample of test windows -> per (window, channel) episode a
   point path and ``--samples`` sample paths.
3. Build a time-of-use price from each window's hour-of-day, size a battery per channel
   relative to its own mean load, and price every schedule against the **true** future:
   naive (ceiling), oracle (perfect-foresight lower bound), deterministic plan, and the
   stochastic (SAA) plan from the sample paths.
4. Write a money table (``results/economic/value_<dataset>.{md,csv}``) and a saved-€
   figure (``figures/presentation/fig_e6_money.png``).

The cross-model CRPS-vs-€ scatter (E6's punchline) needs M1/M2/M3 sample paths too and
is produced once those land; this v1 establishes the machinery and the M0 baseline.

Run::

    python -m experiments.run_economic                       # Electricity, M0 demo
    python -m experiments.run_economic --channels 8 --windows 200 --samples 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.contract import make_windows  # noqa: E402
from src.data.loader import build_dataset  # noqa: E402
from src.eval.economic import Tariff, evaluate_value, size_battery, tou_price  # noqa: E402
from src.models.naive import SeasonalNaiveForecaster  # noqa: E402
from src.utils.config import load_config  # noqa: E402

OUTDIR = REPO_ROOT / "results" / "economic"
FIGDIR = REPO_ROOT / "figures" / "presentation"


# --------------------------------------------------------------------------- helpers


def _select_channels(train: np.ndarray, k: int, how: str, seed: int) -> np.ndarray:
    """Pick ``k`` channel indices from a (L, D) train array.

    ``topvar`` (default): the ``k`` channels whose average daily profile swings the most
    relative to their level — i.e. where storage has the most to arbitrage. ``even``:
    evenly spaced across the D channels. ``random``: a seeded random sample. The choice
    is reported in the table so the demo is reproducible and not accused of cherry-picking.
    """
    D = train.shape[1]
    k = min(k, D)
    if how == "even":
        return np.linspace(0, D - 1, k).astype(int)
    if how == "random":
        return np.sort(np.random.default_rng(seed).choice(D, size=k, replace=False))
    # topvar: rank by daily-swing amplitude / mean level.
    hours = np.arange(train.shape[0]) % 24  # train starts at the nominal midnight (abs idx 0)
    hourly_mean = np.stack([train[hours == h].mean(axis=0) for h in range(24)])  # (24, D)
    amplitude = hourly_mean.max(axis=0) - hourly_mean.min(axis=0)  # (D,)
    level = np.abs(train.mean(axis=0)) + 1e-9
    swing = amplitude / level
    return np.sort(np.argsort(swing)[::-1][:k])


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:.3f}"


# --------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description="E6 economic value — battery dispatch on M0 forecasts.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "data_electricity.yaml"))
    p.add_argument("--channels", type=int, default=6, help="How many client meters to dispatch.")
    p.add_argument("--channel-select", choices=("topvar", "even", "random"), default="topvar")
    p.add_argument("--windows", type=int, default=120, help="Test windows to price (evenly spaced).")
    p.add_argument("--samples", type=int, default=40, help="Sample paths per window (stochastic plan).")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    # Battery + tariff knobs (the E6 sweep axes; defaults give a sane, arbitrage-able regime).
    p.add_argument("--power-frac", type=float, default=0.5, help="P_max as a fraction of mean load.")
    p.add_argument("--hours", type=float, default=4.0, help="Battery duration E_max / P_max (hours).")
    p.add_argument("--eta", type=float, default=0.9, help="Round-trip efficiency.")
    p.add_argument("--peak", type=float, default=0.30)
    p.add_argument("--shoulder", type=float, default=0.15)
    p.add_argument("--offpeak", type=float, default=0.08)
    p.add_argument("--demand-charge", type=float, default=0.0, help="€/unit on peak import (0 = off).")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    # E6/M0 is pure NumPy — seed locally and skip the torch-touching set_seed so the
    # demo stays GPU-free and never trips over a broken local torch wheel. M0's bootstrap
    # RNG is seeded separately via the model constructor below.
    random.seed(seed)
    np.random.seed(seed)

    ds = build_dataset(cfg)
    H, tau, stride = ds.H, ds.tau, int(ds.meta.get("stride", 1))
    raw_train = ds.raw_splits["train"]
    raw_test = ds.raw_splits["test"]
    # Absolute hour-of-day of a test horizon: the series starts at the config's midnight,
    # so abs index % 24 is the hour. test begins after train+val (contiguous splits).
    test_start_abs = len(raw_train) + len(ds.raw_splits["val"])

    # 1) Channels where storage has something to arbitrage.
    sel = _select_channels(raw_train, args.channels, args.channel_select, seed)
    tr_sel = raw_train[:, sel]
    te_sel = raw_test[:, sel]
    k = len(sel)

    # 2) Fit M0 on the reduced channels and forecast an evenly spaced subsample of windows.
    tr_ctx, tr_tgt = make_windows(tr_sel, H, tau, stride=stride)
    te_ctx, te_tgt = make_windows(te_sel, H, tau, stride=stride)
    n_te = te_ctx.shape[0]
    if n_te == 0:
        raise SystemExit("No test windows — check the dataset/config.")
    widx = np.unique(np.linspace(0, n_te - 1, min(args.windows, n_te)).astype(int))

    model = SeasonalNaiveForecaster(
        season_length=int(cfg.get("eval", {}).get("season_length", 24)),
        horizon=tau,
        n_samples=args.samples,
        seed=seed,
    )
    model.fit(tr_ctx, tr_tgt)
    point, samples = model.predict(te_ctx[widx])  # (n, tau, k), (n, S, tau, k)
    truth = te_tgt[widx]  # (n, tau, k) — settle on the realised future

    # 3) Time-of-use price per selected window (depends only on its hour-of-day phase).
    tariff = Tariff(peak=args.peak, shoulder=args.shoulder, offpeak=args.offpeak)
    horizon_hours = (test_start_abs + widx[:, None] * stride + H + np.arange(tau)[None, :]) % 24
    prices = np.stack([tou_price(hrs, tariff) for hrs in horizon_hours])  # (n, tau)

    # 4) Price each channel with its own right-sized battery; accumulate a money table.
    rows = []
    grand = {kk: 0.0 for kk in ("bill_naive", "bill_oracle", "bill_det", "bill_sto")}
    print(f"E6 economic value · {ds.name} · M0 · {k} channels × {len(widx)} windows · S={args.samples}")
    print(f"  tariff peak/shoulder/offpeak = {args.peak}/{args.shoulder}/{args.offpeak}"
          f"  · battery {args.hours:.0f}h @ {args.power_frac:.2f}·mean, η={args.eta}")
    for j in range(k):
        battery = size_battery(tr_sel[:, j], power_frac=args.power_frac, hours=args.hours, eta=args.eta)
        out = evaluate_value(
            point[:, :, j], truth[:, :, j], prices, battery,
            samples=samples[:, :, :, j], demand_charge=args.demand_charge,
        )
        t = out["totals"]
        rows.append({"channel": int(sel[j]), **t})
        for kk in grand:
            grand[kk] += t[kk]
        print(f"  ch {int(sel[j]):>3d}: naive={_fmt(t['bill_naive'])}  oracle={_fmt(t['bill_oracle'])}  "
              f"det={_fmt(t['bill_det'])} (saved {_fmt(t['saved_det'])}, {100*t['frac_det']:.0f}% of max)  "
              f"sto={_fmt(t['bill_sto'])} (Δdist {_fmt(t['value_of_dist'])})")

    attainable = grand["bill_naive"] - grand["bill_oracle"]
    grand["saved_det"] = grand["bill_naive"] - grand["bill_det"]
    grand["saved_sto"] = grand["bill_naive"] - grand["bill_sto"]
    grand["frac_det"] = grand["saved_det"] / attainable if attainable > 1e-9 else float("nan")
    grand["frac_sto"] = grand["saved_sto"] / attainable if attainable > 1e-9 else float("nan")
    grand["value_of_dist"] = grand["bill_det"] - grand["bill_sto"]

    _write_table(ds.name, rows, grand, sel, args, len(widx))
    _plot(ds.name, rows, grand, sel)

    print(f"\n  TOTAL: naive={_fmt(grand['bill_naive'])}  oracle={_fmt(grand['bill_oracle'])}  "
          f"det saved {_fmt(grand['saved_det'])} ({100*grand['frac_det']:.0f}% of attainable)  "
          f"· value-of-distribution {_fmt(grand['value_of_dist'])}")
    print(f"  table -> {OUTDIR}/value_{ds.name}.md   figure -> {FIGDIR}/fig_e6_money.png")


def _write_table(name, rows, grand, sel, args, n_win) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Valore economico (E6) — {name.capitalize()} · M0 seasonal-naive",
        "",
        "> Generato da `experiments/run_economic.py` (dispatch batteria, plan §6.4). "
        "**Demo M0 in locale** — il confronto fra modelli (CRPS-vs-€) arriva con gli "
        "array di forecast di M1/M2/M3.",
        "",
        f"_dispatch senza export · batteria {args.hours:.0f}h @ {args.power_frac:.2f}·media, "
        f"η={args.eta} · tariffa TOU picco/spalla/notte = {args.peak}/{args.shoulder}/{args.offpeak} · "
        f"{len(sel)} canali ({args.channel_select}) × {n_win} finestre · τ=24_",
        "",
        "Bolletta più bassa = meglio. **denaro risparmiato** = bolletta(naive) − bolletta(piano); "
        "**% del max** = frazione del risparmio ottenibile dall'oracolo (previsione perfetta); "
        "**Δdist** = bolletta(deterministico) − bolletta(stocastico) = valore della distribuzione.",
        "",
        "| Canale | naive | oracolo | det | risparmio det | % del max | sto | Δdist |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['channel']} | {_fmt(r['bill_naive'])} | {_fmt(r['bill_oracle'])} | "
            f"{_fmt(r['bill_det'])} | {_fmt(r['saved_det'])} | {100*r['frac_det']:.0f}% | "
            f"{_fmt(r['bill_sto'])} | {_fmt(r['value_of_dist'])} |"
        )
    lines.append(
        f"| **TOTALE** | **{_fmt(grand['bill_naive'])}** | **{_fmt(grand['bill_oracle'])}** | "
        f"**{_fmt(grand['bill_det'])}** | **{_fmt(grand['saved_det'])}** | "
        f"**{100*grand['frac_det']:.0f}%** | **{_fmt(grand['bill_sto'])}** | "
        f"**{_fmt(grand['value_of_dist'])}** |"
    )
    lines.append("")
    (OUTDIR / f"value_{name}.md").write_text("\n".join(lines), encoding="utf-8")

    # Tidy CSV (one row per channel + TOTAL).
    import csv

    cols = ["channel", "bill_naive", "bill_oracle", "bill_det", "saved_det", "frac_det",
            "bill_sto", "saved_sto", "frac_sto", "value_of_dist"]
    with (OUTDIR / f"value_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
        w.writerow({"channel": "TOTAL", **{c: grand.get(c, "") for c in cols if c != "channel"}})


def _plot(name, rows, grand, sel) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    labels = [str(r["channel"]) for r in rows] + ["TOT"]
    saved_det = [r["saved_det"] for r in rows] + [grand["saved_det"]]
    saved_sto = [r["saved_sto"] for r in rows] + [grand["saved_sto"]]
    frac_det = [100 * r["frac_det"] for r in rows] + [100 * grand["frac_det"]]
    frac_sto = [100 * r["frac_sto"] for r in rows] + [100 * grand["frac_sto"]]
    x = np.arange(len(labels))
    w = 0.38

    fig, (axe, axf) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axe.bar(x - w / 2, saved_det, w, color="#1f77b4", edgecolor="black", linewidth=0.5, label="deterministico")
    axe.bar(x + w / 2, saved_sto, w, color="#d62728", edgecolor="black", linewidth=0.5, label="stocastico (SAA)")
    axe.set_xticks(x)
    axe.set_xticklabels(labels, fontsize=8)
    axe.set_ylabel("denaro risparmiato vs naive (€)")
    axe.set_title("Quanto vale il forecast (M0) — euro risparmiati")
    axe.axvline(len(labels) - 1.5, ls=":", color="#888", lw=1.0)
    axe.legend(fontsize=9)

    axf.bar(x - w / 2, frac_det, w, color="#1f77b4", edgecolor="black", linewidth=0.5, label="deterministico")
    axf.bar(x + w / 2, frac_sto, w, color="#d62728", edgecolor="black", linewidth=0.5, label="stocastico (SAA)")
    axf.axhline(100, ls="--", color="#333", lw=1.0)
    axf.text(len(labels) - 0.5, 100, " oracolo = 100%", va="bottom", ha="right", fontsize=8, color="#333")
    axf.set_xticks(x)
    axf.set_xticklabels(labels, fontsize=8)
    axf.set_ylabel("% del risparmio ottenibile (oracolo)")
    axf.set_title("Quota del risparmio dell'oracolo catturata")
    axf.axvline(len(labels) - 1.5, ls=":", color="#888", lw=1.0)
    axf.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"E6 · valore economico del forecast (dispatch batteria) — {name} · M0",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGDIR / "fig_e6_money.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
