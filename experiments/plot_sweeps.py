"""Render the two **sweep** figures for the deck: E2 (horizon) and E3 (denoising steps).

Both read the CSVs produced by the sweep runners and write PNGs to
``figures/presentation/``. Each figure is skipped (with a note) if its CSV is missing,
so the script is safe to run after only one of the two sweeps.

  * **E2** ``results/sweeps/horizon.csv`` -> ``fig_e2_horizon.png``
      Left:  CRPS vs horizon τ, one line per model (lower = better).
      Right: empirical coverage vs τ — 90% (solid) and 50% (dotted) against their
             nominal dashed references; tells whether calibration holds as τ grows.
      The story E2 asks for: how probabilistic quality degrades with the horizon, and
      whether the ranking between models changes along the way.

  * **E3** ``results/sweeps/steps.csv`` -> ``fig_e3_steps.png``
      Left:  CRPS vs number of reverse steps (log-x) — the **quality elbow**; one curve
             per DDIM ``eta``. A dashed floor marks the best (full-step) CRPS and the
             "elbow" step count (cheapest within 2% of the floor) is annotated.
      Right: sampling wall-clock vs steps — **linear in steps**, the price you pay.
      Together: most of the quality is bought in the first handful of steps, while cost
      keeps climbing — the trade-off E3 is about.

Colours follow the project convention (M0 grey, M1 blue, M2 orange, M3 red).

Run::

    python3 -m experiments.plot_sweeps
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEPDIR = REPO_ROOT / "results" / "sweeps"
FIGDIR = REPO_ROOT / "figures" / "presentation"

# Project-wide model identity (same as experiments/plot_presentation.py).
CODE = {"seasonal_naive": "M0", "arima": "M1", "deepar": "M2",
        "deepar_notf": "M2-nf", "timegrad": "M3"}
LABEL = {"seasonal_naive": "M0  seasonal-naive", "arima": "M1  ARIMA",
         "deepar": "M2  DeepAR", "deepar_notf": "M2-nf  DeepAR (no tf)",
         "timegrad": "M3  TimeGrad"}
COLOR = {"seasonal_naive": "#7f7f7f", "arima": "#1f77b4", "deepar": "#ff7f0e",
         "deepar_notf": "#ffbb78", "timegrad": "#d62728"}
ORDER = ["seasonal_naive", "arima", "deepar", "deepar_notf", "timegrad"]

# E3 eta curves: deterministic DDIM (green) vs ancestral/DDPM-like (diffusion red).
ETA_COLOR = {0.0: "#2ca02c", 1.0: "#d62728"}
ETA_LABEL = {0.0: "η=0  DDIM deterministico", 1.0: "η=1  ancestrale (DDPM)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------------------
# E2 — horizon sweep
# ---------------------------------------------------------------------------------------
def fig_horizon() -> bool:
    path = SWEEPDIR / "horizon.csv"
    if not path.exists():
        print(f"[E2] skip — {path.relative_to(REPO_ROOT)} not found")
        return False
    df = pd.read_csv(path)
    if df.empty:
        print("[E2] skip — horizon.csv is empty")
        return False
    dataset = df["dataset"].iloc[0]
    models = [m for m in ORDER if m in set(df["model"])]

    fig, (axc, axk) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for m in models:
        sub = df[df["model"] == m].sort_values("tau")
        col = COLOR[m]
        axc.plot(sub["tau"], sub["CRPS"], "-o", color=col, lw=2.0, ms=7,
                 label=LABEL[m])
        # coverage panel: 90% solid, 50% dotted, same colour per model
        axk.plot(sub["tau"], sub["cov90"], "-o", color=col, lw=2.0, ms=6)
        axk.plot(sub["tau"], sub["cov50"], ":s", color=col, lw=1.6, ms=5, alpha=0.9)

    axc.set_xlabel("orizzonte di previsione  τ  (passi avanti)")
    axc.set_ylabel("CRPS  (più basso = meglio)")
    axc.set_title(f"E2 — qualità vs orizzonte ({dataset}, test)")
    axc.legend(fontsize=9, loc="upper left")

    axk.axhline(0.9, ls="--", color="#333", lw=1.0)
    axk.axhline(0.5, ls="--", color="#333", lw=1.0)
    axk.text(df["tau"].max(), 0.9, " nominale 90%", va="bottom", ha="right",
             fontsize=8, color="#333")
    axk.text(df["tau"].max(), 0.5, " nominale 50%", va="bottom", ha="right",
             fontsize=8, color="#333")
    axk.set_xlabel("orizzonte di previsione  τ")
    axk.set_ylabel("copertura empirica")
    axk.set_title("E2 — calibrazione vs orizzonte")
    axk.set_ylim(0, 1.02)
    # style-only legend explaining the two line styles
    style_handles = [
        plt.Line2D([0], [0], color="#333", ls="-", marker="o", label="copertura 90%"),
        plt.Line2D([0], [0], color="#333", ls=":", marker="s", label="copertura 50%"),
    ]
    axk.legend(handles=style_handles, fontsize=9, loc="lower left")

    fig.suptitle("E2 — degrado della previsione probabilistica con l'orizzonte τ",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGDIR / "fig_e2_horizon.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[E2] wrote {out.relative_to(REPO_ROOT)}  ({len(models)} models, "
          f"τ={sorted(df['tau'].unique().tolist())})")
    return True


# ---------------------------------------------------------------------------------------
# E3 — denoising-steps sweep
# ---------------------------------------------------------------------------------------
def _elbow(steps: np.ndarray, crps: np.ndarray, tol: float = 0.02) -> int:
    """Smallest step count whose CRPS is within ``tol`` (relative) of the best CRPS."""
    floor = float(np.min(crps))
    ok = crps <= floor * (1.0 + tol)
    return int(np.min(steps[ok])) if ok.any() else int(steps[np.argmin(crps)])


def fig_steps() -> bool:
    path = SWEEPDIR / "steps.csv"
    if not path.exists():
        print(f"[E3] skip — {path.relative_to(REPO_ROOT)} not found")
        return False
    df = pd.read_csv(path)
    if df.empty:
        print("[E3] skip — steps.csv is empty")
        return False
    etas = sorted(df["eta"].unique().tolist())

    fig, (axq, axt) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # --- left: CRPS elbow ---
    for eta in etas:
        sub = df[df["eta"] == eta].sort_values("n_sample_steps")
        steps = sub["n_sample_steps"].to_numpy(float)
        crps = sub["CRPS"].to_numpy(float)
        col = ETA_COLOR.get(eta, "#444")
        axq.plot(steps, crps, "-o", color=col, lw=2.0, ms=7,
                 label=ETA_LABEL.get(eta, f"η={eta}"))
        floor = float(np.min(crps))
        axq.axhline(floor, ls="--", color=col, lw=1.0, alpha=0.7)
        kE = _elbow(steps, crps)
        cE = float(crps[steps == kE][0])
        axq.scatter([kE], [cE], s=180, facecolor="none", edgecolor=col,
                    linewidth=2.2, zorder=5)
        axq.annotate(f"elbow ≈ {kE} passi", (kE, cE),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=9, fontweight="bold", color=col)
    axq.set_xscale("log")
    axq.set_xlabel("passi di reverse (denoising)  —  log")
    axq.set_ylabel("CRPS  (più basso = meglio)")
    axq.set_title("E3 — elbow di qualità: la qualità satura presto")
    axq.legend(fontsize=9, loc="upper right")

    # --- right: linear cost ---
    cost = df.groupby("n_sample_steps")["predict_s"].mean().sort_index()
    steps = cost.index.to_numpy(float)
    secs = cost.to_numpy(float)
    axt.plot(steps, secs, "-o", color="#333", lw=2.0, ms=7, label="tempo misurato")
    # linear reference through the origin fitted on the measured points
    slope = float(np.sum(steps * secs) / np.sum(steps * steps))
    axt.plot(steps, slope * steps, ls="--", color="#999", lw=1.4,
             label=f"∝ passi  ({slope*1e3:.1f} ms/passo)")
    axt.set_xscale("log")
    axt.set_xlabel("passi di reverse (denoising)  —  log")
    axt.set_ylabel("tempo di campionamento (s)")
    axt.set_title("E3 — il costo cresce ~linearmente coi passi")
    axt.legend(fontsize=9, loc="upper left")

    n_test = int(df["n_test"].iloc[0]) if "n_test" in df else None
    s = int(df["n_samples"].iloc[0]) if "n_samples" in df else None
    sub = "toy DDPM (pure-NumPy)  ·  target bimodale p(x|c)"
    if n_test and s:
        sub += f"  ·  {n_test} condizioni × {s} campioni"
    fig.suptitle("E3 — quanti passi di denoising servono davvero?", fontweight="bold")
    fig.text(0.5, 0.925, sub, ha="center", va="top", fontsize=9, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGDIR / "fig_e3_steps.png"
    fig.savefig(out)
    plt.close(fig)
    msg = "  ".join(
        f"η={eta}: elbow≈{_elbow(df[df.eta==eta].sort_values('n_sample_steps')['n_sample_steps'].to_numpy(float), df[df.eta==eta].sort_values('n_sample_steps')['CRPS'].to_numpy(float))}"
        for eta in etas
    )
    print(f"[E3] wrote {out.relative_to(REPO_ROOT)}  ({msg})")
    return True


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = sum([fig_horizon(), fig_steps()])
    print(f"done — {made} sweep figure(s) in {FIGDIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
