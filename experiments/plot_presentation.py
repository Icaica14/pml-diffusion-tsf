"""Presentation figures for the **Electricity** deck — with M3 placeholder handling.

Unlike ``experiments/plot_results.py`` (Exchange sandbox, no dataset filter), this
script is **dataset-aware**: it keeps only ``dataset == "electricity"`` rows from
``results/registry.csv``. Its whole reason to exist is FASE A of the presentation: the
M3 TimeGrad run is still on the GPU, so its row is not in the registry yet.

Behaviour:
  * If the Electricity ``timegrad`` row is **absent**, M3 is drawn from the
    PLACEHOLDER values (interval midpoints from
    ``docs/presentation/RESULTS_PLACEHOLDERS_ELECTRICITY.md``) and visibly marked:
    hatched/dashed bars, hollow scatter marker, and a "PLACEHOLDER M3" banner.
  * If the row is **present** (FASE B, after the run), the real numbers are used and
    every placeholder mark disappears automatically. Just re-run the same command.

Figures written to ``figures/presentation/`` (test split, Electricity):
  fig_cmp_crps.png           CRPS bars + the "barra da battere" = M0's CRPS
  fig_cmp_mase.png           MASE bars + the 1.0 persistence reference line
  fig_cmp_calibration.png    reliability: nominal vs empirical coverage @50/90
  fig_cmp_intervals.png      coverage vs nominal + interval width (sharpness)
  fig_cmp_cost.png           fit_s and predict_s (log) wall-clock
  fig_cmp_quality_cost.png   the thesis: CRPS vs predict_s (log-x)

Run::

    python3 -m experiments.plot_presentation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "results" / "registry.csv"
FIGDIR = REPO_ROOT / "figures" / "presentation"
DATASET = "electricity"

# Models shown on the Electricity deck, in fixed order. deepar_notf is included only
# if a row happens to exist (no ablation was run on Electricity by default).
ORDER = ["seasonal_naive", "arima", "deepar", "deepar_notf", "timegrad", "timediff"]
CODE = {
    "seasonal_naive": "M0",
    "arima": "M1",
    "deepar": "M2",
    "deepar_notf": "M2-nf",
    "timegrad": "M3",
    "timediff": "M4",
}
LABEL = {
    "seasonal_naive": "M0  seasonal-naive",
    "arima": "M1  ARIMA",
    "deepar": "M2  DeepAR",
    "deepar_notf": "M2-nf  DeepAR (no time-feat)",
    "timegrad": "M3  TimeGrad",
    "timediff": "M4  TimeDiff",
}
COLOR = {
    "seasonal_naive": "#7f7f7f",
    "arima": "#1f77b4",
    "deepar": "#ff7f0e",
    "deepar_notf": "#ffbb78",
    "timegrad": "#d62728",
    "timediff": "#9467bd",
}

# --- M3 PLACEHOLDER (interval midpoints from RESULTS_PLACEHOLDERS_ELECTRICITY.md) -----
# These are NOT measurements. They vanish the moment a real electricity+timegrad row
# lands in the registry. width50/width90 are not given in the RESULTS file; the values
# here are plausible stand-ins (near DeepAR's) only so the sharpness panel can draw.
PLACEHOLDER_M3 = {
    "MASE": 1.53,
    "CRPS": 210.0,
    "pinball": 110.0,
    "cov50": 0.49,
    "cov90": 0.87,
    "MAE": 300.0,
    "RMSE": 2300.0,
    "width50": 430.0,   # placeholder (not in registry)
    "width90": 1150.0,  # placeholder (not in registry)
    "fit_s": 1800.0,    # ~30 min
    "predict_s": 10800.0,  # ~3 h
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def _load() -> tuple[pd.DataFrame, bool]:
    """Return (electricity dataframe in ORDER, m3_is_placeholder)."""
    df = pd.read_csv(REGISTRY)
    df = df[(df["dataset"] == DATASET) & (df["model"].isin(ORDER))].copy()

    m3_present = (df["model"] == "timegrad").any()
    if not m3_present:
        row = {c: np.nan for c in df.columns}
        row["dataset"] = DATASET
        row["model"] = "timegrad"
        row.update(PLACEHOLDER_M3)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df["is_ph"] = (df["model"] == "timegrad") & (not m3_present)
    df["model"] = pd.Categorical(df["model"], categories=ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    return df, (not m3_present)


def _codes(df: pd.DataFrame) -> list[str]:
    return [CODE[m] for m in df["model"]]


def _colors(df: pd.DataFrame) -> list[str]:
    return [COLOR[m] for m in df["model"]]


def _mark_placeholder_bars(ax, bars, df) -> None:
    """Hatch + dashed edge on any placeholder bar, and a small 'PLACEHOLDER' tag."""
    for b, is_ph in zip(bars, df["is_ph"]):
        if not is_ph:
            continue
        b.set_hatch("////")
        b.set_edgecolor("black")
        b.set_linewidth(1.1)
        b.set_linestyle("--")
        b.set_alpha(0.65)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, "PLACEHOLDER",
                rotation=90, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="black")


def _bar_labels(ax, bars, values, fmt="{:.2f}", dy=0.0):
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=9)


def _ph_banner(fig, is_ph: bool) -> None:
    if is_ph:
        fig.text(0.5, 0.005,
                 "M3 = PLACEHOLDER  [aggiornare appena disponibile results/registry.csv]",
                 ha="center", va="bottom", fontsize=8.5, color="#b22222",
                 fontweight="bold")


def fig_crps(df: pd.DataFrame, is_ph: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    vals = df["CRPS"].to_numpy(dtype=float)
    bars = ax.bar(_codes(df), vals, color=_colors(df), edgecolor="black", linewidth=0.6)
    _mark_placeholder_bars(ax, bars, df)
    _bar_labels(ax, bars, vals, fmt="{:.1f}", dy=vals.max() * 0.01)
    # "barra da battere": M0 (seasonal-naive) CRPS
    naive = float(df.loc[df["model"] == "seasonal_naive", "CRPS"].iloc[0])
    ax.axhline(naive, ls="--", color="#7f7f7f", lw=1.3)
    ax.text(len(df) - 0.5, naive, f"  barra da battere = {naive:.1f} (M0)",
            color="#444", va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("CRPS  (più basso = meglio)")
    ax.set_title("Qualità probabilistica — CRPS (Electricity, test)")
    ax.set_ylim(0, vals.max() * 1.18)
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGDIR / "fig_cmp_crps.png")
    plt.close(fig)


def fig_mase(df: pd.DataFrame, is_ph: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    vals = df["MASE"].to_numpy(dtype=float)
    bars = ax.bar(_codes(df), vals, color=_colors(df), edgecolor="black", linewidth=0.6)
    _mark_placeholder_bars(ax, bars, df)
    _bar_labels(ax, bars, vals, fmt="{:.2f}", dy=vals.max() * 0.01)
    ax.axhline(1.0, ls="--", color="#7f7f7f", lw=1.3)
    ax.text(len(df) - 0.5, 1.0, "  soglia naive = 1.0", color="#444",
            va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("MASE  (più basso = meglio; 1 = naive)")
    ax.set_title("Accuratezza puntuale — MASE (Electricity, test)")
    ax.set_ylim(0, vals.max() * 1.18)
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGDIR / "fig_cmp_mase.png")
    plt.close(fig)


def fig_calibration(df: pd.DataFrame, is_ph: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    nominal = np.array([0.5, 0.9])
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.0, label="calibrazione perfetta")
    for _, row in df.iterrows():
        emp = np.array([row["cov50"], row["cov90"]], dtype=float)
        ph = bool(row["is_ph"])
        ax.plot(nominal, emp, ls="--" if ph else "-", marker="o",
                mfc="none" if ph else COLOR[row["model"]],
                color=COLOR[row["model"]], lw=1.8, ms=9,
                label=LABEL[row["model"]] + (" (PH)" if ph else ""))
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([0.5, 0.9])
    ax.set_xlabel("copertura nominale (target)")
    ax.set_ylabel("copertura empirica (ottenuta)")
    ax.set_title("Calibrazione — sotto la diagonale = troppo sicuro")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGDIR / "fig_cmp_calibration.png")
    plt.close(fig)


def fig_intervals(df: pd.DataFrame, is_ph: bool) -> None:
    fig, (axc, axw) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = np.arange(len(df))
    w = 0.38
    # coverage panel
    bc1 = axc.bar(x - w / 2, df["cov50"].astype(float), w, color=_colors(df),
                  edgecolor="black", linewidth=0.6, label="50%")
    bc2 = axc.bar(x + w / 2, df["cov90"].astype(float), w, color=_colors(df),
                  edgecolor="black", linewidth=0.6, alpha=0.55, label="90%")
    _mark_placeholder_bars(axc, bc1, df)
    _mark_placeholder_bars(axc, bc2, df)
    axc.axhline(0.5, ls="--", color="#333", lw=1.0)
    axc.axhline(0.9, ls="--", color="#333", lw=1.0)
    axc.text(len(df) - 0.5, 0.5, " nominale 50%", va="bottom", ha="right", fontsize=8, color="#333")
    axc.text(len(df) - 0.5, 0.9, " nominale 90%", va="bottom", ha="right", fontsize=8, color="#333")
    axc.set_xticks(x)
    axc.set_xticklabels(_codes(df))
    axc.set_ylabel("copertura empirica")
    axc.set_title("Copertura vs nominale (tratteggio)")
    axc.set_ylim(0, 1.05)
    axc.legend(title="intervallo", fontsize=9)
    # width panel
    bw1 = axw.bar(x - w / 2, df["width50"].astype(float), w, color=_colors(df),
                  edgecolor="black", linewidth=0.6, label="50%")
    bw2 = axw.bar(x + w / 2, df["width90"].astype(float), w, color=_colors(df),
                  edgecolor="black", linewidth=0.6, alpha=0.55, label="90%")
    _mark_placeholder_bars(axw, bw1, df)
    _mark_placeholder_bars(axw, bw2, df)
    axw.set_xticks(x)
    axw.set_xticklabels(_codes(df))
    axw.set_ylabel("ampiezza media intervallo (scala orig.)")
    axw.set_title("Sharpness — più stretto = più informativo")
    axw.legend(title="intervallo", fontsize=9)
    fig.suptitle("Calibrazione = copertura E ampiezza insieme", fontweight="bold")
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGDIR / "fig_cmp_intervals.png")
    plt.close(fig)


def fig_cost(df: pd.DataFrame, is_ph: bool) -> None:
    fig, (axf, axp) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fit = df["fit_s"].astype(float).fillna(0.0).to_numpy()
    pred = df["predict_s"].astype(float).to_numpy()
    bf = axf.bar(_codes(df), fit, color=_colors(df), edgecolor="black", linewidth=0.6)
    _mark_placeholder_bars(axf, bf, df)
    _bar_labels(axf, bf, fit, fmt="{:.0f}s", dy=max(fit) * 0.01)
    axf.set_ylabel("training wall-clock (s)")
    axf.set_title("Costo di training (fit_s)")
    bp = axp.bar(_codes(df), pred, color=_colors(df), edgecolor="black", linewidth=0.6)
    _mark_placeholder_bars(axp, bp, df)
    axp.set_yscale("log")
    for b, v in zip(bp, pred):
        if np.isfinite(v) and v > 0:
            axp.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}s", ha="center",
                     va="bottom", fontsize=9)
    axp.set_ylabel("inference wall-clock (s, log)")
    axp.set_title("Costo di inferenza (predict_s) — asse log")
    fig.suptitle("Costo — il prezzo del sampling di diffusione", fontweight="bold")
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGDIR / "fig_cmp_cost.png")
    plt.close(fig)


def fig_quality_cost(df: pd.DataFrame, is_ph: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for _, row in df.iterrows():
        ph = bool(row["is_ph"])
        ax.scatter(row["predict_s"], row["CRPS"], s=170,
                   color="none" if ph else COLOR[row["model"]],
                   edgecolor=COLOR[row["model"]] if ph else "black",
                   linewidth=2.0 if ph else 0.8, linestyle="--" if ph else "-",
                   zorder=3)
        ax.annotate(CODE[row["model"]] + (" (PH)" if ph else ""),
                    (row["predict_s"], row["CRPS"]),
                    textcoords="offset points", xytext=(9, 4), fontsize=10,
                    fontweight="bold")
    # barra da battere line (M0 CRPS)
    naive = float(df.loc[df["model"] == "seasonal_naive", "CRPS"].iloc[0])
    ax.axhline(naive, ls="--", color="#7f7f7f", lw=1.2)
    ax.text(ax.get_xlim()[1], naive, f" barra da battere {naive:.1f}",
            color="#444", va="bottom", ha="right", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("costo di inferenza — predict_s (s, log)")
    ax.set_ylabel("CRPS  (più basso = meglio)")
    ax.set_title("No free lunch — qualità vs costo\n(in basso a sinistra è meglio)")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLOR[m],
                          markeredgecolor="black", label=LABEL[m]) for m in ORDER]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.9)
    _ph_banner(fig, is_ph)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGDIR / "fig_cmp_quality_cost.png")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df, is_ph = _load()
    state = "PLACEHOLDER (run M3 non ancora nel registry)" if is_ph else "REALE (riga M3 trovata)"
    print(f"[plot_presentation] dataset={DATASET} | M3 = {state}")
    fig_crps(df, is_ph)
    fig_mase(df, is_ph)
    fig_calibration(df, is_ph)
    fig_intervals(df, is_ph)
    fig_cost(df, is_ph)
    fig_quality_cost(df, is_ph)
    made = sorted(p.name for p in FIGDIR.glob("fig_cmp_*.png"))
    print(f"Wrote {len(made)} figures to {FIGDIR}/:")
    for name in made:
        print(f"  {name}")
    if is_ph:
        print("\nNB: le figure M3 sono PLACEHOLDER. Dopo il run, ri-esegui questo "
              "comando: si aggiornano da sole.")


if __name__ == "__main__":
    main()
