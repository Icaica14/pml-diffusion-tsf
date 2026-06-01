"""Assemble the M0–M3 head-to-head comparison tables from the results registry (E1).

Counterpart to ``plot_results.py`` / ``plot_presentation.py`` (which draw the *figures*):
this script turns ``results/registry.csv`` into the **comparison tables** the deck and
report need — one Markdown file and one tidy CSV **per dataset** — so the head-to-head
numbers are *generated*, never hand-typed (the repo's single-source-of-truth rule). It
replaces the by-hand tables currently living in ``STATO_PROGETTO.md`` §2.

For every dataset present in the registry it writes, under ``results/tables/``:

  comparison_<dataset>.md    two readable tables — **Qualità** (MASE/CRPS/pinball/
                             coverage/MAE/RMSE) and **Costo & setup** (fit/predict/
                             epochs/diff_steps/platform) — with the per-column winner
                             in **bold** and the M0 CRPS flagged as the "barra da
                             battere" (the bar M3 must beat).
  comparison_<dataset>.csv   the same rows, full precision, all metric columns, tidy
                             (one row per model, canonical M0→M3 order).

Winner logic is metric-aware: lower-is-better for the error/cost columns, while the
coverage winner is the model **closest to nominal** (0.50 / 0.90), not the largest.

Placeholder handling (mirrors ``plot_presentation.py``): if the Electricity
``timegrad`` row is still missing (M3 in run), an M3 PLACEHOLDER row is shown in
(parentheses) with a dagger and is **excluded** from the winner computation. It is
replaced by the real row — and the parentheses vanish — the moment that row lands in
the registry. Just re-run the command.

Run::

    python -m experiments.make_tables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "results" / "registry.csv"
OUTDIR = REPO_ROOT / "results" / "tables"

# Fixed model order + stable code/name per model — identical to the plot scripts so a
# reader learns the ladder once (M0→M3) and reads every table and figure the same way.
ORDER = ["seasonal_naive", "arima", "deepar", "deepar_notf", "timegrad"]
CODE = {
    "seasonal_naive": "M0",
    "arima": "M1",
    "deepar": "M2",
    "deepar_notf": "M2-nf",
    "timegrad": "M3",
}
NAME = {
    "seasonal_naive": "seasonal-naive",
    "arima": "ARIMA",
    "deepar": "DeepAR",
    "deepar_notf": "DeepAR (no-tf)",
    "timegrad": "TimeGrad",
}

# --- M3 PLACEHOLDER (interval midpoints from RESULTS_PLACEHOLDERS_ELECTRICITY.md) -----
# Used ONLY for Electricity while the real timegrad row is in run. These are NOT
# measurements: they render in (parentheses) with a dagger and never count as a winner.
# epochs/diff_steps are the *planned* config, shown so the cost table reads sensibly.
PLACEHOLDER_M3 = {
    "MASE": 1.53,
    "CRPS": 210.0,
    "pinball": 110.0,
    "cov50": 0.49,
    "cov90": 0.87,
    "MAE": 300.0,
    "RMSE": 2300.0,
    "width50": 430.0,
    "width90": 1150.0,
    "fit_s": 1800.0,
    "predict_s": 10800.0,
    "epochs": 50,
    "diff_steps": 100,
    "platform": "Colab Pro (L4)",
}

# Metric direction. Lower is better for all error/cost columns; coverage is judged by
# closeness to its nominal target, not by magnitude.
LOWER_BETTER = {"MAE", "RMSE", "MASE", "CRPS", "pinball",
                "width50", "width90", "fit_s", "predict_s"}
COV_TARGET = {"cov50": 0.5, "cov90": 0.9}

# Columns rendered as integers (config knobs, not measurements).
INT_COLS = {"epochs", "diff_steps", "num_cells", "layers", "input_size",
            "hidden", "n_windows", "H", "tau", "D", "n_samples", "seed"}

# The two human-facing tables (column order matches STATO_PROGETTO.md §2 exactly).
QUALITY_COLS = ["MASE", "CRPS", "pinball", "cov50", "cov90", "MAE", "RMSE"]
COST_COLS = ["fit_s", "predict_s", "epochs", "diff_steps", "platform"]

# All numeric metric columns (everything except descriptive strings) — used to decide
# which placeholder cells get the (parenthesis)† treatment.
NUMERIC_METRICS = set(QUALITY_COLS) | {"width50", "width90", "fit_s", "predict_s"}

PRETTY = {
    "MASE": "MASE", "CRPS": "CRPS", "pinball": "pinball",
    "cov50": "cov50", "cov90": "cov90", "MAE": "MAE", "RMSE": "RMSE",
    "width50": "width50", "width90": "width90",
    "fit_s": "fit", "predict_s": "predict",
    "epochs": "epochs", "diff_steps": "diff_steps", "platform": "piattaforma",
}

# Tidy-CSV column order (identity + every metric + the run knobs that define the row).
CSV_COLS = (
    ["dataset", "code", "model", "is_placeholder"]
    + ["MASE", "CRPS", "pinball", "cov50", "cov90", "width50", "width90", "MAE", "RMSE"]
    + ["fit_s", "predict_s", "epochs", "diff_steps", "num_cells", "layers",
       "input_size", "hidden", "n_windows", "H", "tau", "D", "n_samples", "seed",
       "season_length", "platform"]
)


def _short_platform(value: object) -> str:
    """Collapse the long ``platform.platform()`` string to where the run lived."""
    s = str(value)
    if s in ("nan", "None", ""):
        return "—"
    if "Colab" in s:
        return s  # already short (placeholder)
    if "Linux" in s:
        return "Linux (Colab)"
    if "macOS" in s or "Darwin" in s:
        return "macOS (locale)"
    return s


def _fmt_value(v: float, col_max: float) -> str:
    """Format a float with per-column decimals chosen from the column's magnitude.

    One rule covers both datasets: Electricity's CRPS≈160 prints as ``160.5`` while
    Exchange's CRPS≈0.007 prints as ``0.007175`` (4 significant figures), so the same
    table generator stays readable whether the numbers are huge or tiny.
    """
    if col_max >= 100:
        return f"{v:,.1f}"
    if col_max >= 10:
        return f"{v:.2f}"
    if col_max >= 0.1:
        return f"{v:.3f}"
    return f"{v:.4g}"


def _format_cell(col: str, v: object, col_max: float) -> str:
    if col == "platform":
        return _short_platform(v)
    try:
        fv = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(fv):
        return "—"
    if col in INT_COLS:
        return f"{int(round(fv))}"
    return _fmt_value(fv, col_max)


def _winner_models(df: pd.DataFrame, col: str) -> set[str]:
    """Models tied for best in ``col`` among the real (non-placeholder) rows."""
    if col not in LOWER_BETTER and col not in COV_TARGET:
        return set()
    sub = df[~df["is_ph"]]
    vals = pd.to_numeric(sub[col], errors="coerce")
    mask = np.isfinite(vals)
    if not mask.any():
        return set()
    if col in COV_TARGET:
        dist = (vals - COV_TARGET[col]).abs()
        best = dist[mask].min()
        sel = mask & np.isclose(dist, best)
    else:
        best = vals[mask].min()
        sel = mask & np.isclose(vals, best)
    return set(sub.loc[sel, "model"])


def _load_dataset(df_all: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, bool]:
    """Return (rows for ``dataset`` in canonical order, m3_is_placeholder).

    For Electricity only, a PLACEHOLDER M3 row is injected when the real timegrad row
    is still missing — so the table draws the full ladder before the GPU run lands.
    """
    df = df_all[(df_all["dataset"] == dataset) & (df_all["model"].isin(ORDER))].copy()
    inject_ph = (dataset == "electricity") and not (df["model"] == "timegrad").any()
    if inject_ph:
        row = {c: np.nan for c in df.columns}
        row["dataset"] = dataset
        row["model"] = "timegrad"
        row.update(PLACEHOLDER_M3)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df["is_ph"] = inject_ph & (df["model"] == "timegrad")
    df["code"] = df["model"].map(CODE)
    df["name"] = df["model"].map(NAME)
    df["model"] = pd.Categorical(df["model"], categories=ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    df["model"] = df["model"].astype(str)
    return df, inject_ph


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render one Markdown table: a 'Modello' column then ``cols`` (winners bold)."""
    colmax: dict[str, float] = {}
    winners: dict[str, set[str]] = {}
    for c in cols:
        vals = pd.to_numeric(df.loc[~df["is_ph"], c], errors="coerce")
        finite = vals[np.isfinite(vals)]
        colmax[c] = float(finite.abs().max()) if len(finite) else 0.0
        winners[c] = _winner_models(df, c)

    header = "| Modello | " + " | ".join(PRETTY[c] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for _, row in df.iterrows():
        label = f"{row['code']} {row['name']}" + ("  ⏳" if row["is_ph"] else "")
        cells = []
        for c in cols:
            cell = _format_cell(c, row[c], colmax[c])
            if row["is_ph"]:
                if c != "platform" and cell != "—":
                    cell = f"({cell})†"
            elif row["model"] in winners[c]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _meta_int(v: object) -> str:
    """Render a run-meta value as a clean integer (the placeholder injection promotes
    int columns to float, so 5071 would otherwise print as 5071.0)."""
    try:
        fv = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"
    return f"{int(round(fv))}" if np.isfinite(fv) else "?"


def _markdown(df: pd.DataFrame, dataset: str, is_ph: bool) -> str:
    """Full Markdown document for one dataset: header + Quality + Cost + notes."""
    n = df.attrs.get("meta", {})
    meta_line = ""
    if n:
        meta_line = (f"_test split · n_windows={_meta_int(n.get('n_windows'))} · "
                     f"H={_meta_int(n.get('H'))} · τ={_meta_int(n.get('tau'))} · "
                     f"D={_meta_int(n.get('D'))} · S={_meta_int(n.get('n_samples'))} · "
                     f"seed={_meta_int(n.get('seed'))} · "
                     f"m={_meta_int(n.get('season_length'))}_")

    # "barra da battere" = M0 CRPS (the persistence bar every deep model must beat).
    naive = df.loc[df["model"] == "seasonal_naive", "CRPS"]
    bar = float(naive.iloc[0]) if len(naive) else float("nan")
    bar_txt = _fmt_value(bar, abs(bar)) if np.isfinite(bar) else "—"

    parts = [
        f"# Comparativa M0–M3 — {dataset.capitalize()}",
        "",
        "> Generato da `experiments/make_tables.py` da `results/registry.csv`. "
        "**Non modificare a mano** — rigenera il file.",
        "",
    ]
    if meta_line:
        parts += [meta_line, ""]
    parts += [
        "## Qualità",
        "",
        _md_table(df, QUALITY_COLS),
        "",
        f"**Barra da battere (CRPS):** M0 = {bar_txt} — il riferimento di persistenza "
        "che i modelli deep (e in particolare M3) devono superare. "
        "_Grassetto = miglior modello per colonna; coverage premia la vicinanza al "
        "nominale (0.50 / 0.90), non il valore più alto._",
        "",
        "## Costo & setup",
        "",
        _md_table(df, COST_COLS),
        "",
    ]
    if is_ph:
        parts += [
            "> † **M3 = PLACEHOLDER** "
            "[aggiornare appena disponibile results/registry.csv]. I valori M3 in "
            "(parentesi) sono segnaposto (midpoint dagli intervalli attesi), esclusi "
            "dal calcolo del vincitore. Spariscono ri-eseguendo lo script quando la "
            "riga reale `electricity,timegrad` entra nella registry.",
            "",
        ]
    parts += [
        "_MASE/CRPS dipendono da scala e denominatore del dataset: **confrontabili "
        "solo entro lo stesso dataset**, mai tra Exchange ed Electricity._",
        "",
    ]
    return "\n".join(parts)


def _tidy_csv(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = dataset
    out["is_placeholder"] = out["is_ph"]
    cols = [c for c in CSV_COLS if c in out.columns]
    return out[cols]


def _console_summary(df: pd.DataFrame, dataset: str, is_ph: bool) -> None:
    real = df[~df["is_ph"]]
    best_crps = real.loc[pd.to_numeric(real["CRPS"], errors="coerce").idxmin()]
    best_mase = real.loc[pd.to_numeric(real["MASE"], errors="coerce").idxmin()]
    tag = "  (M3 = PLACEHOLDER)" if is_ph else ""
    print(f"  {dataset:<12} CRPS best = {best_crps['code']} {best_crps['name']} "
          f"({_fmt_value(float(best_crps['CRPS']), abs(float(best_crps['CRPS'])))})"
          f"   MASE best = {best_mase['code']} {best_mase['name']}"
          f"({float(best_mase['MASE']):.3f}){tag}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(REGISTRY)
    datasets = [d for d in ["exchange", "electricity"]
                if d in set(df_all["dataset"].unique())]
    # any other datasets that show up later, appended after the two known ones
    datasets += [d for d in sorted(df_all["dataset"].unique()) if d not in datasets]

    print(f"[make_tables] registry={REGISTRY.name}  datasets={datasets}")
    written: list[str] = []
    for ds in datasets:
        df, is_ph = _load_dataset(df_all, ds)
        if df.empty:
            continue
        meta_row = df[~df["is_ph"]].iloc[0]
        df.attrs["meta"] = {k: meta_row.get(k) for k in
                            ["n_windows", "H", "tau", "D", "n_samples", "seed",
                             "season_length"]}
        md_path = OUTDIR / f"comparison_{ds}.md"
        csv_path = OUTDIR / f"comparison_{ds}.csv"
        md_path.write_text(_markdown(df, ds, is_ph), encoding="utf-8")
        _tidy_csv(df, ds).to_csv(csv_path, index=False)
        written += [md_path.name, csv_path.name]
        _console_summary(df, ds, is_ph)

    print(f"\nWrote {len(written)} files to {OUTDIR}/:")
    for name in sorted(written):
        print(f"  {name}")
    if any((df_all["dataset"] == "electricity").values) and not (
        ((df_all["dataset"] == "electricity") & (df_all["model"] == "timegrad")).any()
    ):
        print("\nNB: la riga M3 (electricity,timegrad) non è ancora nella registry: "
              "la tabella Electricity usa il PLACEHOLDER. Dopo il run, ri-esegui "
              "questo comando — si aggiorna da sola.")


if __name__ == "__main__":
    main()
