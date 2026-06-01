"""Generic builder for LSTNet-style benchmark datasets.

Exchange, Electricity, Solar and Traffic all ship from the same benchmark repo in
the same shape: a **gzipped, header-less CSV** of float columns, one row per time
step, no timestamps. They therefore share one pipeline:

    download -> parse -> nominal calendar -> temporal split -> train-only scaling

That pipeline lives here once, in :func:`build_dataset`. Per-dataset modules
(`exchange.py`, `electricity.py`, …) are thin wrappers that point at their config
and can add dataset-specific cleaning later. Keeping the build in one place means
the fairness guarantees of the data contract (plan §4.5) hold identically for every
dataset — nobody can split or scale one dataset differently from another.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contract import ForecastDataset, temporal_split
from .scaling import Scaler


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def download(url: str, raw_dir: str | Path, filename: str) -> Path:
    """Idempotently fetch a raw gzipped file into ``raw_dir``.

    Skips the download if the file already exists. ``data/raw/`` is gitignored, so
    the file lives only on the local machine. Network access may be sandboxed; in
    that case download once outside the sandbox and the cached file is reused.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    if dest.exists():
        return dest

    import requests  # local import: only needed on the cache-miss path

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_gz_csv(path: str | Path) -> pd.DataFrame:
    """Parse a gzipped, header-less CSV into a tidy ``(L, D)`` DataFrame.

    The file has no timestamps; columns are bare floats. We return a plain
    integer-indexed frame here — the nominal calendar index is attached later from
    the config, since it is for covariates only.
    """
    path = Path(path)
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, header=None)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df


def attach_calendar(df: pd.DataFrame, start_date: str, freq: str) -> pd.DataFrame:
    """Give the values a *nominal* DatetimeIndex (for calendar features only).

    The heavy (M2/M3) Colab env pins ``pandas < 2.2`` for GluonTS 0.13, and that older
    pandas rejects the modern lowercase sub-daily aliases the configs use (Electricity's
    ``freq="h"`` -> ``ValueError``). When that happens we retry with the legacy spelling
    (:func:`src.utils.freq.gluonts_freq`, ``"h"`` -> ``"H"``); the light/local env keeps
    pandas >= 2.2 and takes the first branch unchanged.
    """
    try:
        idx = pd.date_range(start=start_date, periods=len(df), freq=freq)
    except ValueError:
        from src.utils.freq import gluonts_freq

        idx = pd.date_range(start=start_date, periods=len(df), freq=gluonts_freq(freq))
    out = df.copy()
    out.index = idx
    return out


# ---------------------------------------------------------------------------
# The contract builder
# ---------------------------------------------------------------------------
def build_dataset(config: dict[str, Any]) -> ForecastDataset:
    """Build a :class:`ForecastDataset` from a parsed data config.

    Pipeline: download -> parse -> nominal calendar -> temporal split (no leakage)
    -> fit scaler on TRAIN ONLY -> transform every split. Returns the single object
    every model consumes (plan §4.5). Works for any LSTNet-style gzipped CSV; the
    dataset identity comes entirely from ``config``.
    """
    src = config["source"]
    raw_path = download(src["url"], src["raw_dir"], src["filename"])
    df = load_gz_csv(raw_path)
    df = attach_calendar(df, src["start_date"], src["freq"])

    values = df.to_numpy(dtype=np.float64)  # (L, D), time-major
    raw_splits = temporal_split(values, tuple(config["split"]["ratios"]))

    # Fit on train only — the leakage guard (plan §4.4).
    scaler = Scaler(method=config["scaling"]["method"]).fit(raw_splits["train"])
    splits = {k: scaler.transform(v) for k, v in raw_splits.items()}

    meta = {
        "D": values.shape[1],
        "freq": src["freq"],
        "start_date": src["start_date"],
        "context_length": config["window"]["context_length"],
        "horizon": config["window"]["horizon"],
        "stride": config["window"].get("stride", 1),
        "scaling": config["scaling"]["method"],
        "split_ratios": list(config["split"]["ratios"]),
        "seed": config.get("seed"),
    }
    return ForecastDataset(
        name=config["name"],
        splits=splits,
        raw_splits=raw_splits,
        scaler=scaler,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Shared smoke-test reporter (used by each dataset module's __main__)
# ---------------------------------------------------------------------------
def summarize(ds: ForecastDataset) -> None:
    """Print split/window shapes + a leakage sanity check for a built dataset.

    Window *counts* are computed analytically (:meth:`ForecastDataset.num_windows`)
    so this never allocates the full window tensor — safe even for wide datasets
    like Electricity (D=321), where materializing all windows would be ~8 GB.
    """
    print(f"dataset      : {ds.name}  (D={ds.D}, H={ds.H}, tau={ds.tau})")
    for split in ("train", "val", "test"):
        raw = ds.raw_splits[split]
        n = ds.num_windows(split)
        print(
            f"  {split:5s}: series {raw.shape}  ->  "
            f"{n:>6d} windows of ctx ({ds.H}, {ds.D}) + tgt ({ds.tau}, {ds.D})"
        )
    tr = ds.splits["train"]
    print(
        f"scaled train : mean|max|={np.abs(tr.mean(0)).max():.2e}  "
        f"std~{tr.std(0).mean():.3f}"
    )
