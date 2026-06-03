"""``electricity_nips`` loader — the **E0 reproduce-gate** (plan §4.3, "reproduce-gate").

Our headline Electricity numbers run on the raw **LSTNet** file (``electricity.txt.gz``,
321 series) with *our own* ratio split — great for fast iteration, but the CRPS-sum
figures in the TimeGrad / CSDI / ScoreGrad papers are on the GluonTS **``electricity_nips``**
benchmark: **370** series, the *official* train/test boundary, and a **7-window rolling**
test with ``prediction_length = 24``. E0 swaps the data source to that benchmark so M3's
``CRPS_sum`` can be read next to the published table — *and nothing downstream changes*,
because this returns the very same :class:`~src.data.contract.ForecastDataset` contract
every model already consumes (``to_gluonts_multivariate("train")`` to fit,
``iter_windows("test", stride=τ)`` to score).

How the official split maps onto the contract
---------------------------------------------
* **train** — the official grouped train series, transposed to time-major ``(L_train, D)``.
  The scaler is fit here only (the usual leakage guard).
* **test** — GluonTS scores 7 rolling windows whose 24-step targets tile the **last 7·τ
  steps** of the full series, each conditioned on the history before it. We reproduce that
  by exposing the **tail of length ``H + 7·τ``** and forcing ``stride = τ``: the contract's
  :meth:`~src.data.contract.ForecastDataset.iter_windows` then yields **exactly 7**
  non-overlapping windows, target-for-target identical to the benchmark, each with ``H``
  steps of immediately-preceding context. (GluonTS feeds each window its *full* history and
  the model truncates to ``context_length`` via its lags; with ``H = 168`` ≥ the hourly lag
  span the two conditionings coincide for TimeGrad — the one small, documented caveat.)
* **val** — a *nominal* tail of train, kept non-empty only so ``summarize`` and any sanity
  code have something to window. The published protocol has no validation split and E0 does
  no early stopping, so this is never used to fit or to score.

gluonts is imported **lazily**: the light/local env (which does not install it, FASE B)
can import this module freely; only :func:`build_electricity_nips` touches gluonts, and
only when the ``electricity_nips`` config is actually built (on Colab/GPU).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import ForecastDataset
from .scaling import Scaler


def _require_gluonts():
    """Import the gluonts symbols E0 needs, with a pointed error in the light env."""
    try:
        from gluonts.dataset.multivariate_grouper import MultivariateGrouper
        from gluonts.dataset.repository.datasets import get_dataset
    except ImportError as exc:  # pragma: no cover - heavy/Colab env only
        raise ImportError(
            "build_electricity_nips() needs the heavy/Colab group (gluonts). It is not "
            "installed in the light local env (see requirements.txt / FASE B). Run E0 on "
            "the GPU box with the pinned heavy deps."
        ) from exc
    return get_dataset, MultivariateGrouper


def build_electricity_nips(config: dict[str, Any]) -> ForecastDataset:
    """Build the contract :class:`ForecastDataset` for the published ``electricity_nips`` split.

    Dispatched from :func:`src.data.loader.build_dataset` when the config carries
    ``source.loader: gluonts_electricity_nips``. Reads ``window.context_length`` (``H``) and
    ``window.horizon`` (``τ``) from the config; the rolling test count comes from the
    benchmark itself (``len(test) / len(train)`` — 7 for ``electricity_nips``) unless pinned
    via ``source.num_test_windows``.
    """
    get_dataset, MultivariateGrouper = _require_gluonts()

    src = config["source"]
    win = config["window"]
    H = int(win["context_length"])
    tau = int(win["horizon"])
    gluonts_name = src.get("gluonts_name", "electricity_nips")
    regenerate = bool(src.get("regenerate", False))

    raw = get_dataset(gluonts_name, regenerate=regenerate)

    # D and the rolling-window count come from the benchmark metadata, not a magic number.
    try:
        target_dim = int(raw.metadata.feat_static_cat[0].cardinality)
    except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
        target_dim = sum(1 for _ in raw.train)
    n_train_series = sum(1 for _ in raw.train)
    n_test_series = sum(1 for _ in raw.test)
    num_test_windows = int(
        src.get("num_test_windows", max(1, n_test_series // max(1, n_train_series)))
    )
    freq = raw.metadata.freq

    # Group the per-series univariate entries into one D-dimensional multivariate series.
    train_grouper = MultivariateGrouper(max_target_dim=target_dim)
    test_grouper = MultivariateGrouper(
        num_test_dates=num_test_windows, max_target_dim=target_dim
    )
    train_mv = train_grouper(raw.train)
    test_mv = test_grouper(raw.test)

    train_entry = next(iter(train_mv))
    start = str(train_entry["start"])  # JSON-safe and a valid GluonTS start string
    train_arr = np.asarray(train_entry["target"], dtype=np.float64).T  # (L_train, D)

    # The longest grouped-test entry is the full series (ends at the final timestamp); its
    # last num_test_windows·τ steps are exactly the rolling forecast targets.
    test_entries = list(test_mv)
    full_entry = max(test_entries, key=lambda e: np.asarray(e["target"]).shape[-1])
    full_arr = np.asarray(full_entry["target"], dtype=np.float64).T  # (L_full, D)

    tail_len = H + num_test_windows * tau
    if full_arr.shape[0] < tail_len:
        raise ValueError(
            f"electricity_nips full series has {full_arr.shape[0]} steps but the rolling "
            f"test needs H + {num_test_windows}·τ = {tail_len}. Lower context_length/horizon."
        )
    test_arr = full_arr[-tail_len:]  # stride=τ over this yields exactly num_test_windows wins

    # Nominal, unused validation tail (see module docstring); never fit/scored on.
    val_len = H + tau
    val_arr = train_arr[-val_len:].copy() if train_arr.shape[0] >= val_len else train_arr.copy()

    raw_splits = {"train": train_arr, "val": val_arr, "test": test_arr}
    scaler = Scaler(method=config["scaling"]["method"]).fit(train_arr)
    splits = {k: scaler.transform(v) for k, v in raw_splits.items()}

    meta = {
        "D": target_dim,
        "freq": freq,
        "start_date": start,
        "context_length": H,
        "horizon": tau,
        # FORCED to τ so iter_windows reproduces the non-overlapping rolling protocol
        # exactly, regardless of any window.stride in the config (which is ignored here).
        "stride": tau,
        "scaling": config["scaling"]["method"],
        "split_ratios": None,  # official split — not ratio-based
        "seed": config.get("seed"),
        "num_test_windows": num_test_windows,
        "protocol": f"{gluonts_name} (official split, {num_test_windows}-window rolling)",
    }
    return ForecastDataset(
        name=config["name"],
        splits=splits,
        raw_splits=raw_splits,
        scaler=scaler,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Colab smoke: `python -m src.data.electricity_nips` prints the contract shapes
# so the (untestable-locally) gluonts path can be eyeballed before a heavy run.
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual Colab check
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.loader import summarize
    from src.utils.config import load_config

    cfg_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).resolve().parents[2] / "configs" / "data_electricity_nips.yaml")
    )
    ds = build_electricity_nips(load_config(cfg_path))
    summarize(ds)
    print(
        f"protocol     : {ds.meta['protocol']}\n"
        f"test windows : {ds.num_windows('test')} (expect {ds.meta['num_test_windows']})  "
        f"stride={ds.meta['stride']}  start={ds.meta['start_date']}"
    )
