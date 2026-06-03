"""Electricity loader — the primary dataset for the group project (plan §4.2–4.3).

321 client load series, hourly, ~26 304 steps (2012–2014), from the LSTNet
multivariate benchmark. Electricity is the *headline* dataset: industrial load with
strong daily/weekly seasonality, in the TimeGrad benchmark family, so it carries the
scientific claims of the project once the pipeline is green on Exchange.

The build is fully generic and lives in :mod:`src.data.loader`; this module is a thin,
named entry point (`load_electricity`) plus a smoke test.

Note for the E0 reproduce-gate
------------------------------
This loader uses the raw LSTNet `electricity.txt.gz` (light, downloadable like
Exchange) for fast local iteration and EDA. The published TimeGrad/CSDI **CRPS-sum**
numbers we must reproduce in E0 are reported on the GluonTS `electricity_nips`
dataset, which uses the benchmark's *official* split and a rolling-window test
protocol. E0 swaps to `electricity_nips` via GluonTS; everything else in the pipeline
is unchanged because every model consumes the same `ForecastDataset` contract.

That swap is now implemented — see :mod:`src.data.electricity_nips` (the loader),
``configs/data_electricity_nips.yaml`` (the config, with H/τ identical to this one so
the *split* is the only variable) and the ``--crps-sum`` flag on the runners. Build it
with ``source.loader: gluonts_electricity_nips``; the published metric
(:func:`src.eval.metrics.crps_sum`) is emitted only on that opt-in path.

Run as a script for a smoke test of shapes::

    python -m src.data.electricity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import ForecastDataset
from .loader import build_dataset, summarize


def load_electricity(config: dict[str, Any]) -> ForecastDataset:
    """Build the :class:`ForecastDataset` for Electricity from a parsed config dict."""
    return build_dataset(config)


def _smoke_test() -> None:
    from ..utils.config import load_config
    from ..utils.seeds import set_seed

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "configs" / "data_electricity.yaml")
    set_seed(cfg.get("seed", 42))
    summarize(load_electricity(cfg))


if __name__ == "__main__":
    _smoke_test()
