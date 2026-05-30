"""Exchange loader — the iteration dataset (plan §4.2–4.3).

8 currencies vs USD, daily cadence, ~7588 steps, from the LSTNet multivariate
benchmark (the same lineage TimeGrad uses). Tiny and fast: we bring the whole
pipeline green here before scaling to the primary dataset.

The actual build is fully generic and lives in :mod:`src.data.loader`; this module
is a thin, named entry point (`load_exchange`) plus a smoke test. Dataset-specific
cleaning, if ever needed, would go here.

Run as a script for a smoke test of shapes::

    python -m src.data.exchange
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import ForecastDataset
from .loader import build_dataset, summarize


def load_exchange(config: dict[str, Any]) -> ForecastDataset:
    """Build the :class:`ForecastDataset` for Exchange from a parsed config dict."""
    return build_dataset(config)


def _smoke_test() -> None:
    from ..utils.config import load_config
    from ..utils.seeds import set_seed

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_config(repo_root / "configs" / "data_exchange.yaml")
    set_seed(cfg.get("seed", 42))
    summarize(load_exchange(cfg))


if __name__ == "__main__":
    _smoke_test()
