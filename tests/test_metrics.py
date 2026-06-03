"""Offline invariants for the metrics module — focus on the E0 ``CRPS_sum``.

``CRPS_sum`` is the *published* multivariate metric (TimeGrad / CSDI / GluonTS): it
sums the forecast over the ``D`` channels, scores the aggregate with the decile
quantile loss, and normalizes by ``Σ|y_agg|``. Because the E0 reproduce-gate compares
our number against a paper table, the estimator has to be exactly the GluonTS one — so
these tests pin it against a hand-rolled reference and against the streaming accumulator.

    python -m pytest tests/                 # if pytest is installed
    python tests/test_metrics.py            # plain-python fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import (  # noqa: E402
    _CRPS_SUM_QUANTILES,
    ForecastEvaluator,
    crps_ensemble,
    crps_sum,
    evaluate_forecast,
)


def _toy(N: int = 7, S: int = 60, tau: int = 4, D: int = 5):
    """Reproducible truth ``(N, τ, D)`` and ensemble ``(N, S, τ, D)``."""
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 3.0, size=(N, tau, D))
    samples = rng.normal(10.0, 3.0, size=(N, S, tau, D))
    return y, samples


def _reference_crps_sum(y, samples) -> float:
    """Independent GluonTS-style reference: sum over D, deciles, 2·pinball / Σ|y_agg|."""
    y_agg = y.sum(-1)          # (N, τ)
    s_agg = samples.sum(-1)    # (N, S, τ)
    terms = []
    for q in _CRPS_SUM_QUANTILES:
        pq = np.quantile(s_agg, q, axis=1)            # (N, τ)
        u = y_agg - pq
        pinball = np.maximum(q * u, (q - 1.0) * u)    # ρ_q
        terms.append(2.0 * pinball.sum() / np.abs(y_agg).sum())
    return float(np.mean(terms))


def test_crps_sum_matches_reference() -> None:
    y, samples = _toy()
    assert abs(crps_sum(y, samples) - _reference_crps_sum(y, samples)) < 1e-10


def test_crps_sum_streaming_equals_eager() -> None:
    y, samples = _toy()
    eager = crps_sum(y, samples)
    ev = ForecastEvaluator(np.ones(y.shape[-1]), levels=(0.5, 0.9), crps_sum=True)
    for i in range(0, y.shape[0], 3):  # ragged chunks
        sl = slice(i, i + 3)
        ev.update(y[sl], samples[sl].mean(1), samples[sl])
    assert abs(ev.result()["CRPS_sum"] - eager) < 1e-10


def test_crps_sum_perfect_forecast_is_zero() -> None:
    y, samples = _toy()
    perfect = np.repeat(y[:, None], samples.shape[1], axis=1)  # all samples == truth
    assert crps_sum(y, perfect) < 1e-9


def test_crps_sum_is_opt_in() -> None:
    y, samples = _toy()
    point = samples.mean(1)
    scale = np.ones(y.shape[-1])
    off = evaluate_forecast(y, point, samples, scale)
    on = evaluate_forecast(y, point, samples, scale, crps_sum=True)
    assert "CRPS_sum" not in off  # default dict is byte-for-byte the old one (FASE B)
    assert set(off) == {
        "MAE", "RMSE", "MASE", "CRPS", "pinball",
        "cov50", "width50", "cov90", "width90",
    }
    assert on["CRPS_sum"] >= 0.0


def test_crps_sum_is_not_crps() -> None:
    # Different scale (normalized vs per-position): the two must never be conflated.
    y, samples = _toy()
    assert abs(crps_sum(y, samples) - crps_ensemble(y, samples)) > 1e-6


def _run_all() -> int:
    """Plain-python runner so the file works without pytest installed."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
