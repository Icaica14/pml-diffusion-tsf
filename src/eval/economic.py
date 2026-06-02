"""src/eval/economic.py — E6: from forecast to decision, the economic-value pillar.

A forecast only matters if it changes a *decision*. This module turns each model's
forecast of future load into a **battery-dispatch schedule** and prices it, so we can
report — in euros — how much a better-calibrated distribution is actually worth
(implementation plan §3.5 and §6.4; experiment E6). It reuses the forecasts E1 already
produces: **no extra training**, pure NumPy + a tiny linear program solved with CBC.

The decision problem (a small linear program)
---------------------------------------------
A battery with energy capacity ``e_max``, per-step power limit ``p_max`` and round-trip
efficiency ``eta`` serves a load series under a time-of-use price ``pi_t``. At each step
we pick a charge ``c_t >= 0`` and a discharge ``d_t >= 0``; the grid import is
``g_t = load_t + c_t - d_t`` and the bill is ``sum_t pi_t * max(g_t, 0)``.

Why ``max(g_t, 0)`` and not just ``g_t``? Because we forbid **export** (no selling
back to the grid). That single non-linearity is the whole point: over-discharging
wastes stored energy you already paid for, under-discharging leaves you buying at the
peak — an asymmetric, *newsvendor* cost. Under a linear (net-metering) bill the
expected cost would depend only on the forecast **mean**, and a full distribution would
be worthless; the no-export clamp is exactly what lets a predictive distribution beat a
point forecast.

Round-trip efficiency is charged on the charging leg: storing ``c_t`` raises the
state-of-charge by ``eta * c_t``; discharging ``d_t`` returns ``d_t`` to offset load.
Arbitrage therefore pays whenever ``pi_charge < eta * pi_discharge``. We default the
initial state-of-charge to 0 and require the final state-of-charge ``>= soc0``, so only
genuine *within-horizon* arbitrage counts (no free windfall from dumping a pre-charged
battery).

Two plans per model (plan §6.4)
-------------------------------
* **Deterministic** — feed the LP the point forecast (one scenario, ``S = 1``).
* **Stochastic (SAA)** — feed the LP all ``S`` sample paths and minimise the
  *sample-average* bill; the first-stage controls ``(c, d)`` are shared across
  scenarios while a per-scenario epigraph variable books each path's import. Only
  probabilistic models (DeepAR, TimeGrad, the naive bootstrap) can produce this; that
  asymmetry is precisely what E6 measures.

Value (plan §6.4)
-----------------
Every committed schedule is settled against the **true** future:
* **money saved vs naive** = bill(no-battery) - bill(model schedule);
* **value of the distribution** = bill(deterministic) - bill(stochastic) for the
  *same* model — how much the uncertainty itself is worth;
* both framed between an **oracle** (perfect-foresight LP -> lower bound on the bill)
  and the **naive** no-battery plan (ceiling), so every euro is reported as a *fraction
  of the attainable saving*, never an absolute that hides the dataset's scale.

Verify::

    python -m pytest tests/test_economic.py        # hand-checkable LP correctness
    python tests/test_economic.py                  # same, no pytest needed
    python -m experiments.run_economic             # M0 money table on Electricity
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pulp

__all__ = [
    "BatteryParams",
    "Tariff",
    "tou_price",
    "size_battery",
    "dispatch_lp",
    "realized_bill",
    "no_battery_bill",
    "evaluate_episode",
    "evaluate_value",
]


# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class BatteryParams:
    """Physical spec of the storage asset (plan §3.5).

    Units are arbitrary but must be consistent with the load series (e.g. kWh for
    energy and kW-per-step for power on an hourly grid). Only price *ratios* and the
    battery/load *ratio* drive the optimum, so the absolute scale is free.
    """

    e_max: float  # energy capacity (maximum state-of-charge)
    p_max: float  # per-step charge / discharge power limit
    eta: float = 0.9  # round-trip efficiency in (0, 1], applied on the charge leg
    soc0: float = 0.0  # initial state-of-charge
    terminal_ge_initial: bool = True  # require final SoC >= soc0 (no free initial energy)

    def __post_init__(self) -> None:
        if not (self.e_max > 0 and self.p_max > 0):
            raise ValueError("e_max and p_max must be positive")
        if not (0.0 < self.eta <= 1.0):
            raise ValueError("eta must be in (0, 1]")
        if not (0.0 <= self.soc0 <= self.e_max):
            raise ValueError("soc0 must be in [0, e_max]")


@dataclass(frozen=True)
class Tariff:
    """A simple time-of-use price (plan §6.4).

    Peak hours cost ``peak``, the night off-peak window costs ``offpeak``, everything
    else ``shoulder``. The wider the peak/off-peak ratio, the more a battery — and a
    *distribution* — can earn; E6's cost-asymmetry sweep just rescales these.
    """

    peak: float = 0.30
    shoulder: float = 0.15
    offpeak: float = 0.08
    peak_hours: tuple[int, ...] = (17, 18, 19, 20)  # evening demand peak
    offpeak_hours: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)  # night

    def price_for_hour(self, hour: int) -> float:
        h = int(hour) % 24
        if h in self.peak_hours:
            return self.peak
        if h in self.offpeak_hours:
            return self.offpeak
        return self.shoulder


def tou_price(hours, tariff: Tariff) -> np.ndarray:
    """Map an array of hour-of-day (length ``tau``) to a per-step price vector."""
    hours = np.asarray(hours).ravel()
    return np.array([tariff.price_for_hour(h) for h in hours], dtype=float)


def size_battery(
    load_ref,
    *,
    power_frac: float = 0.5,
    hours: float = 4.0,
    eta: float = 0.9,
    soc0: float = 0.0,
    terminal_ge_initial: bool = True,
) -> BatteryParams:
    """Scale a battery to a reference load.

    ``p_max = power_frac * mean(|load_ref|)`` and ``e_max = p_max * hours`` (a
    ``hours``-hour battery). Sizing relative to the data keeps the demo's €-figures on
    the dataset's own scale instead of hard-coding absolute kW/kWh that only make sense
    for one series.
    """
    load_ref = np.asarray(load_ref, dtype=float)
    mean_load = float(np.nanmean(np.abs(load_ref)))
    if not np.isfinite(mean_load) or mean_load <= 0:
        raise ValueError("load_ref must have a positive, finite mean magnitude")
    p_max = power_frac * mean_load
    return BatteryParams(
        e_max=p_max * hours,
        p_max=p_max,
        eta=eta,
        soc0=soc0,
        terminal_ge_initial=terminal_ge_initial,
    )


# ------------------------------------------------------------------------- the LP


def _check_finite(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(
            f"{name} contains non-finite values (NaN/inf) — refusing to fabricate a "
            "schedule; check the forecast/registry upstream"
        )
    return a


def _solver() -> pulp.LpSolver:
    """A quiet CBC instance (ships with PuLP). Fresh per solve to avoid shared state."""
    return pulp.PULP_CBC_CMD(msg=False)


@contextmanager
def _quiet_pulp():
    """Silence PuLP's PuLP-4.0 DeprecationWarnings locally.

    We deliberately use the ``LpVariable(...)`` / ``PULP_CBC_CMD`` API because the repo
    targets ``PuLP>=2.7`` (requirements.txt) where the 4.0 replacements do not exist.
    The warnings are forward-looking noise — one per variable per solve, i.e. thousands
    across the full E6 run — so we mute them only around our own LP, restoring the
    caller's warning filters on exit (no global state touched).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _v(x) -> float:
    val = pulp.value(x)
    return float(val) if val is not None else 0.0


def dispatch_lp(scenarios, price, battery: BatteryParams, *, demand_charge: float = 0.0) -> dict:
    """Solve the battery-dispatch LP for one episode.

    Parameters
    ----------
    scenarios : array (S, tau) or (tau,)
        Future load paths. ``S == 1`` (or a 1-D array) -> the **deterministic** plan on
        the point forecast; ``S > 1`` -> the **stochastic SAA** plan minimising the
        sample-average bill with shared first-stage controls and a per-scenario import
        epigraph.
    price : array (tau,)
        Per-step time-of-use price ``pi_t``.
    battery : BatteryParams
    demand_charge : float, default 0.0
        Optional €/unit charge on the worst per-step import (across scenarios). 0 keeps
        the headline pure energy arbitrage; >0 adds peak-shaving. Linear via one
        epigraph variable, so the problem stays an LP.

    Returns
    -------
    dict
        ``charge``, ``discharge``, ``soc`` (each a length-``tau`` numpy array) and the
        scalar ``planned_bill`` (the optimised sample-average objective).
    """
    scen = np.atleast_2d(_check_finite("scenarios", scenarios))
    price = _check_finite("price", price)
    n, tau = scen.shape
    if price.shape[0] != tau:
        raise ValueError(f"price length {price.shape[0]} != horizon {tau}")

    with _quiet_pulp():
        prob = pulp.LpProblem("battery_dispatch", pulp.LpMinimize)
        c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=battery.p_max) for t in range(tau)]
        d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=battery.p_max) for t in range(tau)]
        soc = [pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=battery.e_max) for t in range(tau)]
        # Epigraph for max(load + c - d, 0), one per scenario per step (g >= 0 by bound).
        g = [[pulp.LpVariable(f"g_{s}_{t}", lowBound=0) for t in range(tau)] for s in range(n)]

        # State-of-charge dynamics: soc[t] = soc[t-1] + eta * c[t] - d[t].
        prev = battery.soc0
        for t in range(tau):
            prob += soc[t] == prev + battery.eta * c[t] - d[t]
            prev = soc[t]
        if battery.terminal_ge_initial:
            prob += soc[tau - 1] >= battery.soc0

        # No-export import epigraph: g[s][t] >= load[s,t] + c[t] - d[t].
        for s in range(n):
            for t in range(tau):
                prob += g[s][t] >= float(scen[s, t]) + c[t] - d[t]

        energy = pulp.lpSum(price[t] * g[s][t] for s in range(n) for t in range(tau)) / n
        if demand_charge > 0:
            peak = pulp.LpVariable("peak", lowBound=0)
            for s in range(n):
                for t in range(tau):
                    prob += peak >= g[s][t]
            prob += energy + demand_charge * peak
        else:
            prob += energy

        prob.solve(_solver())

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"battery-dispatch LP did not solve to optimality (status={status})")

    return {
        "charge": np.array([_v(c[t]) for t in range(tau)], dtype=float),
        "discharge": np.array([_v(d[t]) for t in range(tau)], dtype=float),
        "soc": np.array([_v(soc[t]) for t in range(tau)], dtype=float),
        "planned_bill": _v(prob.objective),
    }


# ------------------------------------------------------------- settle on the truth


def realized_bill(charge, discharge, true_load, price, *, demand_charge: float = 0.0) -> float:
    """Settle a committed schedule against the realised (true) future load.

    Grid import ``g_t = true_load_t + charge_t - discharge_t``; we pay for imports only,
    with **no export credit**: ``bill = sum_t price_t * max(g_t, 0)`` (plus
    ``demand_charge * max_t g_t`` if a peak charge is set). Over-discharge (``g_t < 0``)
    is wasted energy — the asymmetry the stochastic plan hedges against.
    """
    charge = _check_finite("charge", charge)
    discharge = _check_finite("discharge", discharge)
    true_load = _check_finite("true_load", true_load)
    price = _check_finite("price", price)
    g = np.maximum(true_load + charge - discharge, 0.0)
    bill = float(np.sum(price * g))
    if demand_charge > 0:
        bill += demand_charge * float(np.max(g))
    return bill


def no_battery_bill(true_load, price, *, demand_charge: float = 0.0) -> float:
    """The naive ceiling: serve the load straight from the grid, no battery at all."""
    true_load = _check_finite("true_load", true_load)
    price = _check_finite("price", price)
    g = np.maximum(true_load, 0.0)
    bill = float(np.sum(price * g))
    if demand_charge > 0:
        bill += demand_charge * float(np.max(g))
    return bill


# ------------------------------------------------------------------ value framing


def evaluate_episode(
    point,
    true_load,
    price,
    battery: BatteryParams,
    *,
    samples=None,
    demand_charge: float = 0.0,
) -> dict:
    """Price one (window, channel) episode end-to-end.

    Parameters
    ----------
    point : array (tau,)
        Point forecast of the future load (the deterministic plan's single scenario).
    true_load : array (tau,)
        The realised future load every schedule is settled against.
    price : array (tau,)
        Per-step time-of-use price.
    battery : BatteryParams
    samples : array (S, tau) or None
        Sample paths for the stochastic SAA plan; ``None`` -> deterministic only.
    demand_charge : float, default 0.0

    Returns
    -------
    dict
        ``bill_naive`` (ceiling), ``bill_oracle`` (lower bound), ``bill_det`` and
        ``saved_det`` / ``frac_det`` (vs naive, as a fraction of the attainable saving);
        and, when ``samples`` is given, ``bill_sto`` / ``saved_sto`` / ``frac_sto`` plus
        ``value_of_dist = bill_det - bill_sto``.
    """
    point = _check_finite("point", point)
    true_load = _check_finite("true_load", true_load)
    price = _check_finite("price", price)

    naive = no_battery_bill(true_load, price, demand_charge=demand_charge)
    # Oracle: dispatch on the true future itself -> lower bound on the realised bill.
    plan_oracle = dispatch_lp(true_load, price, battery, demand_charge=demand_charge)
    bill_oracle = realized_bill(
        plan_oracle["charge"], plan_oracle["discharge"], true_load, price, demand_charge=demand_charge
    )
    plan_det = dispatch_lp(point, price, battery, demand_charge=demand_charge)
    bill_det = realized_bill(
        plan_det["charge"], plan_det["discharge"], true_load, price, demand_charge=demand_charge
    )

    attainable = naive - bill_oracle
    out = {
        "bill_naive": naive,
        "bill_oracle": bill_oracle,
        "bill_det": bill_det,
        "saved_det": naive - bill_det,
        "frac_det": (naive - bill_det) / attainable if attainable > 1e-12 else float("nan"),
    }

    if samples is not None:
        samples = np.atleast_2d(_check_finite("samples", samples))
        plan_sto = dispatch_lp(samples, price, battery, demand_charge=demand_charge)
        bill_sto = realized_bill(
            plan_sto["charge"], plan_sto["discharge"], true_load, price, demand_charge=demand_charge
        )
        out["bill_sto"] = bill_sto
        out["saved_sto"] = naive - bill_sto
        out["frac_sto"] = (naive - bill_sto) / attainable if attainable > 1e-12 else float("nan")
        out["value_of_dist"] = bill_det - bill_sto
    return out


def evaluate_value(
    points,
    truths,
    prices,
    battery: BatteryParams,
    *,
    samples=None,
    demand_charge: float = 0.0,
) -> dict:
    """Aggregate :func:`evaluate_episode` over many episodes (window × channel pairs).

    Parameters
    ----------
    points, truths, prices : array (M, tau)
        One row per episode. ``prices`` is per-episode because windows can start at
        different hours-of-day (the time-of-use phase shifts).
    battery : BatteryParams
    samples : array (M, S, tau) or None
    demand_charge : float, default 0.0

    Returns
    -------
    dict
        ``per_episode`` (a dict of length-``M`` arrays), ``totals`` (bills summed across
        episodes, with the savings fractions recomputed on the sums so the headline is a
        *portfolio* bill, not a mean of ratios), and ``n_episodes``.
    """
    points = np.atleast_2d(_check_finite("points", points))
    truths = np.atleast_2d(_check_finite("truths", truths))
    prices = np.atleast_2d(_check_finite("prices", prices))
    m = points.shape[0]
    if not (truths.shape[0] == prices.shape[0] == m):
        raise ValueError("points, truths and prices must share the same number of episodes")
    has_samples = samples is not None
    if has_samples:
        samples = _check_finite("samples", samples)
        if samples.shape[0] != m:
            raise ValueError("samples must have one (S, tau) block per episode")

    cols = ["bill_naive", "bill_oracle", "bill_det", "saved_det", "frac_det"]
    if has_samples:
        cols += ["bill_sto", "saved_sto", "frac_sto", "value_of_dist"]
    acc = {c: np.full(m, np.nan) for c in cols}

    for i in range(m):
        ep = evaluate_episode(
            points[i],
            truths[i],
            prices[i],
            battery,
            samples=(samples[i] if has_samples else None),
            demand_charge=demand_charge,
        )
        for c in cols:
            acc[c][i] = ep[c]

    totals = {
        "bill_naive": float(np.nansum(acc["bill_naive"])),
        "bill_oracle": float(np.nansum(acc["bill_oracle"])),
        "bill_det": float(np.nansum(acc["bill_det"])),
    }
    attainable = totals["bill_naive"] - totals["bill_oracle"]
    totals["saved_det"] = totals["bill_naive"] - totals["bill_det"]
    totals["frac_det"] = totals["saved_det"] / attainable if attainable > 1e-12 else float("nan")
    if has_samples:
        totals["bill_sto"] = float(np.nansum(acc["bill_sto"]))
        totals["saved_sto"] = totals["bill_naive"] - totals["bill_sto"]
        totals["frac_sto"] = totals["saved_sto"] / attainable if attainable > 1e-12 else float("nan")
        totals["value_of_dist"] = totals["bill_det"] - totals["bill_sto"]

    return {"per_episode": acc, "totals": totals, "n_episodes": m}
