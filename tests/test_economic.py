"""Hand-checkable correctness for the E6 battery-dispatch LP (plan §3.5, §6.4).

These run on tiny, pen-and-paper examples — NO network, NO GPU, NO trained model — so
any teammate can confirm the economic pillar's optimiser is sound in seconds:

    python -m pytest tests/test_economic.py     # if pytest is installed
    python tests/test_economic.py               # plain-python fallback

What they pin:
* the LP does nothing when there is nothing to arbitrage (zero load -> €0);
* a worked two-step arbitrage (eta=1 -> bill 2; eta=0.9 -> bill 3);
* state-of-charge stays feasible and the battery ends no emptier than it started;
* the **value-of-distribution invariant**: the stochastic (SAA) schedule never costs
  more, on its own scenario set, than the deterministic schedule built on the mean —
  the formal reason a calibrated distribution can beat a point forecast here;
* oracle (perfect foresight) is a true lower bound, naive (no battery) a true ceiling;
* the time-of-use price map and the NaN guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.economic import (  # noqa: E402
    BatteryParams,
    Tariff,
    dispatch_lp,
    evaluate_episode,
    evaluate_value,
    no_battery_bill,
    realized_bill,
    tou_price,
)

TOL = 1e-4


def _assert_raises(exc, fn, *args, **kwargs) -> None:
    """Plain-python `pytest.raises` so the fallback runner (which only catches
    AssertionError) still reports a clean PASS/FAIL instead of erroring out."""
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}, none raised")


# --------------------------------------------------------------------------- basics


def test_zero_load_does_nothing() -> None:
    """With no load and no export there is nothing to arbitrage: the LP idles at €0."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=1.0)
    price = np.array([1.0, 10.0, 1.0, 10.0])
    plan = dispatch_lp(np.zeros(4), price, batt)
    assert abs(plan["planned_bill"]) < TOL
    assert np.allclose(plan["charge"], 0.0, atol=TOL)
    assert np.allclose(plan["discharge"], 0.0, atol=TOL)


def test_two_step_arbitrage_eta1() -> None:
    """load=[1,1], price=[1,10], 1-unit/1-power battery, eta=1.

    Charge 1 unit in the cheap hour (import 2 @ €1), discharge it in the dear hour
    (import 0 @ €10): planned bill 2 vs naive 11. Worked by hand in the module header.
    """
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=1.0, soc0=0.0)
    load = np.array([1.0, 1.0])
    price = np.array([1.0, 10.0])
    plan = dispatch_lp(load, price, batt)
    assert abs(plan["planned_bill"] - 2.0) < TOL

    bill = realized_bill(plan["charge"], plan["discharge"], load, price)
    naive = no_battery_bill(load, price)
    assert abs(bill - 2.0) < TOL
    assert abs(naive - 11.0) < TOL
    assert abs((naive - bill) - 9.0) < TOL  # money saved vs naive


def test_efficiency_costs_money() -> None:
    """Same arbitrage but eta=0.9: the round-trip loss lifts the bill 2 -> 3."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=0.9, soc0=0.0)
    load = np.array([1.0, 1.0])
    price = np.array([1.0, 10.0])
    plan = dispatch_lp(load, price, batt)
    assert abs(plan["planned_bill"] - 3.0) < TOL


def test_soc_feasible_and_terminal() -> None:
    """SoC stays within [0, e_max], controls within [0, p_max], final SoC >= soc0."""
    batt = BatteryParams(e_max=2.0, p_max=1.5, eta=0.95, soc0=0.5)
    load = np.array([1.0, 2.0, 0.5, 3.0, 1.0])
    price = np.array([1.0, 1.0, 5.0, 5.0, 1.0])
    plan = dispatch_lp(load, price, batt)
    assert plan["charge"].min() >= -TOL and plan["charge"].max() <= batt.p_max + TOL
    assert plan["discharge"].min() >= -TOL and plan["discharge"].max() <= batt.p_max + TOL
    assert plan["soc"].min() >= -TOL and plan["soc"].max() <= batt.e_max + TOL
    assert plan["soc"][-1] >= batt.soc0 - TOL


# ------------------------------------------------------- value-of-distribution (SAA)


def test_stochastic_not_worse_than_deterministic_on_scenarios() -> None:
    """The core SAA guarantee.

    The stochastic schedule minimises the sample-average bill over its scenario set, so
    its average bill over those same scenarios cannot exceed the deterministic
    schedule's (the deterministic schedule is just another feasible point). This is the
    formal reason a full predictive distribution can be worth real money here.
    """
    batt = BatteryParams(e_max=2.0, p_max=2.0, eta=1.0, soc0=0.0)
    price = np.array([1.0, 5.0])
    scen = np.array([[1.0, 3.0], [1.0, 0.0]])  # spread in the expensive hour
    mean = scen.mean(axis=0)

    det = dispatch_lp(mean, price, batt)
    sto = dispatch_lp(scen, price, batt)

    avg_det = np.mean([realized_bill(det["charge"], det["discharge"], s, price) for s in scen])
    avg_sto = np.mean([realized_bill(sto["charge"], sto["discharge"], s, price) for s in scen])

    assert avg_sto <= avg_det + TOL
    # the LP's reported objective should equal that realised sample-average
    assert abs(sto["planned_bill"] - avg_sto) < 1e-3


def test_oracle_is_lower_bound_naive_is_ceiling() -> None:
    """Oracle (perfect-foresight LP) <= any committed schedule's realised bill, and the
    no-battery naive plan is also bounded below by the oracle."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=0.9, soc0=0.0)
    point = np.array([1.0, 2.0, 1.0, 0.0])
    truth = np.array([1.0, 1.0, 1.0, 1.0])
    price = np.array([1.0, 1.0, 5.0, 1.0])
    ep = evaluate_episode(point, truth, price, batt)
    assert ep["bill_oracle"] <= ep["bill_det"] + TOL
    assert ep["bill_oracle"] <= ep["bill_naive"] + TOL


# ------------------------------------------------------------------- API + guards


def test_tou_price_map() -> None:
    """Hour-of-day -> off-peak / shoulder / peak under the default tariff."""
    t = Tariff()
    got = tou_price([0, 12, 18, 23], t)
    assert np.allclose(got, [t.offpeak, t.shoulder, t.peak, t.shoulder])


def test_evaluate_episode_reports_value_of_distribution() -> None:
    """With samples supplied the episode dict exposes the probabilistic figures."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=1.0, soc0=0.0)
    point = np.array([1.0, 1.0])
    truth = np.array([1.0, 1.0])
    samples = np.array([[1.0, 0.5], [1.0, 1.5], [1.0, 1.0]])
    price = np.array([1.0, 10.0])
    ep = evaluate_episode(point, truth, price, batt, samples=samples)
    for key in ("bill_sto", "saved_sto", "frac_sto", "value_of_dist"):
        assert key in ep
    assert abs(ep["value_of_dist"] - (ep["bill_det"] - ep["bill_sto"])) < TOL


def test_evaluate_value_totals_are_consistent() -> None:
    """Aggregation sums bills and recomputes savings on the sums."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=1.0, soc0=0.0)
    points = np.array([[1.0, 1.0], [2.0, 2.0]])
    truths = np.array([[1.0, 1.0], [2.0, 2.0]])
    prices = np.array([[1.0, 10.0], [1.0, 10.0]])
    out = evaluate_value(points, truths, prices, batt)
    assert out["n_episodes"] == 2
    assert len(out["per_episode"]["bill_det"]) == 2
    t = out["totals"]
    assert abs(t["saved_det"] - (t["bill_naive"] - t["bill_det"])) < TOL
    assert t["bill_oracle"] <= t["bill_det"] + TOL


def test_nan_input_is_refused() -> None:
    """A NaN forecast must raise, not silently fabricate a schedule (plan §6.6)."""
    batt = BatteryParams(e_max=1.0, p_max=1.0, eta=1.0)
    _assert_raises(ValueError, dispatch_lp, np.array([1.0, np.nan]), np.array([1.0, 1.0]), batt)
    _assert_raises(ValueError, realized_bill, np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                   np.array([1.0, np.inf]), np.array([1.0, 1.0]))


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
