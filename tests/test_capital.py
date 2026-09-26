"""Detection d'un apport de capital + deploiement progressif (DCA / RSI) — src/bourso/capital.py.

Aucun appel binaire ni reseau : la detection repose sur le cash lu par `trade summary`
compare au cash attendu (cash precedent +/- flux des ordres executes)."""

from datetime import date

import pytest

from src.bourso.capital import (
    get_entry, detect_deposit, record_expected_cash, start_dca, dca_step, reserved_cash,
    dca_summary, apply_withdrawal, deposit_threshold, DEPOSIT_MIN_EUR, DEPOSIT_TRADE_TOL,
)
from src.risk_off_strategy.strategy import (
    dca_tranche, DCA_RSI_GATE, DCA_BASE, DCA_SLOPE, DCA_MAX_WEEKS, rsi_wilder,
)

D0 = date(2026, 10, 1)


def _entry():
    return get_entry({}, 1)


# ── tranche RSI ───────────────────────────────────────────
def test_tranche_zero_above_gate():
    assert dca_tranche(50.0) == 0.0
    assert dca_tranche(64.4) == 0.0


def test_tranche_grows_when_rsi_falls():
    assert dca_tranche(49.0) == pytest.approx(DCA_BASE + DCA_SLOPE)
    assert dca_tranche(40.0) == pytest.approx(0.40)
    assert dca_tranche(30.0) == pytest.approx(0.60)
    assert dca_tranche(0.0) == 1.0                       # borne


def test_tranche_unknown_rsi_is_base():
    assert dca_tranche(None) == DCA_BASE
    assert (DCA_RSI_GATE, DCA_BASE, DCA_SLOPE, DCA_MAX_WEEKS) == (50.0, 0.20, 0.02, 26)


def test_rsi_wilder_bounds():
    import numpy as np
    up = np.linspace(100, 200, 60)
    assert rsi_wilder(up)[-1] > 90
    assert rsi_wilder(up[::-1])[-1] < 10
    assert rsi_wilder(np.full(30, 100.0))[-1] == 50.0   # sans mouvement -> 50


# ── detection ─────────────────────────────────────────────
def test_first_read_only_initialises_reference():
    e = _entry()
    assert detect_deposit(e, 9400.0, 54000.0, D0) == (0.0, 0.0)
    record_expected_cash(e, 9400.0, [], D0)
    assert e["expected_cash"] == 9400.0


def test_deposit_detected_from_unexplained_cash():
    e = _entry()
    record_expected_cash(e, 9400.0, [], D0)
    dep, wd = detect_deposit(e, 14400.0, 59000.0, D0)
    assert dep == pytest.approx(5000.0) and wd == 0.0
    assert e["deposits"][-1]["amount"] == 5000.0


def test_small_noise_is_not_a_deposit():
    e = _entry()
    record_expected_cash(e, 9400.0, [], D0)
    assert detect_deposit(e, 9400.0 + DEPOSIT_MIN_EUR - 1, 54000.0, D0) == (0.0, 0.0)


def test_expected_cash_follows_executed_orders():
    e = _entry()
    orders = [{"side": "sell", "instrument": "LQQ", "quantity": 953, "price": 9.821, "status": "executed"},
              {"side": "buy", "instrument": "PUST", "quantity": 217, "price": 86.0, "status": "executed"},
              {"side": "buy", "instrument": "LQQ", "quantity": 10, "price": 9.821, "status": "error"}]
    record_expected_cash(e, 9367.0, orders, D0)
    assert e["expected_cash"] == pytest.approx(9367.0 + 953 * 9.821 * 0.995 - 217 * 86.0, abs=0.01)
    assert e["traded_notional"] == pytest.approx(953 * 9.821 + 217 * 86.0, abs=0.01)
    # ecart de prix de remplissage (< 5% du notionnel) : pas un apport
    thr = deposit_threshold(e, 54000.0)
    assert thr == pytest.approx(DEPOSIT_TRADE_TOL * e["traded_notional"])
    assert detect_deposit(e, e["expected_cash"] + thr - 1, 54000.0, D0) == (0.0, 0.0)


def test_withdrawal_detected():
    e = _entry()
    record_expected_cash(e, 9400.0, [], D0)
    assert detect_deposit(e, 4400.0, 49000.0, D0) == (0.0, pytest.approx(5000.0))


# ── DCA ───────────────────────────────────────────────────
def test_dca_first_tranche_on_detection_day_if_rsi_low():
    e = _entry()
    start_dca(e, 10000.0, D0)
    assert reserved_cash(e, 19400.0) == 10000.0
    tr = dca_step(e, 40.0, D0)                          # RSI 40 -> 0.40
    assert tr == pytest.approx(0.40)
    assert reserved_cash(e, 19400.0) == pytest.approx(6000.0)
    # pas de nouvelle tranche avant 7 jours
    assert dca_step(e, 30.0, date(2026, 10, 6)) == 0.0
    assert dca_step(e, 30.0, date(2026, 10, 8)) == pytest.approx(0.60)
    assert e["dca"]["done"] is True
    assert reserved_cash(e, 19400.0) == 0.0


def test_dca_waits_while_rsi_high_then_safety_net():
    e = _entry()
    start_dca(e, 10000.0, D0)
    d = D0
    for w in range(DCA_MAX_WEEKS - 1):
        assert dca_step(e, 70.0, d) == 0.0
        d = date.fromordinal(d.toordinal() + 7)
    assert reserved_cash(e, 10000.0) == 10000.0        # rien libere tant que RSI >= 50
    assert dca_step(e, 70.0, d) == pytest.approx(1.0)  # filet : 26e semaine -> tout
    assert e["dca"]["done"] is True


def test_dca_summary_and_next_tranche():
    e = _entry()
    start_dca(e, 5000.0, D0)
    dca_step(e, 45.0, D0)
    s = dca_summary(e, 5000.0)
    assert s["deposit"] == 5000.0 and s["released"] == pytest.approx(0.30)
    assert s["reserved"] == pytest.approx(3500.0)
    assert s["next_tranche"] == "2026-10-08"
    assert dca_summary(_entry(), 0.0) is None


def test_second_deposit_merges_into_running_dca():
    e = _entry()
    start_dca(e, 10000.0, D0)
    dca_step(e, 40.0, D0)                               # libere 40% -> reserve 6000
    start_dca(e, 4000.0, date(2026, 10, 2))             # reserve -> 10000 sur 14000
    assert e["dca"]["deposit"] == 14000.0
    assert reserved_cash(e, 20000.0) == pytest.approx(10000.0)


def test_withdrawal_reduces_reserve_first():
    e = _entry()
    start_dca(e, 10000.0, D0)
    apply_withdrawal(e, 4000.0)
    assert reserved_cash(e, 6000.0) == pytest.approx(6000.0)
    apply_withdrawal(e, 8000.0)
    assert e["dca"]["done"] is True and reserved_cash(e, 0.0) == 0.0


def test_reserve_never_exceeds_cash():
    e = _entry()
    start_dca(e, 10000.0, D0)
    assert reserved_cash(e, 2500.0) == 2500.0
