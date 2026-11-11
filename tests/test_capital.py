"""Detection d'un apport de capital — src/bourso/capital.py.

Aucun appel binaire ni reseau : la detection repose sur le cash lu par `trade summary`
compare au cash attendu (cash precedent +/- flux des ordres executes)."""

from datetime import date

import pytest

from src.bourso.capital import (
    get_entry, detect_deposit, record_expected_cash, deposit_threshold,
    DEPOSIT_MIN_EUR, DEPOSIT_TRADE_TOL,
)
from src.risk_off_strategy.strategy import rsi_wilder

D0 = date(2026, 10, 1)


def _entry():
    return get_entry({}, 1)


# ── RSI (information du signal) ──────────────────────────
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
    # ventes creditees nettes de 0.5%, achats debites avec 0.5% de frais (BUY_FEE)
    assert e["expected_cash"] == pytest.approx(9367.0 + 953 * 9.821 * 0.995 - 217 * 86.0 * 1.005, abs=0.01)
    assert e["traded_notional"] == pytest.approx(953 * 9.821 + 217 * 86.0, abs=0.01)
    # ecart de prix de remplissage (< 5% du notionnel) : pas un apport
    thr = deposit_threshold(e, 54000.0)
    assert thr == pytest.approx(DEPOSIT_TRADE_TOL * e["traded_notional"])
    assert detect_deposit(e, e["expected_cash"] + thr - 1, 54000.0, D0) == (0.0, 0.0)


def test_withdrawal_detected():
    e = _entry()
    record_expected_cash(e, 9400.0, [], D0)
    assert detect_deposit(e, 4400.0, 49000.0, D0) == (0.0, pytest.approx(5000.0))


# ── ancien etat DCA ───────────────────────────────────────
def test_legacy_dca_state_is_dropped():
    state = {"2": {"expected_cash": 100.0, "as_of": "2026-11-06", "traded_notional": 0.0,
                   "deposits": [], "dca": {"deposit": 20000.0, "released": 1.0, "done": True}}}
    e = get_entry(state, 2)
    assert "dca" not in e and e["expected_cash"] == 100.0
