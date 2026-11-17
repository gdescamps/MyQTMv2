"""Garde-fou lecture PEA incomplete (incident 2026-11-17, compte 2 : titres 48 999 EUR
mais aucune ligne PUST dans `trade summary` -> achat parasite de 12 PUST)."""
from unittest import mock

import pytest

import src.real_bourso as R
import src.bourso.prepare as P
import src.bourso.accounts as A
import src.bourso.quote as Q


def _pos(qty, last):
    return {"quantity": qty, "last_price": last, "label": "x", "amount": qty * last, "currency": "EUR"}


def _state(summary):
    acc = A.BoursoAccount(slot=2, bourso_id="i", bourso_code="c", name="PEA X")
    with mock.patch.object(P, "get_account_summary", return_value=summary), \
         mock.patch.object(A, "resolve_pea", lambda a: None), \
         mock.patch.object(Q, "get_quote", return_value={"last": 99.42}):
        return R.get_pea_state(acc, quiet=True)


def test_missing_position_line_is_refused():
    # 17/11 : valorisation 48 999 EUR, liste de positions vide
    summary = {"account": {"name": "PEA X", "cash": 3159.26, "stocks": 48999.27}, "positions": {}}
    with pytest.raises(R.IncompleteSummary, match="lecture PEA incomplete"):
        _state(summary)


def test_complete_summary_passes():
    summary = {"account": {"name": "PEA X", "cash": 1966.58, "stocks": 50085.9},
               "positions": {"1rTPUST": _pos(505, 99.18)}}
    st = _state(summary)
    assert st["positions"]["PUST"]["shares"] == 505
    assert st["other_stocks"] < 1


def test_other_listed_line_is_not_an_incoherence():
    # une vraie ligne hors strategie (listee) reste acceptee et ignoree
    summary = {"account": {"name": "PEA X", "cash": 1000.0, "stocks": 50085.9 + 5000.0},
               "positions": {"1rTPUST": _pos(505, 99.18), "1rPAIR": _pos(50, 100.0)}}
    st = _state(summary)
    assert st["other_stocks"] == pytest.approx(5000.0, abs=1)


def test_small_gap_is_tolerated():
    # decalage cours valorisation / ligne : sous max(100 EUR, 1% equity)
    R.check_summary_consistency(50085.9 + 80, {"1rTPUST": _pos(505, 99.18)}, 52000)
    with pytest.raises(R.IncompleteSummary):
        R.check_summary_consistency(50085.9 + 800, {"1rTPUST": _pos(505, 99.18)}, 52000)
