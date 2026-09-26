"""Tests unitaires de compute_orders (PUST + LQQ, exposition plafonnee, bande asymetrique).

Aucun appel binaire ni reseau : on verifie la logique de decision d'ordre du chemin
LIVE (real_bourso.compute_orders), qui doit rester alignee sur le backtest net de
frais (strategy.simulate_net avec e_max=E_MAX) :
  - exposition cible E = min(2 . alloc, E_MAX), composition drag-minimale
    (E <= 1 : PUST + cash ; E > 1 : PUST = 2-E, LQQ = E-1, cash = 0)
  - achat (gratuit) seulement si delta expo >= BUY_THR_E (0.50)
  - vente (0.5%) seulement si delta expo <= -SELL_THR_E (1.00)
  - FORCE vers le cash (garde-fous macro / emergency, target=0) meme sous le seuil
  - FORCE depuis le cash total (premiere entree) meme sous le seuil d'achat
  - restructuration a expo constante quand le compte est levier avec du cash oisif
    (= migration immediate de l'ancienne realisation "LQQ + cash")
  - ventes avant achats ; achats plafonnes par le cash (+ produit des ventes)
"""

import pytest

from src.real_bourso import (
    compute_orders, exposure_after, detect_split, execute_order, target_exposure,
    BUY_THR_ALLOC, SELL_THR_ALLOC, BUY_THR_E, SELL_THR_E, SPLIT_DETECT_FACTOR,
    LIMIT_TOLERANCE_PCT, IDLE_CASH_TOL,
)
from src.risk_off_strategy.strategy import E_MAX, composition, simulate_net

PP, PL = 100.0, 10.0      # prix PUST / LQQ


def _state(pust=0, lqq=0, cash=0.0):
    """PEA fictif : 1 part PUST = 100 EUR, 1 part LQQ = 10 EUR."""
    positions = {"PUST": {"shares": pust, "price": PP, "value": pust * PP},
                 "LQQ": {"shares": lqq, "price": PL, "value": lqq * PL}}
    stocks = pust * PP + lqq * PL
    return dict(cash=cash, stocks=stocks, equity=cash + stocks, positions=positions,
                other_stocks=0.0)


def _orders(plan):
    return [(o["instrument"], o["side"], o["quantity"]) for o in plan["orders"]]


def _apply(plan, state):
    for o in plan["orders"]:
        o["status"] = "dry-run"
    return exposure_after(plan, state)


# ── source unique ─────────────────────────────────────────
def test_thresholds_match_strategy():
    assert (BUY_THR_ALLOC, SELL_THR_ALLOC) == (0.25, 0.50)
    assert (BUY_THR_E, SELL_THR_E) == (0.50, 1.00)
    assert E_MAX == 1.7


def test_target_exposure_capped():
    assert target_exposure(0.5) == pytest.approx(1.0)
    assert target_exposure(0.8) == pytest.approx(1.6)
    assert target_exposure(1.0) == pytest.approx(E_MAX)
    assert target_exposure(0.0) == 0.0


def test_composition_drag_minimal():
    assert composition(0.6) == pytest.approx((0.6, 0.0, 0.4))
    assert composition(1.0) == pytest.approx((1.0, 0.0, 0.0))
    assert composition(1.7) == pytest.approx((0.3, 0.7, 0.0))
    assert composition(2.0) == pytest.approx((0.0, 1.0, 0.0))


# ── bande asymetrique en exposition ───────────────────────
def test_buy_below_band_ignored():
    # PUST 60% + cash 40% (E=0.6), cible 1.0 -> +0.40 < 0.50 : on n'achete pas
    plan = compute_orders(1.0, _state(pust=600, cash=40000))
    assert plan["orders"] == [] and plan["mode"] is None
    assert "Achat ignore" in plan["reason"]


def test_buy_above_band_executes_into_composition():
    # E=0.6, cible 1.2 -> +0.60 >= 0.50 : composition PUST 80% + LQQ 20%, cash 0
    st = _state(pust=600, cash=40000)
    plan = compute_orders(1.2, st)
    assert plan["mode"] == "buy_band"
    assert _orders(plan) == [("PUST", "buy", 200), ("LQQ", "buy", 2000)]
    assert _apply(plan, st) == pytest.approx(1.2, abs=0.01)


def test_sell_below_band_ignored():
    # PUST 30% + LQQ 70% (E=1.7), cible 1.0 -> -0.70 < 1.00 : on encaisse le chop
    plan = compute_orders(1.0, _state(pust=300, lqq=7000))
    assert plan["orders"] == []
    assert "Vente ignoree" in plan["reason"]


def test_sell_above_band_executes():
    # E=1.7, cible 0.6 -> -1.10 >= 1.00 : vente LQQ totale + une partie de PUST, cash 40%
    st = _state(pust=300, lqq=7000)
    plan = compute_orders(0.6, st)
    assert plan["mode"] == "sell_band"
    sells = [o for o in plan["orders"] if o["side"] == "sell"]
    assert [(o["instrument"], o["quantity"]) for o in sells] == [("LQQ", 7000)]
    # PUST cible 60% de 100k = 60k -> on en detient 30k -> achat de PUST avec le produit
    buys = [o for o in plan["orders"] if o["side"] == "buy"]
    assert buys and buys[0]["instrument"] == "PUST"
    assert _apply(plan, st) == pytest.approx(0.6, abs=0.01)


def test_macro_off_forces_full_liquidation():
    """target=0 (garde-fous macro / emergency) DOIT tout liquider meme si -0.3 < 1.0."""
    plan = compute_orders(0.0, _state(pust=300, cash=70000))
    assert plan["mode"] == "to_cash"
    assert _orders(plan) == [("PUST", "sell", 300)]


def test_macro_off_liquidates_both_instruments():
    plan = compute_orders(0.0, _state(pust=300, lqq=7000))
    assert sorted(_orders(plan)) == [("LQQ", "sell", 7000), ("PUST", "sell", 300)]


def test_entry_from_full_cash_below_buy_band():
    """Premiere entree depuis le cash total : s'execute meme sous le seuil d'achat."""
    plan = compute_orders(0.10, _state(cash=100000))
    assert plan["mode"] == "from_cash"
    assert _orders(plan) == [("PUST", "buy", 100)]


def test_flat_cash_target_zero_no_action():
    plan = compute_orders(0.0, _state(cash=100000))
    assert plan["orders"] == []


# ── migration LQQ+cash -> PUST+LQQ (restructuration a expo constante) ─────────
def test_legacy_lqq_cash_is_restructured_at_constant_exposure():
    # ancien live : LQQ 83% + cash 17% (E=1.66), cible 1.57 dans la bande
    st = _state(lqq=8300, cash=17000)
    plan = compute_orders(1.57, st)
    assert plan["mode"] == "restructure"
    sells = [o for o in plan["orders"] if o["side"] == "sell"]
    buys = [o for o in plan["orders"] if o["side"] == "buy"]
    assert sells[0]["instrument"] == "LQQ" and buys[0]["instrument"] == "PUST"
    # expo inchangee (1.66 <= E_MAX), plus de cash oisif
    assert _apply(plan, st) == pytest.approx(1.66, abs=0.01)
    assert plan["target_weights"]["cash"] == 0.0


def test_restructure_capped_at_e_max():
    # LQQ 90% + cash 10% : E=1.8 > E_MAX -> restructuration a 1.7
    st = _state(lqq=9000, cash=10000)
    plan = compute_orders(1.7, st)
    assert plan["mode"] == "restructure"
    assert _apply(plan, st) == pytest.approx(E_MAX, abs=0.01)


def test_no_restructure_when_cash_is_residual():
    # PUST 30% + LQQ 68% + cash 2% (< IDLE_CASH_TOL) -> rien
    st = _state(pust=300, lqq=6800, cash=2000)
    plan = compute_orders(1.7, st)
    assert plan["orders"] == []
    assert IDLE_CASH_TOL == 0.05


def test_no_restructure_below_leverage():
    # PUST 60% + cash 40% (E=0.6) : le cash est voulu (composition E<=1) -> rien
    plan = compute_orders(0.7, _state(pust=600, cash=40000))
    assert plan["orders"] == []


# ── DCA : reserve + achat sans bande ──────────────────────
def test_reserved_cash_is_excluded_from_equity_and_buys():
    # 100k investis a E=1.7 + apport 20k entierement reserve -> aucun cash oisif
    st = _state(pust=300, lqq=7000, cash=20000)
    plan = compute_orders(1.7, st, reserved=20000)
    assert plan["orders"] == []
    assert plan["equity"] == pytest.approx(100000)


def test_force_buy_deploys_released_tranche_without_band():
    # tranche liberee 20% (reserve 16k) : delta +0.07 < 0.50 mais force_buy -> achats
    st = _state(pust=300, lqq=7000, cash=20000)
    plan = compute_orders(1.7, st, reserved=16000, force_buy=True)
    assert plan["mode"] == "force_buy"
    assert all(o["side"] == "buy" for o in plan["orders"])
    spent = sum(o["value"] for o in plan["orders"])
    assert spent <= 4000 + 1e-6 and spent > 3800      # ~la tranche, a l'arrondi pres


def test_buys_only_never_sells():
    st = _state(lqq=8300, cash=17000)
    plan = compute_orders(1.57, st, force_buy=True, buys_only=True)
    assert all(o["side"] == "buy" for o in plan["orders"])


# ── coherence avec le backtest net ───────────────────────
def test_live_decision_matches_simulate_net_band():
    """Meme regle de bande que simulate_net(e_max=E_MAX) : sur une serie de cibles,
    les jours de rebalancement du backtest sont exactement ceux ou compute_orders
    passe des ordres (prix constants, donc pas de derive entre deux jours)."""
    import numpy as np
    price = np.full(12, 100.0)
    alloc = np.array([0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.85, 0.2, 0.2, 0.0, 0.0, 0.5])
    _, _, revis = simulate_net(price, alloc, leverage=2, funding=np.zeros(12), e_max=E_MAX)
    st = _state(cash=100000)
    live_revis = 0
    for a in alloc[:-1]:                      # exec_lag=1 : la cible du jour t agit en t+1
        plan = compute_orders(target_exposure(a), st)
        if plan["orders"]:
            live_revis += 1
            for o in plan["orders"]:
                p = st["positions"][o["instrument"]]
                if o["side"] == "buy":
                    p["shares"] += o["quantity"]; st["cash"] -= o["value"]
                else:
                    p["shares"] -= o["quantity"]; st["cash"] += o["value"] * 0.995
                p["value"] = p["shares"] * p["price"]
            st["stocks"] = sum(p["value"] for p in st["positions"].values())
            st["equity"] = st["stocks"] + st["cash"]
    assert live_revis == round(revis * 12 / 252)


# ── detection de split ────────────────────────────────────
def test_split_factor_is_1_5():
    assert SPLIT_DETECT_FACTOR == 1.5


def test_no_split_on_normal_move():
    assert detect_split(100.0, 105.0) is None


def test_no_split_on_extreme_x2_day():
    assert detect_split(100.0, 130.0) is None


def test_split_forward_200_for_1():
    r = detect_split(1954.0, 9.88)
    assert r is not None and r > 190


def test_split_2_for_1():
    assert detect_split(100.0, 50.0) is not None


def test_reverse_split_detected():
    assert detect_split(10.0, 20.0) is not None


def test_no_reference_no_detection():
    assert detect_split(None, 9.88) is None
    assert detect_split(0.0, 9.88) is None


# ── ordre limite avec tolerance ───────────────────────────
def test_limit_tolerance_default():
    assert LIMIT_TOLERANCE_PCT == 3.0


def test_execute_order_dryrun_carries_limit_tolerance():
    r = execute_order("PUST", "buy", 3, dry_run=True)
    assert r["order_type"] == "LIM" and r["tolerance"] == LIMIT_TOLERANCE_PCT
    assert r["instrument"] == "PUST"
