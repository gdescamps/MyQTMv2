"""Tests unitaires de compute_orders (bande de non-action asymetrique).

Aucun appel binaire ni reseau : on verifie la logique de decision d'ordre du
chemin LIVE (real_bourso.compute_orders), qui doit rester alignee sur le backtest
net de frais (strategy.simulate_net) :
  - achat (gratuit) seulement si delta >= BUY_THR_ALLOC (0.25)
  - vente (0.5%) seulement si delta <= -SELL_THR_ALLOC (0.50)
  - FORCE vers le cash (garde-fous macro / emergency, target=0) meme sous le seuil
  - FORCE depuis le cash total (premiere entree) meme sous le seuil d'achat
"""

from src.real_bourso import compute_orders, BUY_THR_ALLOC, SELL_THR_ALLOC


def _state(shares, price=100.0, equity=100000.0, cash=100000.0):
    """PEA fictif : equity 100k, prix 100 -> 1 part = 1% d'allocation."""
    return dict(equity=equity, etf_price=price, etf_shares=shares, cash=cash)


def test_thresholds_match_strategy():
    """Source unique : les seuils live == ceux de la strategie backtestee."""
    assert (BUY_THR_ALLOC, SELL_THR_ALLOC) == (0.25, 0.50)


def test_buy_below_band_ignored():
    # detient 46%, cible 55% -> +9% < 25% : on n'achete pas
    side, qty, _ = compute_orders(0.55, _state(460))
    assert side is None and qty == 0


def test_buy_above_band_executes():
    # detient 46%, cible 75% -> +29% >= 25% : achat vers la cible
    side, qty, _ = compute_orders(0.75, _state(460))
    assert side == "buy" and qty == 290


def test_sell_below_band_ignored():
    # detient 80%, cible 55% -> -25% < 50% : on ne vend pas (on encaisse le chop)
    side, qty, _ = compute_orders(0.55, _state(800))
    assert side is None and qty == 0


def test_sell_above_band_executes():
    # detient 80%, cible 25% -> -55% >= 50% : vente
    side, qty, _ = compute_orders(0.25, _state(800))
    assert side == "sell" and qty == 550


def test_macro_off_forces_full_liquidation():
    """target=0 (garde-fous macro / emergency) DOIT liquider meme si -30% < 50%."""
    side, qty, _ = compute_orders(0.0, _state(300))
    assert side == "sell" and qty == 300


def test_entry_from_full_cash_below_buy_band():
    """Premiere entree depuis le cash total : s'execute meme sous le seuil d'achat."""
    side, qty, _ = compute_orders(0.10, _state(0))
    assert side == "buy" and qty == 100


def test_flat_cash_target_zero_no_action():
    side, qty, _ = compute_orders(0.0, _state(0))
    assert side is None and qty == 0
